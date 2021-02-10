import os
import math
import random
import hashlib
import augments
import numpy as np
from tqdm import tqdm
from constant import *
import tensorflow as tf
from tensorflow.python.util import compat
from utils import set_seed

set_seed()


class AudioLoader:
    def __init__(self, data_dir, silence_percentage, negative_percentage,
                 validation_percentage, model_settings, augment_dir=None):
        self.data_dir = data_dir
        self.augment_dir = augment_dir
        self.silence_percentage = silence_percentage
        self.negative_percentage = negative_percentage
        self.validation_percentage = validation_percentage
        self.model_settings = model_settings

        self.data_index = {'validation': [], 'training': []}
        self.word_to_index = {}
        self.background_data = []

        self.total_sample_count = self.prepare_data_index()
        self.prepare_background_data()
        self.mfcc_input_, self.mfcc_ = self.prepare_processing_graph()

    @staticmethod
    def which_set(filename, word, validation_percentage):
        base_name = os.path.basename(filename)

        if word == POSITIVE_LABEL or word == AUGMENT_POSITIVE_LABEL:
            hash_name = base_name.split('_')[0]
        else:
            hash_name = base_name
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
                           * (100.0 / MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'validation'
        else:
            result = 'training'
        return result

    def prepare_data_index(self):
        negative_index = {'validation': [], 'training': []}
        real_negative_index = {'validation': [], 'training': []}
        all_words = {}

        # Look through all the sub folders to find audio samples
        all_wav_paths = tf.io.gfile.glob(os.path.join(self.data_dir, '*', '*.wav')) \
                        + tf.io.gfile.glob(os.path.join(self.data_dir, '{}/*/*.wav'.format(POSITIVE_LABEL))) \
                        + tf.io.gfile.glob(os.path.join(self.data_dir, '{}/*/*.wav'.format(VOCAL_WORD_LABEL)))
        for wav_path in tqdm(all_wav_paths):
            if (POSITIVE_LABEL in wav_path and AUGMENT_POSITIVE_LABEL not in wav_path) or VOCAL_WORD_LABEL in wav_path:
                word = os.path.dirname(wav_path).split('/')[-2]
            else:
                _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = self.which_set(wav_path, word, self.validation_percentage)
            if word == POSITIVE_LABEL or word == AUGMENT_POSITIVE_LABEL:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            elif word == VOCAL_WORD_LABEL:
                negative_index[set_index].append({'label': word, 'file': wav_path})
            elif word == REAL_NEGATIVE_LABEL:
                real_negative_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at {}'.format(self.data_dir))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({'label': SILENCE_LABEL, 'file': silence_wav_path})

            # Pick some negative to add to each partition of the data set.
            random.shuffle(negative_index[set_index])
            if self.negative_percentage > 0:
                negative_size = int(math.ceil(set_size * self.negative_percentage / 100))
                self.data_index[set_index].extend(negative_index[set_index][:negative_size])
            else:
                self.data_index[set_index].extend(negative_index[set_index])

            # Pick real silence to add to each partition of the data set.
            random.shuffle(real_negative_index[set_index])
            self.data_index[set_index].extend(real_negative_index[set_index])

        # Prepare the rest of the result data structure.
        for word in all_words:
            if word == POSITIVE_LABEL or word == AUGMENT_POSITIVE_LABEL:
                self.word_to_index[word] = POSITIVE_WORD_INDEX
            else:
                self.word_to_index[word] = NEGATIVE_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = NEGATIVE_WORD_INDEX

        # Make sure the ordering is random.
        print('-----')
        total_count_label = {}
        for set_index in ['validation', 'training']:
            random.shuffle(self.data_index[set_index])

            # count label
            count_label = {}
            samples = self.data_index[set_index]
            for sample in samples:
                label = sample['label']
                if label in count_label:
                    count_label[label] += 1
                else:
                    count_label[label] = 1
                if label in total_count_label:
                    total_count_label[label] += 1
                else:
                    total_count_label[label] = 1
            print('Set index:', set_index, ' - Count:', sorted(count_label.items()),
                  ' - Sum:', sum(count_label.values()))
            sum_total_count_label = sum(total_count_label.values())
        print('Total count:', sorted(total_count_label.items()), ' - Sum:', sum_total_count_label)
        print('-----')
        result_count = {'count_{}'.format(count_item): total_count_label[count_item]
                        for count_item in total_count_label}
        result_count['count_sum'] = sum_total_count_label
        return result_count

    def prepare_background_data(self):
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
            wav_loader = tf.io.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
            for wav_path in tf.io.gfile.glob(search_path):
                wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in {}'.format(search_path))

    def prepare_processing_graph(self):
        desired_samples = self.model_settings['desired_samples']
        wav_filename_placeholder_ = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = tf.io.read_file(wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)

        # Allow the audio sample's volume to be adjusted.
        foreground_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder_)

        # Shift the sample's start position, and pad any gaps with zeros.
        time_shift_padding_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2, 2])
        time_shift_offset_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples, -1])

        # # Mix in background noise.
        background_data_placeholder_ = tf.compat.v1.placeholder(tf.float32, [desired_samples, 1])
        background_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [])
        background_mul = tf.multiply(background_data_placeholder_, background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)

        # Down volume
        down_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [])
        down_volume = tf.multiply(background_add, down_volume_placeholder_)
        background_clamp = tf.clip_by_value(down_volume, -1.0, 1.0)

        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = tf.raw_ops.AudioSpectrogram(input=background_clamp,
                                                  window_size=self.model_settings['window_size_samples'],
                                                  stride=self.model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
        mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=wav_decoder.sample_rate,
                                dct_coefficient_count=self.model_settings['dct_coefficient_count'])
        return {'wav_filename_placeholder_': wav_filename_placeholder_,
                'foreground_volume_placeholder_': foreground_volume_placeholder_,
                'time_shift_padding_placeholder_': time_shift_padding_placeholder_,
                'time_shift_offset_placeholder_': time_shift_offset_placeholder_,
                'background_data_placeholder_': background_data_placeholder_,
                'background_volume_placeholder_': background_volume_placeholder_,
                'down_volume_placeholder_': down_volume_placeholder_}, mfcc_

    def load_batch(self, sess, batch_size=100, offset=0,
                   background_frequency=0, background_silence_frequency=0,
                   background_silence_volume_range=0,
                   down_volume_frequency=0, down_volume_range=0,
                   time_shift=0, mode='training'):
        positive_count = 0
        negative_count = 0

        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if batch_size == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(batch_size, len(candidates) - offset))

        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, self.model_settings['fingerprint_size']))
        labels = np.zeros((sample_count, self.model_settings['label_count']))
        desired_samples = self.model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')

        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in range(offset, offset + sample_count):
            # Pick which audio sample to use.
            sample = candidates[i]

            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {self.mfcc_input_['wav_filename_placeholder_']: sample['file'],
                          self.mfcc_input_['time_shift_padding_placeholder_']: time_shift_padding,
                          self.mfcc_input_['time_shift_offset_placeholder_']: time_shift_offset}

            # Choose a section of background noise to mix in.
            sample_label = sample['label']
            augment_output = None
            if use_background:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(0, len(background_samples)
                                                      - self.model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])

                background_random = np.random.uniform(0, 1)
                if sample_label == SILENCE_LABEL and background_random < background_silence_frequency:
                    background_volume = np.random.uniform(0, background_silence_volume_range)
                elif sample_label != SILENCE_LABEL and sample_label != REAL_NEGATIVE_LABEL \
                        and background_random < background_frequency:
                    background_volume = 0
                    augment_output = augments.process_file(sample['file'],
                                                           os.path.join(self.augment_dir, sample_label),
                                                           background_noise=background_reshaped, max_no_chain=0)
                    input_dict[self.mfcc_input_['wav_filename_placeholder_']] = augment_output
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[self.mfcc_input_['background_data_placeholder_']] = background_reshaped
            input_dict[self.mfcc_input_['background_volume_placeholder_']] = background_volume

            # If we want silence, mute out the main sample but leave the background.
            if sample_label == SILENCE_LABEL:
                input_dict[self.mfcc_input_['foreground_volume_placeholder_']] = 0
            else:
                input_dict[self.mfcc_input_['foreground_volume_placeholder_']] = 1

            # Down volume
            down_volume_random = np.random.uniform(0, 1)
            if down_volume_random < down_volume_frequency:
                input_dict[self.mfcc_input_['down_volume_placeholder_']] = np.random.uniform(down_volume_range, 1)
            else:
                input_dict[self.mfcc_input_['down_volume_placeholder_']] = 1

            # Run the graph to produce the output audio.
            data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
            label_index = self.word_to_index[sample['label']]
            if label_index == POSITIVE_WORD_INDEX:
                positive_count += 1
            else:
                negative_count += 1
            labels[i - offset, label_index] = 1
            if augment_output:
                os.remove(augment_output)
        return data, labels, positive_count, negative_count

    def size(self, mode='training'):
        return len(self.data_index[mode])

    def shuffle(self, set_index='training'):
        random.shuffle(self.data_index[set_index])
