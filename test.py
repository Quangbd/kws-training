import time
import random
import librosa
import numpy as np
from utils import *
from config import *
import tensorflow as tf
from data import AudioLoader
from models import select_model


def test_batch():
    def _process(_):
        random.seed(RANDOM_SEED)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        sess = tf.compat.v1.InteractiveSession()

        # model
        wanted_words = args.wanted_words.split(',')
        model = select_model(len(prepare_words_list(wanted_words)), window_size_ms=args.window_size_ms,
                             window_stride_ms=args.window_stride_ms, dct_coefficient_count=args.dct_coefficient_count,
                             name=args.model_architecture)
        model_settings = model.prepare_model_settings()
        print('-----\nModel settings: {}'.format(model_settings))

        audio_loader = AudioLoader(args.data_dir, wanted_words, SILENCE_PERCENTAGE, UNKNOWN_PERCENTAGE,
                                   VALIDATION_PERCENTAGE, TESTING_PERCENTAGE, model_settings)
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']
        fingerprint_input = tf.compat.v1.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

        ground_truth_input = tf.compat.v1.placeholder(tf.float32, [None, label_count], name='groundtruth_input')
        logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info)

        predicted_indices = tf.argmax(logits, 1)
        expected_indices = tf.argmax(ground_truth_input, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        model.load_variables_from_checkpoint(sess, args.checkpoint)

        # test set
        test_size = audio_loader.size('testing')
        tf.compat.v1.logging.info('set_size=%d', test_size)
        total_accuracy = 0
        total_conf_matrix = None
        for i in range(0, test_size, args.batch_size):
            test_fingerprints, test_ground_truth = audio_loader \
                .load_batch(sess, args.batch_size, offset=i, background_frequency=0,
                            background_volume_range=0, time_shift=0, mode='testing')
            test_accuracy, test_matrix = sess.run(
                [evaluation_step, confusion_matrix],
                feed_dict={fingerprint_input: test_fingerprints,
                           ground_truth_input: test_ground_truth,
                           dropout_prob: 1.0})
            batch_size = min(args.batch_size, test_size - i)
            total_accuracy += (test_accuracy * batch_size) / test_size
            if total_conf_matrix is None:
                total_conf_matrix = test_matrix
            else:
                total_conf_matrix += test_matrix
        tf.compat.v1.logging.info('Final confusion matrix: \n %s' % total_conf_matrix)
        tf.compat.v1.logging.info('Final accuracy {}'.format(total_accuracy))
        sess.close()

    args = prepare_config()
    tf.compat.v1.app.run(main=_process)


def test_checkpoint():
    def _process(_):
        random.seed(RANDOM_SEED)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        sess = tf.compat.v1.InteractiveSession()

        labels = prepare_words_list(args.wanted_words.split(','))
        model = select_model(len(labels), window_size_ms=args.window_size_ms,
                             window_stride_ms=args.window_stride_ms, dct_coefficient_count=args.dct_coefficient_count,
                             name=args.model_architecture)
        model_settings = model.prepare_model_settings()
        print('-----\nModel settings: {}\n-----'.format(model_settings))

        wav_filename_placeholder_ = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = tf.io.read_file(wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1,
                                          desired_samples=model_settings['desired_samples'])
        spectrogram = tf.raw_ops.AudioSpectrogram(input=wav_decoder.audio,
                                                  window_size=model_settings['window_size_samples'],
                                                  stride=model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
        mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=wav_decoder.sample_rate,
                                dct_coefficient_count=model_settings['dct_coefficient_count'])
        fingerprint_frequency_size = model_settings['dct_coefficient_count']
        fingerprint_time_size = model_settings['spectrogram_length']
        fingerprint_input = tf.reshape(mfcc_, [-1, fingerprint_time_size * fingerprint_frequency_size])
        logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info, is_training=False)
        in_predicted_results = tf.nn.softmax(logits, name='labels_softmax')
        in_predicted_indices = tf.argmax(logits, 1)
        model.load_variables_from_checkpoint(sess, args.checkpoint)

        predicted_results, predicted_indices = sess.run(
            [in_predicted_results, in_predicted_indices],
            feed_dict={wav_filename_placeholder_: '/Users/quangbd/Documents/data/kws-data/'
                                                  'speech_commands_v0.02/yes/8134f43f_nohash_1.wav'})

        print('Result: {} {}'.format(labels[predicted_indices[0]], predicted_results[0][predicted_indices[0]]))
        sess.close()

    args = prepare_config()
    tf.compat.v1.app.run(main=_process)


def test_pb():
    audio, sr = librosa.load('/Users/quangbd/Documents/data/kws-data/speech_commands_v0.02/go/0b7ee1a0_nohash_0.wav',
                             sr=16000, duration=1)
    audio = np.reshape(audio, [16000, 1])

    args = prepare_config()
    with tf.compat.v1.gfile.FastGFile(args.output_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    with tf.compat.v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
        predictions = sess.run(softmax_tensor, {'decoded_sample_data:0': audio})[0]
        re_index = np.argmax(predictions)
        labels = prepare_words_list(args.wanted_words.split(','))
        print('Result: {} {}'.format(labels[re_index], predictions[re_index]))


def test_tflite():
    interpreter = tf.lite.Interpreter(
        model_path='/Users/quangbd/Documents/data/model/kws/speech_commands_v0.02/ds_cnn/ds_cnn3.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    interpreter.allocate_tensors()

    start = time.time()
    audio, sr = librosa.load('/Users/quangbd/Documents/data/kws-data/speech_commands_v0.02/go/0b7ee1a0_nohash_0.wav',
                             sr=16000, duration=1)
    audio = np.reshape(audio, [16000, 1])
    input_data = np.array(audio, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    print('Time: ', (time.time() - start) * 1000)
    re_index = np.argmax(output_data)

    args = prepare_config()
    labels = prepare_words_list(args.wanted_words.split(','))
    print('Result: {} {}'.format(labels[re_index], output_data[re_index]))


if __name__ == '__main__':
    # test_batch()
    # test_checkpoint()
    # test_pb()
    test_tflite()