import time
import librosa
import numpy as np
from utils import *
from glob import glob
import tensorflow as tf
from scipy.io.wavfile import write
from train import init_session, init_model, init_data, init_placeholder, init_graph


def test_batch():
    def _process(_):
        sess = init_session()
        model, model_settings = init_model()
        audio_loader = init_data(model_settings)

        time_shift_samples, _, _, fingerprint_input, ground_truth_input = init_placeholder(model_settings,
                                                                                           is_train=False)
        evaluation_step, _, _, _, confusion_matrix, dropout_prob = \
            init_graph(model, model_settings, fingerprint_input, ground_truth_input)

        # test set
        test_size = audio_loader.size('testing')
        tf.compat.v1.logging.info('set_size=%d', test_size)
        total_accuracy = 0
        total_conf_matrix = None
        for i in range(0, test_size, args.batch_size):
            test_fingerprints, test_ground_truth, _, _ = audio_loader \
                .load_batch(sess, args.batch_size, offset=i, background_frequency=0, time_shift=0, mode='testing')
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
        sess = init_session()
        model, model_settings = init_model()

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
            feed_dict={wav_filename_placeholder_: 'tmp.wav'})

        print('Result: {} {}'.format(predicted_indices[0], predicted_results[0][predicted_indices[0]]))
        sess.close()

    args = prepare_config()
    tf.compat.v1.app.run(main=_process)


def test_pb():
    audio, sr = librosa.load('tmp.wav',
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
        print('Result: {} {}'.format(re_index, predictions[re_index]))


def test_tflite():
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    interpreter.allocate_tensors()

    for file in glob('/Users/quangbd/Downloads/soundcore/*.wav'):
        print(file)
        tmp_audio = np.random.uniform(low=0.0001, high=0.0005, size=(16000,))
        audio, sr = librosa.load(file, sr=16000, duration=1)
        if len(audio) < 16000:
            tmp_audio[:len(audio)] = audio
        else:
            tmp_audio[:] = audio[:16000]
        audio = tmp_audio
        audio = np.reshape(audio, [16000, 1])
        input_data = np.array(audio, dtype=np.float32)
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        print(time.time() - start)
        print(output_data)
        write('/Users/quangbd/Desktop/tmp/{}.wav'.format(time.time()), 16000, audio)

        # re_index = np.argmax(output_data)
        # args = prepare_normal_config()
        # labels = prepare_words_list(args.wanted_words.split(','))
        # score = output_data[re_index]
        # label = labels[re_index]
        # print(label)
        # if label == 'viet_nam' and score < 0.6:
        # print(file)
        # print('Result: {} {}'.format(label, score))
        # print('-------')


def test_tflite_large_file():
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    interpreter.allocate_tensors()

    for file in glob('/Users/quangbd/Desktop/*.wav'):
        print(file)
        audio, sr = librosa.load(file, sr=16000)
        for i in range(0, len(audio), 512):
            input_buffer = audio[i:i + 16000]
            if len(input_buffer) == 16000:
                input_buffer = np.reshape(input_buffer, [16000, 1])
                input_data = np.array(input_buffer, dtype=np.float32)
                start = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                if output_data[1] > output_data[0]:
                    print(time.time() - start)
                    print(output_data)
                    write('/Users/quangbd/Desktop/tmp/{}_{}_{}.wav'
                          .format(time.time(), int(output_data[1] * 100), i / 16000), 16000, input_buffer)


if __name__ == '__main__':
    # test_batch()
    # test_checkpoint()
    # test_pb()
    args = prepare_config()
    test_tflite_large_file()
