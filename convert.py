from utils import *
import tensorflow as tf
from models import select_model
from tensorflow.python.framework import graph_util


def checkpoint2pb():
    def _process(_):
        # random.seed(RANDOM_SEED)
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

        # wav_data_placeholder = tf.compat.v1.placeholder(tf.string, [], name='wav_data')
        # decoded_sample_data = tf.audio.decode_wav(wav_data_placeholder,
        #                                           desired_channels=1,
        #                                           desired_samples=model_settings['desired_samples'],
        #                                           name='decoded_sample_data')
        # spectrogram = tf.raw_ops.AudioSpectrogram(input=decoded_sample_data.audio,
        #                                           window_size=model_settings['window_size_samples'],
        #                                           stride=model_settings['window_stride_samples'],
        #                                           magnitude_squared=True)
        # mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
        #                         sample_rate=decoded_sample_data.sample_rate,
        #                         dct_coefficient_count=model_settings['dct_coefficient_count'])
        decoded_sample_data = tf.compat.v1.placeholder(tf.float32, [model_settings['sample_rate'], 1],
                                                       name='decoded_sample_data')
        spectrogram = tf.raw_ops.AudioSpectrogram(input=decoded_sample_data,
                                                  window_size=model_settings['window_size_samples'],
                                                  stride=model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
        mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=model_settings['sample_rate'],
                                dct_coefficient_count=model_settings['dct_coefficient_count'])

        fingerprint_frequency_size = model_settings['dct_coefficient_count']
        fingerprint_time_size = model_settings['spectrogram_length']
        reshaped_input = tf.reshape(mfcc_, [-1, fingerprint_time_size * fingerprint_frequency_size])
        logits, dropout_prob = model.forward(reshaped_input, args.model_size_info, is_training=False)

        # Create an output to use for inference.
        tf.nn.softmax(logits, name='labels_softmax')

        model.load_variables_from_checkpoint(sess, args.checkpoint)
        nodes = [op.name for op in tf.compat.v1.get_default_graph().get_operations()]
        print(nodes)

        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['labels_softmax'])
        tf.compat.v1.train.write_graph(frozen_graph_def,
                                       os.path.dirname(args.pb),
                                       os.path.basename(args.pb),
                                       as_text=False)
        tf.compat.v1.logging.info('Saved frozen graph to %s', args.pb)

        sess.close()

    args = prepare_normal_config()
    tf.compat.v1.app.run(main=_process)


def pb2tflite():
    args = prepare_normal_config()
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        args.pb,
        input_arrays=['decoded_sample_data'],
        output_arrays=['labels_softmax'])
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(args.tflite, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # checkpoint2pb()
    pb2tflite()
