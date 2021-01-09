import random
import argparse
from constant import *
import tensorflow as tf
from models import select_model


def init_session():
    random.seed(RANDOM_SEED)
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.InteractiveSession()
    return sess


def init_model(args):
    model = select_model(window_size_ms=args.window_size_ms, window_stride_ms=args.window_stride_ms,
                         dct_coefficient_count=args.dct_coefficient_count,
                         name=args.model_architecture)
    model_settings = model.prepare_model_settings()
    return model, model_settings


def read_file(wav_path, session=None):
    """
    Read audio to chunks per second
    :param session: TF session
    :param wav_path: File path
    :return: numpy array
    """
    if session:
        wav_filename_placeholder_ = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = tf.io.read_file(wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
        return session.run(wav_decoder, feed_dict={wav_filename_placeholder_: wav_path}).audio
    else:
        wav_loader = tf.io.read_file(wav_path)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1).audio
        return wav_decoder.numpy()


def load_model(args, model_type, session=None):
    """
    Load model
    :param args: Args
    :param model_type: checkpoint, pb, tflite
    :param session: tf session
    :return: model
    """
    if model_type == ModelType.CHECKPOINT:
        model, model_settings = init_model(args)
        wav_data = tf.compat.v1.placeholder(tf.float32, [model_settings['desired_samples'], 1],
                                            name='decoded_sample_data')
        spectrogram = tf.raw_ops.AudioSpectrogram(input=wav_data,
                                                  window_size=model_settings['window_size_samples'],
                                                  stride=model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
        mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=model_settings['sample_rate'],
                                dct_coefficient_count=model_settings['dct_coefficient_count'])
        fingerprint_frequency_size = model_settings['dct_coefficient_count']
        fingerprint_time_size = model_settings['spectrogram_length']
        fingerprint_input = tf.reshape(mfcc_, [-1, fingerprint_time_size * fingerprint_frequency_size])
        logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info, is_training=False)
        result = tf.sigmoid(logits, name='labels_sigmoid')
        model.load_variables_from_checkpoint(session, args.checkpoint)
        return wav_data, result, model_settings['desired_samples']
    elif model_type == ModelType.GRAPH:
        with tf.compat.v1.gfile.FastGFile(args.pb, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        softmax_tensor = session.graph.get_tensor_by_name('labels_sigmoid:0')
        wav_data = 'decoded_sample_data:0'
        return wav_data, softmax_tensor, DESIRED_SAMPLE
    elif model_type == ModelType.TFLITE:
        interpreter = tf.lite.Interpreter(model_path=args.tflite)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        return input_details[0]['index'], output_details[0]['index'], interpreter, DESIRED_SAMPLE


def prepare_config():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws/train/',
                        help='Where to download the speech training data to.')
    parser.add_argument('--augment_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws/tmp/',
                        help='Where to save add noise data.')
    parser.add_argument('--window_size_ms',
                        type=float,
                        default=40.0,
                        help='How long each spectrogram time slice is.')
    parser.add_argument('--window_stride_ms',
                        type=float,
                        default=20.0,
                        help='How long each spectrogram time slice is.')
    parser.add_argument('--dct_coefficient_count',
                        type=int,
                        default=10,
                        help='How many bins to use for the MFCC fingerprint.')
    parser.add_argument('--training_steps',
                        type=str,
                        default='15000,25000,20000',
                        help='How many training loops to run.')
    parser.add_argument('--eval_step_interval',
                        type=int,
                        default=400,
                        help='How often to evaluate the training results.')
    parser.add_argument('--learning_rate',
                        type=str,
                        default='0.0001,0.00005,0.00001',
                        help='How large a learning rate to use when training.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='How many items to train with at once.')
    parser.add_argument('--summaries_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/tmp/train_logs',
                        help='Where to save summary logs for TensorBoard.')
    parser.add_argument('--train_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/tmp/checkpoints',
                        help='Directory to write event logs and checkpoint.')

    # For div train / val
    parser.add_argument('--silence_percentage',
                        type=int,
                        default=500,
                        help='How much of the training data should be silence.')
    parser.add_argument('--negative_percentage',
                        type=int,
                        default=-1,
                        help='How much of the training data should be negative, -1 for all.')
    parser.add_argument('--validation_percentage',
                        type=int,
                        default=10,
                        help='What percentage of wavs to use as a validation set.')

    # For volume
    parser.add_argument('--background_frequency',
                        type=int,
                        default=0.7,
                        help='How many of the training samples have background noise mixed in.')
    parser.add_argument('--background_silence_frequency',
                        type=int,
                        default=0.95,
                        help='How many of the silence samples.')
    parser.add_argument('--background_silence_volume',
                        type=int,
                        default=1,
                        help='How loud of the silence samples.')

    # For model
    parser.add_argument('--model_architecture',
                        type=str,
                        default='ds_cnn',
                        help='What model architecture to use')
    parser.add_argument('--model_size_info',
                        type=int,
                        nargs="+",
                        default=[5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1],
                        help='Model dimensions - different for various models.')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1/'
                                'training/best/ds_cnn_9964.ckpt-43000',
                        help='Checkpoint to load the weights from.')
    parser.add_argument('--pb',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1.pb',
                        help='Where to save the frozen graph.')
    parser.add_argument('--tflite',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1.tflite',
                        help='Where to save the frozen graph.')
    # For recorder
    parser.add_argument('--chunk_size',
                        type=int,
                        default=1024,
                        help='Chunk size')
    parser.add_argument('--record_time',
                        type=int,
                        default=600,
                        help='Record time in seconds')

    # For test
    parser.add_argument('--test_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws/test/',
                        help='Where to save test data.')
    parser.add_argument('--test_model_type',
                        type=str,
                        default='pb',
                        help='Model type for test.')

    return parser.parse_args()
