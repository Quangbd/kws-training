import re
import os
import hashlib
import argparse
from config import *
from tensorflow.python.util import compat


def prepare_words_list(wanted_words):
    return [NEGATIVE_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)

    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
                       * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def prepare_normal_config():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws-data/viet_nam_20201103/',
                        help='Where to download the speech training data to.')
    parser.add_argument('--window_size_ms',
                        type=float,
                        default=40.0,
                        help='How long each spectrogram timeslice is.')
    parser.add_argument('--window_stride_ms',
                        type=float,
                        default=20.0,
                        help='How long each spectrogram timeslice is.')
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
                        default='/Users/quangbd/Documents/data/model/kws/tmp_speech_commands_v0.02/train_logs',
                        help='Where to save summary logs for TensorBoard.')
    parser.add_argument('--train_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/tmp_speech_commands_v0.02/checkpoints',
                        help='Directory to write event logs and checkpoint.')
    parser.add_argument('--wanted_words',
                        type=str,
                        default='heyvf',
                        help='Words to use (others will be added to an unknown label).', )
    # parser.add_argument('--model_architecture',
    #                     type=str,
    #                     default='ds_cnn',
    #                     help='What model architecture to use')
    # parser.add_argument('--model_size_info',
    #                     type=int,
    #                     nargs="+",
    #                     default=[6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1,
    #                              276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1],
    #                     help='Model dimensions - different for various models.')
    # parser.add_argument('--checkpoint',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/ds_cnn/ds_cnn3/'
    #                             'training/best/ds_cnn_9974.ckpt-52000',
    #                     help='Checkpoint to load the weights from.')
    # parser.add_argument('--pb',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/ds_cnn/ds_cnn3.pb',
    #                     help='Where to save the frozen graph.')
    # parser.add_argument('--tflite',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/ds_cnn/ds_cnn3.tflite',
    #                     help='Where to save the frozen graph.')
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
                                'training/best/ds_cnn_9985.ckpt-46000',
                        help='Checkpoint to load the weights from.')
    parser.add_argument('--pb',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1.pb',
                        help='Where to save the frozen graph.')
    parser.add_argument('--tflite',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1.tflite',
                        help='Where to save the frozen graph.')
    # parser.add_argument('--model_architecture',
    #                     type=str,
    #                     default='crnn',
    #                     help='What model architecture to use')
    # parser.add_argument('--model_size_info',
    #                     type=int,
    #                     nargs="+",
    #                     default=[48, 10, 4, 2, 2, 2, 60, 84],
    #                     help='Model dimensions - different for various models.')
    # parser.add_argument('--checkpoint',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/crnn/crnn1/'
    #                             'training/best/crnn_9860.ckpt-3000',
    #                     help='Checkpoint to load the weights from.')
    # parser.add_argument('--pb',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/crnn/crnn1.pb',
    #                     help='Where to save the frozen graph.')
    # parser.add_argument('--tflite',
    #                     type=str,
    #                     default='/Users/quangbd/Documents/data/model/kws/viet_nam/crnn/crnn1.tflite',
    #                     help='Where to save the frozen graph.')
    return parser.parse_args()


def prepare_record_config():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--model_path',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/heyvf/ds_cnn/ds_cnn1.tflite',
                        help='Tflite model path')
    parser.add_argument('--wanted_words',
                        type=str,
                        default='heyvf',
                        help='Words to use (others will be added to an unknown label).')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=1024,
                        help='Chunk size')
    parser.add_argument('--record_time',
                        type=int,
                        default=600,
                        help='Record time in seconds')
    return parser.parse_args()
