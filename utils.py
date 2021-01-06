import argparse


def prepare_config():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws/train/',
                        help='Where to download the speech training data to.')
    parser.add_argument('--augment_dir',
                        type=str,
                        default='/Users/quangbd/Documents/data/kws/tmp/',
                        help='Where to download the speech training data to.')
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

    # For div train / test
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
    parser.add_argument('--testing_percentage',
                        type=int,
                        default=10,
                        help='What percentage of wavs to use as a testing set.')

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
                        default='/Users/quangbd/Documents/data/model/kws/tmp/ds_cnn/ds_cnn1/'
                                'training/best/ds_cnn_9994.ckpt-56000',
                        help='Checkpoint to load the weights from.')
    parser.add_argument('--pb',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/tmp/ds_cnn/ds_cnn1.pb',
                        help='Where to save the frozen graph.')
    parser.add_argument('--tflite',
                        type=str,
                        default='/Users/quangbd/Documents/data/model/kws/tmp/ds_cnn/ds_cnn1.tflite',
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
    return parser.parse_args()

