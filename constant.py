from enum import Enum

RANDOM_SEED = 150595

CHANNELS = 1
SAMPLE_RATE = 16_000
DESIRED_SAMPLE = 16_000
CLIP_DURATION_MS = 1000
TIME_SHIFT_MS = 0
FILE_SIZE = 32044

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
BACKGROUND_NOISE_DIR_NAME = 'background_noise'
SILENCE_LABEL = 'silence'
VOCAL_WORD_LABEL = 'vocal'
POSITIVE_LABEL = 'positive'
NEGATIVE_LABEL = 'negative'
AUGMENT_POSITIVE_LABEL = 'augment_positive'
REAL_NEGATIVE_LABEL = 'real_negative'

NEGATIVE_WORD_INDEX = 0
POSITIVE_WORD_INDEX = 1

# For firebase
BUCKET_NAME = 'voice-kws.appspot.com'
KEY_PATH = 'firebase.json'
VAD_WAV_PATH = '/home/ubuntu/projects/VAD/data'

MIN_TEST_THRESHOLD = 0
MAX_TEST_THRESHOLD = 1.02
RANGE_TEST_THRESHOLD = 0.02


class ModelType(Enum):
    CHECKPOINT = 1
    GRAPH = 2
    TFLITE = 3
