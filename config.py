RANDOM_SEED = 150595

CHANNELS = 1
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
TIME_SHIFT_MS = 0

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
VOCAL_WORD_LABEL = '_vocal_'
NEGATIVE_WORD_LABEL = '_negative_'
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
REAL_NEGATIVE_LABEL = '_real_negative_'
AUGMENT_POSITIVE_LABEL = '_augment_positive_'

NEGATIVE_WORD_INDEX = 0
POSITIVE_WORD_INDEX = 1

SILENCE_PERCENTAGE = 500
VOCAL_PERCENTAGE = -1
NEGATIVE_PERCENTAGE = -1
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

BACKGROUND_FREQUENCY = 0.7
BACKGROUND_VOLUME = 0.1
BACKGROUND_SILENCE_FREQUENCY = 0.95
BACKGROUND_SILENCE_VOLUME = 0.1
DOWN_VOLUME_FREQUENCY = 0
DOWN_VOLUME_RANGE = 0
