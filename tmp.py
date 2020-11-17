import os
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.io.wavfile as wav
from scipy.io.wavfile import write

if __name__ == '__main__':
    background_noise_files = glob('/Users/quangbd/Documents/data/kws-data/viet_nam_20201113/_background_noise_/*.wav')
    ran = np.random.randint(0, len(background_noise_files))
    background_noise, sr = librosa.load(background_noise_files[ran], sr=None)
    ran = np.random.randint(0, len(background_noise) - sr)
    background_noise = background_noise[ran:ran + sr]

    test_files = glob('/Users/quangbd/Documents/data/kws-data/viet_nam_20201113/viet_nam/*.wav')
    ran = np.random.randint(0, len(test_files))
    y, _ = librosa.load(test_files[ran], sr=None)
    y += background_noise * np.random.uniform(0, 0.3)
    write('/Users/quangbd/Desktop/tmp.wav', sr, np.int16(y * 32767))
