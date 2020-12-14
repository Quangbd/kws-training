import os
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
import pvporcupine
from scipy.io import wavfile
import soundfile


if __name__ == '__main__':
    # background_noise_files = glob('/Users/quangbd/Documents/data/kws-data/viet_nam_20201113/_background_noise_/*.wav')
    # ran = np.random.randint(0, len(background_noise_files))
    # background_noise, sr = librosa.load(background_noise_files[ran], sr=None)
    # ran = np.random.randint(0, len(background_noise) - sr)
    # background_noise = background_noise[ran:ran + sr]
    #
    # test_files = glob('/Users/quangbd/Documents/data/kws-data/viet_nam_20201113/viet_nam/*.wav')
    # ran = np.random.randint(0, len(test_files))
    # y, _ = librosa.load(test_files[ran], sr=None)
    # y += background_noise * np.random.uniform(0, 0.3)
    # write('/Users/quangbd/Desktop/tmp.wav', sr, np.int16(y * 32767))
    handle = pvporcupine.create(keyword_paths=['/Users/quangbd/Documents/data/model/kws/hey_vinfast/hey_vin_fast_mac_1_6_2021_v1.9.0.ppn'], sensitivities=[1])

    # print(handle.frame_length)
    # print(handle.sample_rate)
    audio, sample_rate = soundfile.read('/Users/quangbd/Desktop/untitledfolder/manhnt2-20201208.wav', dtype='int16')
    num_frames = len(audio)
    for i in range(num_frames):
        frame = audio[i * handle.frame_length:(i + 1) * handle.frame_length]
        result = handle.process(frame)
        if result >= 0:
            print(
                "Detected '%s' at %.2f sec" %
                ('abc', float(i * handle.frame_length) / float(handle.sample_rate)))

    handle.delete()


