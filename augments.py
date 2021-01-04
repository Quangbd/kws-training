"""
Install WavAugment
git clone git@github.com:facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop
python -m pytest -v --doctest-modules
"""

import os
import time
import torch
import random
import librosa
import augment
import torchaudio
import numpy as np
from glob import glob
from dataclasses import dataclass


class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)


class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max

    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)


@dataclass
class RandomReverb:
    reverb_min: int = 50
    reverb_max: int = 50
    damping_min: int = 50
    damping_max: int = 50
    room_scale_min: int = 0
    room_scale_max: int = 100

    def __call__(self):
        reverb = np.random.randint(self.reverb_min, self.reverb_max + 1)
        damping = np.random.randint(self.damping_min, self.damping_max + 1)
        room_scale = np.random.randint(self.room_scale_min, self.room_scale_min + 1)
        return [reverb, damping, room_scale]


class AddNoise:
    def __init__(self, background_noise, total_sample_in_file=16_000, min_snr=3, max_snr=15):
        self.noise_data = background_noise
        self.total_sample_in_file = total_sample_in_file
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self):
        return torch.from_numpy(self.noise_data)

    def get_snr(self):
        return random.randint(self.min_snr, self.max_snr)


def augmentation_factory(methods, background_noise, sampling_rate=16_000, total_sample=16_000):
    """
    Select chain for process
    :param background_noise: Background data
    :param methods: All random methods
    :param sampling_rate: Sample rate
    :param total_sample: Total sample of a wav file
    :return: Chain object
    """
    chain = augment.EffectChain()
    for method in methods:
        if method == 'pitch':
            pitch_randomizer = RandomPitchShift()
            chain = chain.pitch(pitch_randomizer).rate(sampling_rate)
        elif method == 'clip':
            chain = chain.clip(RandomClipFactor())
        elif method == 'reverb':
            randomized_params = RandomReverb()
            chain = chain.reverb(randomized_params).channels()
        if method == 'noise':
            add_noise = AddNoise(background_noise, total_sample_in_file=total_sample)
            chain.additive_noise(add_noise, add_noise.get_snr())
    return chain


def process_file(file_path, output_dir, background_noise=None, file_name=None):
    """
    Process wav file by chains
    :param file_name: Optional for file name
    :param file_path: Input path
    :param output_dir: Output directory path
    :param background_noise: Background data
    :return: Save to a wav file, output path
    """
    chains = ['pitch', 'reverb', 'clip']
    number_method_random = random.randint(0, len(chains))
    methods = []
    if background_noise is not None:
        methods.append('noise')
    for i in range(number_method_random):
        random_index = random.randint(0, len(chains) - 1)
        methods.append(chains[random_index])
        chains.pop(random_index)
    x, sampling_rate = torchaudio.load(file_path)
    augmentation_chain = augmentation_factory(methods, background_noise, total_sample=x.shape[1])
    y = augmentation_chain.apply(x, src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
                                 target_info=dict(rate=sampling_rate, length=0))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if file_name:
        output_path = os.path.join(output_dir, '{}_{}_{}.wav'.format(file_name, '_'.join(methods), time.time()))
    else:
        output_path = os.path.join(output_dir, '{}_{}.wav'.format('_'.join(methods), time.time()))
    if y.numel() > 16000:
        y = y[:, :16000]
        print(output_path)
    torchaudio.save(output_path, y, sampling_rate)
    return output_path


def augment_positive(total_sample=16_000, limit=5000):
    # get background noise
    background_all_data = []
    for background_noise_file in glob('/Users/quangbd/Documents/data/kws/train/background_noise/*.wav'):
        background_all_data.append(librosa.load(background_noise_file, sr=None)[0])

    count = 0
    while True:
        for file in glob('/Users/quangbd/Documents/data/kws/train/keyword/*.wav'):
            background_index = random.randint(0, len(background_all_data) - 1)
            background_data = background_all_data[background_index]
            random_index = random.randint(0, len(background_data) - total_sample - 1)
            noise = background_data[random_index:random_index + total_sample]
            process_file(file, '/Users/quangbd/Documents/data/kws/train/augment_positive', noise,
                         file.split('/')[-1].split('_')[0])
            count += 1
        if count > limit:
            break


if __name__ == '__main__':
    # back, _ = librosa.load('/Users/quangbd/Desktop/custom1_pi_noise.wav', sr=None)
    # back = back[:16000]
    # process_file('/Users/quangbd/Desktop/0_1607514663.416621.wav',
    #              '/Users/quangbd/Desktop/tmp.wav', back)
    augment_positive()
