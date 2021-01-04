import os
import sox
import time
import random
import numpy as np
from constant import *
from tqdm import tqdm
from glob import glob

random.seed(RANDOM_SEED)


def get_all_wav_file(wav_folder_path, shuffle=True):
    """
    Get all wav file paths
    :param shuffle: Shuffle in list
    :param wav_folder_path: Folder path input
    :return: List
    """
    print('Get all files')
    all_paths = []
    for file in tqdm(glob(os.path.join(wav_folder_path, '*.wav'))):
        all_paths.append(file)
    if shuffle:
        random.shuffle(all_paths)
    print('Count files: {}'.format(len(all_paths)))
    return all_paths


def trim_to_second(wav_file, output_file, min_duration=1, max_duration=5):
    """
    Trim a wav file in 1 second.
    :param output_file: Output path
    :param wav_file: Wav file path
    :param min_duration: For skip
    :param max_duration: For skip
    :return: Done
    """
    transformer = sox.Transformer()
    stat = transformer.stat(wav_file)
    duration = float(stat['Length (seconds)'])
    if duration < min_duration or duration > max_duration:
        return 0
    if float(stat['RMS amplitude']) == 0:
        return 0
    start = duration / 2 - 0.5
    transformer.trim(start, start + 1)
    transformer.build_file(wav_file, output_file)

    # check output
    transformer = sox.Transformer()
    stat = transformer.stat(output_file)
    if float(stat['RMS amplitude']) == 0:
        # remove file
        os.remove(output_file)
        return 0
    return 1


def trim_folder(wav_folder_path, wav_output_path, name, count=100_000):
    """
    Trim all files in folder.
    :param name: The name for file
    :param wav_output_path: Output path
    :param count: Number of file for process
    :param wav_folder_path: Path to all wav files
    """
    all_paths = get_all_wav_file(wav_folder_path)
    count_file = 0
    for file in tqdm(all_paths):
        count_file += trim_to_second(file, os.path.join(wav_output_path, '{}_{}.wav'.format(name, time.time())))
        if count_file > count:
            break


def split_to_second(wav_file, output_folder_path, window=0.5, max_count_in_file=10):
    """
    Split a wav file to many small files of 1 second in length.
    :param max_count_in_file: Max small file
    :param window: Duration between 2 samples
    :param wav_file: Input path
    :param output_folder_path: Output path without name
    :return Number of files after split
    """
    transformer = sox.Transformer()
    stat = transformer.stat(wav_file)
    duration = float(stat['Length (seconds)'])
    count = 0
    for index, start in enumerate(np.arange(0, duration, window)):
        transformer = sox.Transformer()
        if start + 1 > duration:
            break
        transformer.trim(start, start + 1)
        output_file_path = '{}_{}_{}.wav'.format(output_folder_path, time.time(), index)
        transformer.build_file(wav_file, output_file_path)
        transformer = sox.Transformer()
        stat = transformer.stat(output_file_path)
        if float(stat['RMS amplitude']) == 0:
            # remove file
            os.remove(output_file_path)
            continue
        count += 1
        if count >= max_count_in_file:
            break
    return count


def split_folder(wav_folder_path, wav_output_path, name, count=100_000):
    """
    Split all files in folder.
    :param name: The name for file
    :param wav_output_path: Output path
    :param count: Number of file for process
    :param wav_folder_path: Path to all wav files
    """
    all_paths = get_all_wav_file(wav_folder_path)
    count_file = 0
    for file in tqdm(all_paths):
        count_file += split_to_second(file, os.path.join(wav_output_path, name))
        if count_file > count:
            break


if __name__ == '__main__':
    split_folder('/home/ubuntu/new_kws/kws_data/raw/vocal_vn',
                 '/home/ubuntu/new_kws/kws_data/train/vocal_vn',
                 name='vocal_vn')
