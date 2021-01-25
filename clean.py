import os
import sox
import time
import shutil
import random
import librosa
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from constant import *
from google.cloud import storage
from scipy.io.wavfile import write

random.seed(RANDOM_SEED)


def convert_librosa(input_file, output_file):
    y, sr = librosa.load(input_file, sr=None)
    if len(y) > DESIRED_SAMPLE:
        y = y[:DESIRED_SAMPLE]
    else:
        y = np.pad(y, (0, DESIRED_SAMPLE - len(y)))
    scaled = np.int16(y * 32767)
    write(output_file, sr, scaled)


def convert_standard_wav(wav_folder_path, output_path, seconds=None):
    """
    Convert raw wav to 16kHz and 1 channel wav file
    :param seconds: Desired sample
    :param wav_folder_path: Folder contain wav files
    :param output_path: Output folder path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if seconds:
        for file in glob(os.path.join(wav_folder_path, '*.wav')):
            transformer = sox.Transformer()
            transformer.trim(0, 1)
            output_filename = os.path.join(output_path, file.split('/')[-1])
            transformer.build_file(file, output_filename)
            filesize = os.path.getsize(output_filename)
            if filesize != FILE_SIZE:
                print(output_filename)
                os.remove(output_filename)
                convert_librosa(file, output_filename)
    else:
        for file in glob(os.path.join(wav_folder_path, '*.wav')):
            subprocess.call(
                'ffmpeg -i {} -ar 16000 -ac 1 {}'.format(file.replace(' ', '\ '),
                                                         os.path.join(output_path, '{}.wav'.format(time.time()))),
                shell=True)


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


def download_firebase(days, output_path):
    """
    Download wav files, write spk2utt - utt2spk - wav.scp for VAD
    :param days: [2021-01-01, 2021-01-02]
    :param output_path: Output folder path
    :return: The folder contains data
    """

    storage_client = storage.Client.from_service_account_json(KEY_PATH)
    bucket = storage_client.bucket(BUCKET_NAME)
    wavs_path = os.path.join(output_path, 'wavs')
    tmp_file = 'tmp.wav'
    all_files = []
    all_names = {}
    if not os.path.exists(wavs_path):
        os.makedirs(wavs_path)
    for day in days:
        print('Download day:', day)
        blobs = bucket.list_blobs(prefix='kws/{}'.format(day))
        for blob in blobs:
            # blob.download_to_filename(tmp_file)
            username = blob.name.split('/')[-1].split('@')[0].lower().replace('v.', '')
            email = '{}@{}'.format(blob.name.split('/')[-1].split('@')[0].lower().split('_')[0],
                                   blob.name.split('/')[-1].split('@')[1].split('||')[0].lower())
            if email in all_names:
                all_names[email] += 1
            else:
                all_names[email] = 1
            filename = '{}_{}.wav'.format(username, time.time())
            all_files.append(filename)
            # subprocess.call('ffmpeg -i {} -ar 16000 -ac 1 {}'
            #                 .format(tmp_file, os.path.join(wavs_path, filename)), shell=True)
            # os.remove(tmp_file)

    # write spk2utt
    spk2utt_file = os.path.join(output_path, 'spk2utt')
    with open(spk2utt_file, 'w') as output_file:
        for filename in all_files:
            output_file.write('{} {}\n'.format(filename, filename))
            # convert

    # write utt2spk
    shutil.copy(spk2utt_file, os.path.join(output_path, 'utt2spk'))

    # write wav.scp
    with open(os.path.join(output_path, 'wav.scp'), 'w') as output_file:
        for filename in all_files:
            output_file.write('{} {}\n'.format(filename, os.path.join(VAD_WAV_PATH,
                                                                      output_path.split('/')[-1], 'wavs',
                                                                      filename)))

    # write static file
    all_names = [(username, all_names[username]) for username in all_names]
    all_names = sorted(all_names, key=lambda user: user[0])
    with open(os.path.join(output_path, 'statistics.csv'), 'w') as output_file:
        output_file.write('username\tnumber of files\n')
        for u in all_names:
            output_file.write('{}\t{}\n'.format(u[0], u[1]))


def segment_to_wavs(wav_dir, output_dir, segment_file):
    """
    Read segment file then split wav
    :param wav_dir: The dir contains wav
    :param output_dir: Output directory
    :param segment_file: Path to segment file
    :return: Write to file
    """
    data_dict = {}
    with open(segment_file) as input_file:
        for line in input_file:
            components = line.split(' ')
            filename = components[1]
            duration_time = float(components[2]), float(components[3])
            if filename in data_dict:
                data_dict[filename].append(duration_time)
            else:
                data_dict[filename] = [duration_time]

    # statistic
    bad_files = []
    good_files = []
    for filename in data_dict:
        if len(data_dict[filename]) != 5:
            bad_files.append(filename)
        else:
            good_files.append(filename)

    # split file
    ex_bad_count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in good_files:
        for index, duration_time in enumerate(data_dict[filename]):
            start = duration_time[0]
            end = duration_time[1]
            if end - start >= 1:
                start += (end - start - 1) / 2
            else:
                start -= (1 - end - start) / 2
            start = round(start, 3)
            transformer = sox.Transformer()
            transformer.trim(start, start + 1)
            output_file_path = os.path.join(output_dir, '{}_{}.wav'.format(filename.replace('.wav', ''), index))
            transformer.build_file(os.path.join(wav_dir, filename), output_file_path)
            if os.path.getsize(output_file_path) != FILE_SIZE:
                ex_bad_count += 1
                bad_files.append(filename)
                for rm_filename in glob(os.path.join(output_dir, '{}_*.wav'.format(filename.replace('.wav', '')))):
                    os.remove(rm_filename)
                break

    print('Bad files {} - good files {}'.format(len(bad_files), len(good_files) - ex_bad_count))
    bad_dir = os.path.join('/'.join(output_dir.split('/')[:-1]), 'bad_files')
    if not os.path.exists(bad_dir):
        os.makedirs(bad_dir)
    for filename in bad_files:
        print(filename)
        shutil.copy(os.path.join(wav_dir, filename), os.path.join(bad_dir, filename))


if __name__ == '__main__':
    # split_folder('/home/ubuntu/new_kws/kws_data/raw/vocal_vn',
    #              '/home/ubuntu/new_kws/kws_data/train/vocal_vn',
    #              name='vocal_vn')
    convert_standard_wav('/Users/quangbd/Desktop/tmp3',
                         '/Users/quangbd/Desktop/tmp3_split', seconds=1)
    # download_firebase(['2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10'], '/Users/quangbd/Desktop/heyvf_20210110')
    # segment_to_wavs('/Users/quangbd/Documents/data/kws/vin_collect/heyvf_20210110/wavs',
    #                 '/Users/quangbd/Documents/data/kws/vin_collect/heyvf_20210110/splits',
    #                 '/Users/quangbd/Documents/data/kws/vin_collect/heyvf_20210110/segments20210110')
    # convert_standard_wav('/Users/quangbd/Documents/data/kws/vin_collect/heyvf_20210110/tmp',
    #                      '/Users/quangbd/Documents/data/kws/vin_collect/heyvf_20210110/tmp_split',
    #                      seconds=1)
