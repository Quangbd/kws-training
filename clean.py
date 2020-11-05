import os
import re
import sox
import json
import shutil
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool as Pool


def clean_vinai_data_split_word():
    def word2list(sentence, script):
        words = []
        index = 0
        for start_time, duration, phone in sentence:
            start_time = float(start_time)
            duration = float(duration)
            if phone[-1] == 'B':
                words.append([start_time, duration, script[index]])
            elif phone[-1] == 'E':
                words[index][1] = round(start_time - words[index][0] + duration, 3)
                index += 1
        return words

    def sentences2file(directory, sentence_list):
        for file_index, sentence_info in tqdm(sentence_list):
            if file_index < 100_000:
                file_index = str(100_000 + file_index)[1:]
            else:
                file_index = str(1000_000 + file_index)[1:]
            check_word = {}
            for word_info in sentence_info:
                if word_info[2] in check_word:
                    check_word[word_info[2]] += 1
                else:
                    check_word[word_info[2]] = 0
                output_wav = '{}_{}.wav'.format(file_index, check_word[word_info[2]])
                input_wav = '{}.wav'.format(file_index)
                output_dir = os.path.join(directory, 'clean_word', word_info[2])
                os.makedirs(output_dir, exist_ok=True)
                duration = word_info[1]
                if duration > 1:
                    duration = 1
                tfm = sox.Transformer()
                tfm.trim(word_info[0], word_info[0] + duration)
                # tfm.tempo(0.8)
                pad = (1 - duration) / 2
                tfm.pad(pad, pad)
                tfm.build_file(os.path.join(directory, 'raw', input_wav), os.path.join(output_dir, output_wav))

    directory_ = '/home/ubuntu/viet_nam'
    script_map = {}
    dictionary = {}
    with open(os.path.join(directory_, 'script.txt')) as input_file:
        for line in input_file:
            line = line.lower().strip()
            components = re.split('\\s+', line)
            script_map[int(components[0])] = components[1:]
            for word in components[1:]:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    count_word = 0
    for word in dictionary:
        count_word += dictionary[word]
    print('count word:', count_word)

    word_list = []
    with open(os.path.join(directory_, 'alignment.txt.ctm')) as input_file:
        for line in input_file:
            components = re.split('\\s+', line.strip())
            word_list.append((int(components[0].split('_')[0]), components[2:]))

    sentence_dict = defaultdict(list)
    for key, value in word_list:
        sentence_dict[key].append(value)

    sentence_list_ = [(key, word2list(sentence_, script_map[key])) for key, sentence_ in sentence_dict.items()]

    # Trim to a new wav file
    # core = 8
    # batch_size = len(sentence_list_) // core
    # freeze_support()
    # pool = Pool(core)
    # for i in range(0, len(sentence_list_), batch_size):
    #     pool.apply_async(sentences2file, (directory_, sentence_list_[i: i + batch_size]))
    # pool.close()
    # pool.join()
    sentences2file(directory_, sentence_list_)


def clean_vinai_data_split_time():
    # Read align file----------
    def read_align_file(root_dir):
        file_start_duration_phone_map = {}
        with open(os.path.join(root_dir, 'alignment.txt.ctm')) as input_file:
            for line in input_file:
                components = line.strip().split()
                file_index = int(components[0].split('_')[0])
                start_time = round(float(components[2]), 2)
                duration = round(float(components[3]), 2)
                phoneme = components[4]
                if file_index in file_start_duration_phone_map:
                    file_start_duration_phone_map[file_index][0].append(start_time)
                    file_start_duration_phone_map[file_index][1].append(duration)
                    file_start_duration_phone_map[file_index][2].append(phoneme)
                else:
                    file_start_duration_phone_map[file_index] = [[start_time], [duration], [phoneme]]
        print('Number of files: ', len(file_start_duration_phone_map))
        return file_start_duration_phone_map

    def get_aligin_kw(name, file_start_duration_phone_map, file_kw_list,
                      keyword_1=None, keyword_2=None, keyword_3=None, keyword_4=None):
        if keyword_4 is None:
            keyword_4 = []
        if keyword_3 is None:
            keyword_3 = []
        if keyword_2 is None:
            keyword_2 = []
        if keyword_1 is None:
            keyword_1 = []
        phone_len_1 = len(keyword_1)
        phone_len_2 = len(keyword_2)
        phone_len_3 = len(keyword_3)
        phone_len_4 = len(keyword_4)
        for file_index in file_start_duration_phone_map:
            file_info = file_start_duration_phone_map[file_index]
            start_list = file_info[0]
            duration_list = file_info[1]
            phone_list = file_info[2]
            check_kw_index_list = []
            check_kw_len_list = []
            for i in range(len(phone_list)):
                if len(keyword_1) > 0 and phone_list[i:i + phone_len_1] == keyword_1:
                    check_kw_index_list.append(i)
                    check_kw_len_list.append(phone_len_1)
                elif len(keyword_2) > 0 and phone_list[i:i + phone_len_2] == keyword_2:
                    check_kw_index_list.append(i)
                    check_kw_len_list.append(phone_len_2)
                elif len(keyword_3) > 0 and phone_list[i:i + phone_len_3] == keyword_3:
                    check_kw_index_list.append(i)
                    check_kw_len_list.append(phone_len_3)
                elif len(keyword_4) > 0 and phone_list[i:i + phone_len_4] == keyword_4:
                    check_kw_index_list.append(i)
                    check_kw_len_list.append(phone_len_4)
            file_info_result = []
            for i in range(len(check_kw_len_list)):
                check_kw_index = check_kw_index_list[i]
                check_kw_len = check_kw_len_list[i] - 1
                file_info_result.append({'start': start_list[check_kw_index],
                                         'duration': round(start_list[check_kw_index + check_kw_len] -
                                                           start_list[check_kw_index] +
                                                           duration_list[check_kw_index + check_kw_len], 2)})
            if file_index in file_kw_list:
                file_kw_list[file_index].append({'keyword': name, 'align': file_info_result})
            else:
                file_kw_list[file_index] = [{'keyword': name, 'align': file_info_result}]
        return file_kw_list

    def to_raw_align_file(root_dir, raw_align_file):
        file_start_duration_phone_map_ = read_align_file(root_dir)

        # kw: Việt Nam
        keyword_1_ = ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'n9_B', 'aef9_I', 'm9_E']
        keyword_2_ = ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']
        keyword_3_ = ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'n0_B', 'a0_I', 'm0_E']
        keyword_4_ = ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'sil', 'n0_B', 'a0_I', 'm0_E']
        file_kw_list_ = {}
        file_kw_list_ = get_aligin_kw('việt nam', file_start_duration_phone_map_, file_kw_list_,
                                      keyword_1_, keyword_2_, keyword_3_, keyword_4_)

        # kw: Việt
        keyword_1_ = ['v5_B', 'i5_I', 'ah5_I', 't5_E']
        file_kw_list_ = get_aligin_kw('việt', file_start_duration_phone_map_, file_kw_list_, keyword_1=keyword_1_)

        # kw: Nam
        keyword_1_ = ['n9_B', 'aef9_I', 'm9_E']
        keyword_2_ = ['n0_B', 'a0_I', 'm0_E']
        file_kw_list_ = get_aligin_kw('nam', file_start_duration_phone_map_, file_kw_list_,
                                      keyword_1=keyword_1_, keyword_2=keyword_2_)

        # Write file kw list to file:
        with open(raw_align_file, 'w') as kw_output_file:
            for align_kw in file_kw_list_:
                kw_output_file.write('{}\n'.format(
                    json.dumps({'index': align_kw, 'keywords': file_kw_list_[align_kw]}, ensure_ascii=False)))

    # Change volume----------
    def change_volume(raw_align_file, kw_align_file, raw_dir, change_volume_dir):
        file_kw_list = []
        with open(raw_align_file) as align_input:
            for line in align_input:
                file_kw_list.append(json.loads(line))

        os.makedirs(change_volume_dir, exist_ok=True)

        # change volume
        core = 8
        batch_size = len(file_kw_list) // core
        freeze_support()
        pool = Pool(core)
        for i in range(0, len(file_kw_list), batch_size):
            pool.apply_async(change_item_volume, (file_kw_list[i:i + batch_size],
                                                  raw_dir, change_volume_dir, kw_align_file))
        pool.close()
        pool.join()

    def change_item_volume(file_kw_list, raw_dir, change_volume_dir, kw_align_file):
        with open(kw_align_file, 'a') as output_file:
            for kw_info in tqdm(file_kw_list):
                file_index = kw_info['index']
                # change file index
                if file_index < 100_000:
                    file_index = str(100_000 + file_index)[1:]
                else:
                    file_index = str(1000_000 + file_index)[1:]

                # check volume
                transformer = sox.Transformer()
                input_file = os.path.join(raw_dir, '{}.wav'.format(file_index))
                stat = transformer.stat(input_file)
                rms_amplitude = float(stat['RMS amplitude'])
                is_change_volume = False
                if rms_amplitude == 0:
                    continue
                elif rms_amplitude < 0.003:
                    transformer.gain(5)
                    change_volume_input_file = os.path.join(change_volume_dir, '{}.wav'.format(file_index))
                    transformer.build_file(input_file, change_volume_input_file)
                    is_change_volume = True
                kw_info['is_change_volume'] = is_change_volume
                kw_info['duration'] = float(stat['Length (seconds)'])
                output_file.write('{}\n'.format(json.dumps(kw_info, ensure_ascii=False)))

    # Split keyword----------
    def get_align_data(kw_align_file, raw_dir, change_volume_dir, kw_index):
        align_kw_data = []
        with open(kw_align_file) as kw_input_file:
            for line in kw_input_file:
                item = json.loads(line)
                file_index = item['index']
                is_change_volume = item['is_change_volume']
                if file_index < 100_000:
                    file_index = str(100_000 + file_index)[1:]
                else:
                    file_index = str(1000_000 + file_index)[1:]
                if is_change_volume:
                    input_file = os.path.join(change_volume_dir, '{}.wav'.format(file_index))
                else:
                    input_file = os.path.join(raw_dir, '{}.wav'.format(file_index))
                align_kw_data.append({'input_file': input_file, 'align': item['keywords'][kw_index]['align'],
                                      'duration': item['duration']})
        return align_kw_data

    def split_kw(kw_dir, align_kw_data, tempo=False):
        os.makedirs(kw_dir, exist_ok=True)
        # core = 16
        # batch_size = len(align_kw_data) // core
        # freeze_support()
        # pool = Pool(core)
        # for i in range(0, len(align_kw_data), batch_size):
        #     pool.apply_async(split_item_kw, (align_kw_data[i:i + batch_size], kw_dir, tempo))
        # pool.close()
        # pool.join()
        split_item_kw(align_kw_data, kw_dir, tempo)

    def split_item_kw(align_kw_data, kw_dir, tempo=False):
        for align_kw in tqdm(align_kw_data):
            input_file = align_kw['input_file']
            time_kw_info_list = align_kw['align']
            for i, time_kw_info in enumerate(time_kw_info_list):
                start_time = time_kw_info['start']
                duration = time_kw_info['duration']
                transformer = sox.Transformer()
                if duration > 1:
                    duration = 1
                transformer.trim(start_time, start_time + duration)

                # check < 0.5 -> change time
                if tempo and duration < 0.5:
                    transformer.tempo(duration / 0.5)
                pad = (1 - duration) / 2
                transformer.pad(pad, pad)
                transformer.build_file(input_file, os.path.join(kw_dir, '{}_{}.wav'.format(
                    input_file.split('/')[-1].split('.')[0], i)))

    # Split other----------
    def split_other(unknown_dir, align_kw_data):
        os.makedirs(unknown_dir, exist_ok=True)
        core = 8
        batch_size = len(align_kw_data) // core
        freeze_support()
        pool = Pool(core)
        for i in range(0, len(align_kw_data), batch_size):
            pool.apply_async(split_item_other, (align_kw_data[i:i + batch_size], unknown_dir))
        pool.close()
        pool.join()
        # split_item_other(align_kw_data, unknown_dir)

    def split_item_other(align_kw_data, unknown_dir):
        for align_kw in tqdm(align_kw_data):
            input_file = align_kw['input_file']
            time_kw_info_list = align_kw['align']
            all_file_duration = align_kw['duration']
            kw_index_count = 0
            time_index = time_kw_info_list[0]['start']
            while time_index < all_file_duration:
                transformer = sox.Transformer()
                start_time = time_index
                end_time = start_time + 1
                end_pad = 0
                begin_pad = 0
                if kw_index_count < len(time_kw_info_list) \
                        and end_time > time_kw_info_list[kw_index_count]['start']:
                    time_kw_info = time_kw_info_list[kw_index_count]
                    kw_start_time = time_kw_info['start']
                    kw_duration = time_kw_info['duration']
                    end_time = kw_start_time
                    start_time = end_time - 1
                    kw_index_count += 1
                    time_index = kw_start_time + kw_duration
                    if start_time < 0:
                        start_time = 0
                        if end_time - start_time < 0.5:
                            continue
                        else:
                            begin_pad = 1 - (end_time - start_time)
                elif end_time > all_file_duration:
                    end_time = all_file_duration
                    time_index = end_time
                    if end_time - start_time < 0.5:
                        break
                    else:
                        end_pad = (1 - (end_time - start_time))
                else:
                    time_index += 0.5
                transformer.trim(start_time, end_time)
                if end_pad > 0:
                    transformer.pad(0, end_pad)
                if begin_pad > 0:
                    transformer.pad(begin_pad, 0)
                transformer.build_file(input_file, os.path.join(
                    unknown_dir, '{}_{}.wav'.format(input_file.split('/')[-1].split('.')[0], round(time_index, 2))))

    # Remove noise----------
    def remove_too_small_wav(dir_path, blacklist_path, threshold=0.):
        os.makedirs(blacklist_path, exist_ok=True)
        blacklist = []
        for file in tqdm(glob(os.path.join(dir_path, '*.wav'))):
            transformer = sox.Transformer()
            stat = transformer.stat(file)
            rms_amplitude = float(stat['RMS amplitude'])
            if rms_amplitude <= threshold:
                blacklist.append(file)
                shutil.move(file, os.path.join(blacklist_path, file.split('/')[-1]))
        print('Count blacklist: {}'.format(len(blacklist)))

    def remove_blacklist(vn_kw_dir, viet_kw_dir, nam_kw_dir, unknown_dir,
                         noise_vn_kw_dir, noise_viet_kw_dir, noise_nam_kw_dir, noise_unknown_dir):
        noise_ids = {106281, 32357, 32358, 10906, 32356, 32355, 17806, 36021, 17577,
                     60215, 63375, 52916, 72577, 922, 57884}
        blacklist = []
        for file in glob(os.path.join(vn_kw_dir, '*.wav')):
            file_index = int(file.split('/')[-1].split('_')[0])
            if file_index in noise_ids:
                blacklist.append(file)
                shutil.move(file, os.path.join(noise_vn_kw_dir, file.split('/')[-1]))
        for file in glob(os.path.join(viet_kw_dir, '*.wav')):
            file_index = int(file.split('/')[-1].split('_')[0])
            if file_index in noise_ids:
                blacklist.append(file)
                shutil.move(file, os.path.join(noise_viet_kw_dir, file.split('/')[-1]))
        for file in glob(os.path.join(nam_kw_dir, '*.wav')):
            file_index = int(file.split('/')[-1].split('_')[0])
            if file_index in noise_ids:
                blacklist.append(file)
                shutil.move(file, os.path.join(noise_nam_kw_dir, file.split('/')[-1]))
        for file in glob(os.path.join(unknown_dir, '*.wav')):
            file_index = int(file.split('/')[-1].split('_')[0])
            if file_index in noise_ids:
                blacklist.append(file)
                shutil.move(file, os.path.join(noise_unknown_dir, file.split('/')[-1]))
        for file in blacklist:
            os.remove(file)

    # Run----------
    root_dir_ = '/home/ubuntu/viet_nam'
    # root_dir_ = '/Users/quangbd/Documents/data/kws-data/viet_nam'
    raw_align_file_ = os.path.join(root_dir_, 'raw_kw_alignment.json')
    clean_kw_align_file_ = os.path.join(root_dir_, 'clean_kw_alignment.json')
    change_volume_dir_ = os.path.join(root_dir_, 'change_volume')
    raw_dir_ = os.path.join(root_dir_, 'raw')
    vn_kw_dir_ = os.path.join(root_dir_, 'clean/viet_nam')
    viet_kw_dir_ = os.path.join(root_dir_, 'clean/viet')
    nam_kw_dir_ = os.path.join(root_dir_, 'clean/nam')
    unknown_dir_ = os.path.join(root_dir_, 'clean/other')
    noise_vn_kw_dir_ = os.path.join(root_dir_, 'noise/viet_nam')
    noise_viet_kw_dir_ = os.path.join(root_dir_, 'noise/viet')
    noise_nam_kw_dir_ = os.path.join(root_dir_, 'noise/nam')
    noise_unknown_dir_ = os.path.join(root_dir_, 'noise/other')

    # to_raw_align_file(root_dir_, raw_align_file_)
    #
    # print('Change volume')
    # change_volume(raw_align_file_, clean_kw_align_file_, raw_dir_, change_volume_dir_)
    #
    # print('Split viet_nam')
    # align_kw_data_ = get_align_data(clean_kw_align_file_, raw_dir_, change_volume_dir_, 0)
    # split_kw(vn_kw_dir_, align_kw_data_, tempo=True)
    #
    # print('Split viet')
    # align_kw_data_ = get_align_data(clean_kw_align_file_, raw_dir_, change_volume_dir_, 1)
    # split_kw(viet_kw_dir_, align_kw_data_, tempo=False)
    #
    # print('Split nam')
    # align_kw_data_ = get_align_data(clean_kw_align_file_, raw_dir_, change_volume_dir_, 2)
    # split_kw(nam_kw_dir_, align_kw_data_, tempo=False)
    #
    # print('Split other')
    # align_kw_data_ = get_align_data(clean_kw_align_file_, raw_dir_, change_volume_dir_, 0)
    # split_other(unknown_dir_, align_kw_data_)

    print('Remove volume is too small')
    # remove_too_small_wav(vn_kw_dir_, noise_vn_kw_dir_, threshold=0.0015)
    # remove_too_small_wav(viet_kw_dir_, noise_viet_kw_dir_, threshold=0.002)
    # remove_too_small_wav(nam_kw_dir_, noise_nam_kw_dir_, threshold=0.002)
    # remove_too_small_wav(unknown_dir_, noise_unknown_dir_, threshold=0.005)

    print('Remove blacklist')
    remove_blacklist(vn_kw_dir_, viet_kw_dir_, nam_kw_dir_, unknown_dir_,
                     noise_vn_kw_dir_, noise_viet_kw_dir_, noise_nam_kw_dir_, noise_unknown_dir_)


def split_phrase():
    def read_align_file(root_dir):
        file_start_duration_phone_map = {}
        with open(os.path.join(root_dir, 'alignment.txt.ctm')) as input_file:
            for line in input_file:
                components = line.strip().split()
                file_index = components[0].split('_')[1]
                start_time = round(float(components[2]), 2)
                duration = round(float(components[3]), 2)
                phoneme = components[4]
                if file_index in file_start_duration_phone_map:
                    file_start_duration_phone_map[file_index][0].append(start_time)
                    file_start_duration_phone_map[file_index][1].append(duration)
                    file_start_duration_phone_map[file_index][2].append(phoneme)
                else:
                    file_start_duration_phone_map[file_index] = [[start_time], [duration], [phoneme]]
        print('Number of files: ', len(file_start_duration_phone_map))
        return file_start_duration_phone_map

    kw_phone_list = []
    # đông nam
    kw_phone_list.append(('dong_nam', ['dd0_B', 'oo0_I', 'ng0_E', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('dong_nam', ['dd0_B', 'oo0_I', 'ng0_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('dong_nam', ['dd0_B', 'oo0_I', 'ng0_E', 'n0_B', 'a0_I', 'm0_E']))
    kw_phone_list.append(('dong_nam', ['dd0_B', 'oo0_I', 'ng0_E', 'sil', 'n0_B', 'a0_I', 'm0_E']))

    # tây nam
    kw_phone_list.append(('tay_nam', ['t0_B', 'aa0_I', 'iz0_E', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('tay_nam', ['t0_B', 'aa0_I', 'iz0_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('tay_nam', ['t0_B', 'aa0_I', 'iz0_E', 'n0_B', 'a0_I', 'm0_E']))
    kw_phone_list.append(('tay_nam', ['t0_B', 'aa0_I', 'iz0_E', 'sil', 'n0_B', 'a0_I', 'm0_E']))

    # bắc nam
    kw_phone_list.append(('bac_nam', ['b1_B', 'aw1_I', 'k1_E', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('bac_nam', ['b1_B', 'aw1_I', 'k1_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('bac_nam', ['b1_B', 'aw1_I', 'k1_E', 'n0_B', 'a0_I', 'm0_E']))
    kw_phone_list.append(('bac_nam', ['b1_B', 'aw1_I', 'k1_E', 'sil', 'n0_B', 'a0_I', 'm0_E']))

    # hà nam
    kw_phone_list.append(('ha_nam', ['h2_B', 'a2_E', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('ha_nam', ['h2_B', 'a2_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('ha_nam', ['h2_B', 'a2_E', 'n0_B', 'a0_I', 'm0_E']))
    kw_phone_list.append(('ha_nam', ['h2_B', 'a2_E', 'sil', 'n0_B', 'a0_I', 'm0_E']))

    # vân nam
    kw_phone_list.append(('van_nam', ['v0_B', 'aa0_I', 'n0_E', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('van_nam', ['v0_B', 'aa0_I', 'n0_E', 'sil', 'n9_B', 'aef9_I', 'm9_E']))
    kw_phone_list.append(('van_nam', ['v0_B', 'aa0_I', 'n0_E', 'n0_B', 'a0_I', 'm0_E']))
    kw_phone_list.append(('van_nam', ['v0_B', 'aa0_I', 'n0_E', 'sil', 'n0_B', 'a0_I', 'm0_E']))

    # việt bắc
    kw_phone_list.append(('viet_bac', ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'sil', 'b1_B', 'aw1_I', 'k1_E']))
    kw_phone_list.append(('viet_bac', ['v5_B', 'i5_I', 'ah5_I', 't5_E', 'b1_B', 'aw1_I', 'k1_E']))

    # hà giang
    kw_phone_list.append(('ha_giang', ['h2_B', 'a2_E', 'rz0_B', 'a0_I', 'ng0_E']))
    kw_phone_list.append(('ha_giang', ['h2_B', 'a2_E', 'sil', 'rz0_B', 'a0_I', 'ng0_E']))

    file_start_duration_phone_map = read_align_file('/Users/quangbd/Desktop')
    file_kw_list = []
    for file_index in file_start_duration_phone_map:
        file_info = file_start_duration_phone_map[file_index]
        start_list = file_info[0]
        duration_list = file_info[1]
        phone_list = file_info[2]
        check_kw_index_list = []
        check_kw_len_list = []
        check_kw_list = []
        for i in range(len(phone_list)):
            for kw_phone in kw_phone_list:
                if phone_list[i:i + len(kw_phone[1])] == kw_phone[1]:
                    check_kw_index_list.append(i)
                    check_kw_len_list.append(len(kw_phone[1]))
                    check_kw_list.append(kw_phone[0])

        file_info_result = []
        for i in range(len(check_kw_len_list)):
            check_kw_index = check_kw_index_list[i]
            check_kw_len = check_kw_len_list[i] - 1
            file_info_result.append({'start': start_list[check_kw_index],
                                     'duration': round(start_list[check_kw_index + check_kw_len] -
                                                       start_list[check_kw_index] +
                                                       duration_list[check_kw_index + check_kw_len], 2),
                                     'keyword': check_kw_list[i]})
        file_kw_list.append({'file_index': file_index, 'align': file_info_result})

    for align_kw in tqdm(file_kw_list):
        input_file = os.path.join('/Users/quangbd/Desktop/data_for_quang_201104', '{}.wav'
                                  .format(align_kw['file_index']))
        time_kw_info_list = align_kw['align']
        for i, time_kw_info in enumerate(time_kw_info_list):
            start_time = time_kw_info['start']
            duration = time_kw_info['duration']
            keyword = time_kw_info['keyword']
            transformer = sox.Transformer()
            if duration > 1:
                duration = 1
            transformer.trim(start_time, start_time + duration)

            pad = (1 - duration) / 2
            transformer.pad(pad, pad)
            kw_dir = os.path.join('/Users/quangbd/Desktop', keyword)
            if not os.path.exists(kw_dir):
                os.makedirs(kw_dir, exist_ok=True)
            transformer.build_file(input_file, os.path.join(kw_dir, '{}_{}.wav'.format(
                input_file.split('/')[-1].split('.')[0], i)))


def clean_test():
    with open('/Users/quangbd/Documents/data/kws-data/viet_nam_20201103/vocal_vin.txt') as input_file:
        for line in input_file:
            line = line.split(' ', 1)
            file_index = line[0]
            content = line[1].strip().lower()
            if content == 'việt bắc':
                if os.path.exists(os.path.join('/Users/quangbd/Documents/data/kws-data/viet_nam_20201103/_vocal_', file_index)):
                    print(line)
                    shutil.move(
                        os.path.join('/Users/quangbd/Documents/data/kws-data/viet_nam_20201103/_vocal_', file_index),
                        os.path.join('/Users/quangbd/Documents/data/kws-data/viet_nam_20201103/tmp', file_index))


if __name__ == '__main__':
    clean_test()
