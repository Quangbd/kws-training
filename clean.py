import os
import re
import sox
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool as Pool


def clean_vinai_data():
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
            check_word = {}
            for word_info in sentence_info:
                if word_info[2] in check_word:
                    check_word[word_info[2]] += 1
                else:
                    check_word[word_info[2]] = 0
                output_wav = '{}_{}.wav'.format(file_index, check_word[word_info[2]])
                input_wav = '{}.wav'.format(file_index)
                output_dir = os.path.join(directory, 'clean', word_info[2])
                os.makedirs(output_dir, exist_ok=True)
                duration = word_info[1]
                if duration > 1:
                    duration = 1
                tfm = sox.Transformer()
                tfm.trim(word_info[0], word_info[0] + duration)
                tfm.tempo(0.8)
                pad = (1 - duration) / 2
                tfm.pad(pad, pad)
                tfm.build_file(os.path.join(directory, 'raw', input_wav), os.path.join(output_dir, output_wav))

    script_map = {}
    dictionary = {}
    with open('/Users/quangbd/Documents/data/kws-vinai/script.txt') as input_file:
        for line in input_file:
            line = line.lower().strip()
            components = re.split('\\s+', line)
            script_map[components[0]] = components[1:]
            for word in components[1:]:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    count_other_word = 0
    for word in dictionary:
        if word != 'quang':
            count_other_word += dictionary[word]
    print('count quang word:', dictionary['quang'])
    print('count other word:', count_other_word)

    word_list = []
    with open('/Users/quangbd/Documents/data/kws-vinai/alignment.txt.ctm') as input_file:
        for line in input_file:
            components = re.split('\\s+', line.strip())
            word_list.append((components[0].split('_')[0], components[2:]))

    sentence_dict = defaultdict(list)
    for key, value in word_list:
        sentence_dict[key].append(value)

    sentence_list_ = [(key, word2list(sentence_, script_map[key])) for key, sentence_ in sentence_dict.items()]

    # Trim to a new wav file
    core = 8
    batch_size = len(sentence_list_) // core
    freeze_support()
    pool = Pool(core)
    directory_ = '/Users/quangbd/Documents/data/kws-vinai'
    for i in range(0, len(sentence_list_), batch_size):
        pool.apply_async(sentences2file, (directory_, sentence_list_[i: i + batch_size]))
    pool.close()
    pool.join()


if __name__ == '__main__':
    clean_vinai_data()
