import os
import time
import numpy as np
from utils import *
from glob import glob
from tqdm import tqdm
from constant import *
from train import init_session


def run(args, model_type):
    """
    Run test
    :param args: Args
    :param model_type: checkpoint, pb, tflite
    :return: label
    """

    # get wav files
    print('Test model type:', model_type.name, '-----')
    all_wav_paths = glob(os.path.join(args.test_dir, '*/*.wav'))
    positive_total = 0
    negative_total = 0
    for file in all_wav_paths:
        if POSITIVE_LABEL in file:
            positive_total += 1
        else:
            negative_total += 1
    print('Total {} - positive {} - negative {}'.format(len(all_wav_paths), positive_total, negative_total))

    # init model
    wav_data_placeholder = None
    tf_result = None
    desired_samples = None
    interpreter = None
    session = None
    if model_type == ModelType.CHECKPOINT or model_type == ModelType.GRAPH:
        session = init_session()
        wav_data_placeholder, tf_result, desired_samples = load_model(args, model_type, session)
    elif model_type == ModelType.TFLITE:
        wav_data_placeholder, tf_result, interpreter, desired_samples = load_model(args, model_type, session)

    # predict
    positive_count = 0
    negative_count = 0
    latency_total = 0
    latency_count = 0
    for file in tqdm(all_wav_paths):
        wav_data_all = read_file(file, session=session)
        wav_len = wav_data_all.shape[0]
        for i in range(0, wav_len, args.chunk_size):
            if i + desired_samples > wav_len:
                i = wav_len - desired_samples
            wav_data = wav_data_all[i: i + desired_samples]
            start = time.time()
            if model_type == ModelType.CHECKPOINT or model_type == ModelType.GRAPH:
                result = session.run(tf_result, feed_dict={wav_data_placeholder: wav_data})[0]
            elif model_type == ModelType.TFLITE:
                interpreter.set_tensor(wav_data_placeholder, np.array(wav_data, dtype=np.float32))
                interpreter.invoke()
                result = interpreter.get_tensor(tf_result)[0]
            else:
                raise Exception('This model type does not exists')
            latency_total += time.time() - start
            latency_count += 1
            if POSITIVE_LABEL in file and result[1] > result[0]:
                positive_count += 1
                break
            elif NEGATIVE_LABEL in file and result[1] > result[0]:
                negative_count += 1
                break
            if i + desired_samples == wav_len:
                break

    negative_count = negative_total - negative_count
    print('Result: positive {}/{} = {}% - negative {}/{} = {}% - avg latency {}s/{} = {}ms'.format(
        positive_count, positive_total, positive_count / positive_total * 100,
        negative_count, negative_total, negative_count / negative_total * 100,
        int(latency_total), latency_count, round(latency_total / latency_count * 1000, 2)))

    if session:
        session.close()


if __name__ == '__main__':
    args_ = prepare_config()
    if args_.test_model_type == 'checkpoint':
        run(args_, ModelType.CHECKPOINT)
    elif args_.test_model_type == 'pb':
        run(args_, ModelType.GRAPH)
    elif args_.test_model_type == 'tflite':
        run(args_, ModelType.TFLITE)
