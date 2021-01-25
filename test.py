import sox
import wandb
import datetime
import torchaudio
from utils import *
from tqdm import tqdm
from glob import glob
from constant import *
from train import init_session


def run(args, model_type):
    """
    Run test
    :param args: Args
    :param model_type: checkpoint, pb, tflite
    :return: label
    """
    print('Test model type:', model_type.name, '-----')

    # read info file
    total_positive = 0
    total_negative_time = 0
    for filename in glob(os.path.join(args.test_dir, '*/*.wav')):
        if POSITIVE_LABEL in filename:
            total_positive += 1
        elif NEGATIVE_LABEL in filename:
            transformer = sox.Transformer()
            stat = transformer.stat(filename)
            duration = float(stat['Length (seconds)'])
            total_negative_time += duration
    total_negative_time = total_negative_time / 3600  # hours
    wandb.init(project='kws-testing', name=args.name, config={'model_type': model_type.name,
                                                              'date': datetime.datetime.now(),
                                                              'total_positive': total_positive,
                                                              'total_negative_hours': total_negative_time})

    # init model
    session = None
    model = None
    if model_type == ModelType.CHECKPOINT or model_type == ModelType.GRAPH:
        session = init_session()
        wav_data_placeholder, tf_result, desired_samples = load_model(args, model_type, session)
        model = (wav_data_placeholder, tf_result, desired_samples, session)
    elif model_type == ModelType.TFLITE:
        model = load_model(args, model_type, session)

        # predict
    positive_detect_time = 0
    negative_detect_time = 0
    true_positive_rate = np.zeros(len(np.arange(MIN_TEST_THRESHOLD, MAX_TEST_THRESHOLD, RANGE_TEST_THRESHOLD)))
    false_positive_rate = np.zeros(len(np.arange(MIN_TEST_THRESHOLD, MAX_TEST_THRESHOLD, RANGE_TEST_THRESHOLD)))
    total_positive_predict_count = 0
    total_negative_predict_count = 0
    for file in glob(os.path.join(args.test_dir, '*/*.wav')):
        print(file)
        if POSITIVE_LABEL in file:
            predict_result, rate, predict_count = predict(file, model_type=model_type, model=model)
            true_positive_rate += rate
            total_positive_predict_count += predict_count
            positive_detect_time += len(predict_result)
            if len(predict_result) > 0:
                wandb.log({'positive_predict_time': positive_detect_time,
                           'positive_predict_score': predict_result[0][2] - predict_result[0][1]})
        else:
            predict_result, rate, predict_count = predict(file, model_type=model_type, model=model, chunk_size=1024,
                                                          count_threshold=5, score_threshold=0.5,
                                                          max_score_threshold=0.6)
            false_positive_rate += rate
            total_negative_predict_count += predict_count
            negative_detect_time += len(predict_result)
            wandb.log({'negative_predict_time': negative_detect_time})
        print(predict_result)

    if total_positive_predict_count == 0:
        total_positive_predict_count = 1
    if total_negative_predict_count == 0:
        total_negative_predict_count = 1
    for i, threshold in enumerate(np.arange(MIN_TEST_THRESHOLD, MAX_TEST_THRESHOLD, RANGE_TEST_THRESHOLD)):
        wandb.log({'threshold': threshold, 'true_positive_rate': true_positive_rate[i] / total_positive_predict_count,
                   'false_positive_rate': false_positive_rate[i] / total_negative_predict_count})

    print('Result: positive {}/{} = {}% - negative {}/{} hours'.format(
        positive_detect_time, total_positive, round(positive_detect_time / total_positive * 100, 2),
        negative_detect_time, total_negative_time))

    if session:
        session.close()


def predict(filename, model_type, model, output_wav_dir=None, split_time=0.5, chunk_size=64, count_threshold=0,
            score_threshold=0., max_score_threshold=0.):
    """
    Wakeup prediction
    :param max_score_threshold: The max score threshold
    :param score_threshold: The average score threshold
    :param count_threshold: The count threshold
    :param filename: The input file path
    :param model_type: The model type Tflite, Graph, Checkpoint
    :param model: Model components
    :param output_wav_dir: The output directory contains wakeup wav data
    :param split_time: The split time between wakeup times
    :param chunk_size: Chunk size between window time in 1 second
    :return: Wakeup times
    """
    session = None
    interpreter = None
    if model_type == ModelType.TFLITE:
        wav_data_placeholder, tf_result, interpreter, desired_samples = model
    else:
        wav_data_placeholder, tf_result, desired_samples, session = model
    wav_data_all = read_file(filename, session=session)
    wav_len = wav_data_all.shape[0]
    end = False
    queue = []
    info_result = []
    rate = np.zeros(len(np.arange(MIN_TEST_THRESHOLD, MAX_TEST_THRESHOLD, RANGE_TEST_THRESHOLD)))
    predict_count = 0
    for i in tqdm(range(0, wav_len, chunk_size)):
        if i + desired_samples >= wav_len:
            i = wav_len - desired_samples
            end = True
        wav_data = wav_data_all[i: i + desired_samples]
        if model_type == ModelType.TFLITE:
            interpreter.set_tensor(wav_data_placeholder, np.array(wav_data, dtype=np.float32))
            interpreter.invoke()
            result = interpreter.get_tensor(tf_result)[0]
        else:
            result = session.run(tf_result, feed_dict={wav_data_placeholder: wav_data})[0]
        current_time = i / SAMPLE_RATE
        predict_count += 1
        result[0] = result[0] / (result[1] + result[0])
        result[1] = result[1] / (result[1] + result[0])
        rate += count_result(result[1])
        if result[1] > result[0]:
            avg_score, max_score = get_score(queue)
            if len(queue) > count_threshold and max_score > max_score_threshold and avg_score > score_threshold \
                    and current_time - queue[-1][2] > split_time:  # Wakeup
                info_result.append(is_wakeup(queue, filename, output_wav_dir))
                queue = []
            elif len(queue) > 0 and current_time - queue[-1][2] > split_time:
                queue = []
            queue.append((wav_data, result, current_time))
        if end:
            avg_score, max_score = get_score(queue)
            if len(queue) > count_threshold and max_score > max_score_threshold and avg_score > score_threshold:
                info_result.append(is_wakeup(queue, filename, output_wav_dir))
            break
    return info_result, rate, predict_count


def get_score(queue):
    """
    Get max score and avg score in queue
    :param queue: The queue
    :return: Max, avg
    """
    total_score = 0
    max_score = 0
    for item in queue:
        score = item[1][1] - item[1][0]
        total_score += score
        max_score = max(max_score, score)
    if total_score > 0:
        return total_score / len(queue), max_score
    return total_score, max_score


def is_wakeup(queue, input_filename, output_wav_dir=None):
    """
    Detect wakeup
    :param queue: The wakeup queue
    :param input_filename: The input wav filename
    :param output_wav_dir: The output directory contains wav files
    :return: Start time, wakeup score, non wakeup score, queue size
    """
    best_wav_data = sorted(queue, key=lambda t: t[1][1] - t[1][0], reverse=True)[0]
    if output_wav_dir:
        torchaudio.save(os.path.join(output_wav_dir, '{}_{}_{}_{}.wav'
                                     .format(input_filename.split('/')[-1].split('.')[0], round(best_wav_data[2], 2),
                                             round((best_wav_data[1][1] - best_wav_data[1][0]) * 100, 2),
                                             len(queue))),
                        torch.reshape(torch.from_numpy(best_wav_data[0]), [-1]), SAMPLE_RATE)
    return best_wav_data[2], best_wav_data[1][0], best_wav_data[1][1], len(queue)


def predict_file(args, filepath, output_wav_dir, model_type=ModelType.TFLITE):
    """
    Predict single file
    :param args: The config
    :param filepath: The file path
    :param output_wav_dir: The directory contain wav file
    :param model_type: The model type
    :return: Wakeup times with info
    """
    predict_result = None
    if model_type == ModelType.CHECKPOINT or model_type == ModelType.GRAPH:
        session = init_session()
        wav_data_placeholder, tf_result, desired_samples = load_model(args, model_type, session)
        predict_result = predict(filepath, model_type,
                                 (wav_data_placeholder, tf_result, desired_samples, session),
                                 output_wav_dir=output_wav_dir)
        session.close()
    elif model_type == ModelType.TFLITE:
        predict_result = predict(filepath, model_type, load_model(args, model_type, session=None),
                                 output_wav_dir=output_wav_dir)
    return predict_result


def count_result(score):
    result = []
    for threshold in np.arange(MIN_TEST_THRESHOLD, MAX_TEST_THRESHOLD, RANGE_TEST_THRESHOLD):
        if score > threshold:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


if __name__ == '__main__':
    args_ = prepare_config()
    if args_.test_model_type == 'checkpoint':
        run(args_, ModelType.CHECKPOINT)
    elif args_.test_model_type == 'pb':
        run(args_, ModelType.GRAPH)
    elif args_.test_model_type == 'tflite':
        run(args_, ModelType.TFLITE)
    # print(predict_file(args_, '/Users/quangbd/Documents/data/kws/test/positive/positive_vn_1610557833.6352.wav',
    #                    '/Users/quangbd/Desktop/tmp', model_type=ModelType.TFLITE))
    # print(count_result([0.4, 0.5]))
