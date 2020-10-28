import time
import pyaudio
import numpy as np
from utils import *
from config import *
import tensorflow as tf


def main():
    # Init model
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    labels = prepare_words_list(args.wanted_words.split(','))

    # Init recoder
    p = pyaudio.PyAudio()
    chunk = args.chunk_size
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=chunk)

    print('Recording')
    total_sample = SAMPLE_RATE * 1  # 1 second
    audio = np.zeros([total_sample])
    for i in range(0, int(SAMPLE_RATE / chunk * args.record_time)):
        data = stream.read(chunk)
        sub_audio = np.frombuffer(data, dtype=np.int16) / 32768.
        sub_audio_len = sub_audio.shape[0]
        print(sub_audio_len)
        audio[:total_sample - sub_audio_len] = audio[sub_audio_len:]
        audio[total_sample - sub_audio_len:] = sub_audio

        # predict
        start = time.time()
        input_data = np.array(audio.reshape([total_sample, 1]), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        re_index = int(np.argmax(output_data))
        re_score = output_data[re_index]
        print('Result: {} {} - Latency {}'.format(labels[re_index], re_score, (time.time() - start) * 1000))

    print('Done recording')
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    args = prepare_record_config()
    main()