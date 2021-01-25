import time
import pyaudio
import numpy as np
from utils import *
from constant import *


def run():
    input_placeholder, ouput_placeholder, interpreter, desired_samples = load_model(args, ModelType.TFLITE)

    # Init recoder
    p = pyaudio.PyAudio()
    chunk = args.chunk_size
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=chunk)

    print('Recording')
    audio = np.zeros([desired_samples])
    for i in range(0, int(SAMPLE_RATE / chunk * args.record_time)):
        data = stream.read(chunk)
        sub_audio = np.frombuffer(data, dtype=np.int16) / 32768.
        sub_audio_len = sub_audio.shape[0]
        audio[:desired_samples - sub_audio_len] = audio[sub_audio_len:]
        audio[desired_samples - sub_audio_len:] = sub_audio

        # predict
        input_data = np.array(audio.reshape([desired_samples, 1]), dtype=np.float32)
        interpreter.set_tensor(input_placeholder, input_data)
        start = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(ouput_placeholder)[0]

        if output_data[1] > output_data[0]:
            print(output_data, time.time() - start)

    print('Done recording')
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    args = prepare_config()
    run()
