import time
import sys
import pyaudio
import numpy as np
import librosa
from tensorflow import keras
from ascii_magic import AsciiArt
from extract import extract_features
from constants import *

p = pyaudio.PyAudio()
model = None

FORMAT = pyaudio.paFloat32
_TOTAL_SHOWERING_TIME = 0


def callback(in_data, frame_count, time_info, flag):
    global _TOTAL_SHOWERING_TIME
    # using Numpy to convert to array for processing
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = librosa.to_mono(audio_data)
    features = extract_features(audio_data, RATE).reshape(-1, 256, 173, 1)
    network_output = model.predict(
        features, verbose=0, use_multiprocessing=True, batch_size=150
    )

    class_confidences = np.mean(network_output, axis=0)

    duration = CHUNK / RATE
    if class_confidences[0] > 0.5:
        _TOTAL_SHOWERING_TIME = _TOTAL_SHOWERING_TIME + duration

    sys.stdout.write(
        "\033[KShowering Time:{time}, Classes: [showering={cf_showering},misc={cf_misc}]\r".format(
            time=_TOTAL_SHOWERING_TIME,
            cf_showering=class_confidences[0],
            cf_misc=class_confidences[1],
        )
    )

    return in_data, pyaudio.paContinue


def start_sampling():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
    )

    stream.start_stream()
    while stream.is_active():
        time.sleep(3)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        model = keras.models.load_model(sys.argv[1])
    else:
        model = keras.models.load_model("./dolphin_model")

    dolphin_img = AsciiArt.from_image("./static/icon.png")
    dolphin_img.to_terminal(columns=90)

    start_sampling()
