import librosa
import numpy as np
import sys
import skimage.io
import os
from dask.array.image import imread
import h5py
from audiomentations import Compose, AddGaussianNoise, Gain
from constants import *


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def extract_features(audio_data, sample_rate):
    spec = librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate, n_fft=NFFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mels = np.log(spec + 1e-9)

    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    return img


def chunked_extract(class_index, audio_data, sample_rate):
    """Will call extract features for length of N seconds of data"""
    len_per_sample = sample_rate * SAMPLE_LENGTH_SEC
    outfolder = f"./data/class_{class_index}/"
    os.makedirs(outfolder, exist_ok=True)

    augment = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=1.0),
            Gain(min_gain_db=-4, max_gain_db=4, p=1.0),
        ]
    )

    for i in range(0, len(audio_data), int(sample_rate / 2)):
        sample = audio_data[i : i + len_per_sample]
        if len(sample) != len_per_sample:
            continue

        img = extract_features(sample, sample_rate)
        skimage.io.imsave(outfolder + str(i) + ".jpg", img)

        # augmentation
        augmented_samples = augment(samples=sample, sample_rate=sample_rate)
        aug_img = extract_features(augmented_samples, sample_rate)
        skimage.io.imsave(outfolder + str(i) + "_aug.jpg", aug_img)


if __name__ == "__main__":
    dataset = None
    class_count = len(sys.argv) - 1

    class_index_map = {}

    for i in range(1, len(sys.argv)):
        sample_class_index = i - 1

        audio_data, sample_rate = librosa.load(sys.argv[i], sr=44100)
        chunked_extract(sample_class_index, audio_data, sample_rate)
        imread(f"./data/class_{sample_class_index}/*.jpg").to_hdf5(
            sys.argv[i] + ".h5", "data"
        )
