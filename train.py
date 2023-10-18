import sys
from tensorflow import keras
import matplotlib.pyplot as plt
import h5py
import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input
from keras.optimizers import Nadam
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
from constants import *


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({"lr": K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def train(model_file):
    showering_x = h5py.File("data/showering.wav.h5")["data"]
    showering_y = np.zeros((showering_x.shape[0], CLASS_COUNT))
    showering_y[:, 0] = 1

    other_x = h5py.File("data/other.wav.h5")["data"]
    other_y = np.zeros((other_x.shape[0], CLASS_COUNT))
    other_y[:, 1] = 1

    X = np.concatenate((showering_x, other_x))
    y = np.concatenate((showering_y, other_y))

    p = np.random.permutation(len(X))

    X_shuf = X[p]
    y_shuf = y[p]

    print(X_shuf.shape, y_shuf.shape)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=2200, decay_rate=0.9, staircase=True
    )

    if not model_file:
        i = Input(shape=INPUT_DIMENSION)
        m = Conv2D(16, (5, 5), activation="elu", padding="same")(i)
        m = MaxPooling2D()(m)
        m = Conv2D(32, (4, 4), activation="elu", padding="same")(m)
        m = MaxPooling2D()(m)
        m = Conv2D(64, (3, 3), activation="elu", padding="same")(m)
        m = MaxPooling2D()(m)
        m = Conv2D(128, (3, 3), activation="elu", padding="same")(m)
        m = MaxPooling2D()(m)
        m = Conv2D(256, (3, 3), activation="elu", padding="same")(m)
        m = MaxPooling2D()(m)
        m = Flatten()(m)
        m = Dense(256, activation="elu")(m)
        m = Dropout(0.5)(m)
        o = Dense(OUTPUT_DIMENSION, activation="softmax")(m)
        model = Model(inputs=i, outputs=o)
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Nadam(learning_rate=lr_schedule),
            metrics=["accuracy"],
        )
    else:
        model = keras.models.load_model(model_file)

    my_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./training/model.{epoch:02d}-{val_loss:.4f}.h5"
        ),
        keras.callbacks.TensorBoard(log_dir="./logs"),
        LRTensorBoard(log_dir="./logs"),
    ]

    history = model.fit(
        X_shuf,
        y_shuf,
        validation_split=0.25,
        epochs=15,
        batch_size=100,
        verbose=1,
        callbacks=my_callbacks,
    )

    save_model(model, "dolphin_model")

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    if len(sys.argv) >= 3:
        train(sys.argv[1])
    else:
        train(None)
