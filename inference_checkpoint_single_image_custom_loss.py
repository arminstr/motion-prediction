"""
This is used to train the rainbow motion prediction cnn.
Created in Mai 2021
Author: Armin Straller
Email: armin.straller@hs-augsburg.de
"""

import os
from os import walk
import numpy as np
import imageio

import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


from convert_single_scenario import tf_example_scenario

PATHNAME = '/media/dev/data/waymo_motion/training'
GRID_SIZE = 128
DATA_LENGTH = 90

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def load_scenarios(n_samples, n_frames=10):
    """
    load scenarios that were saved using convert_training_data.py
    """
    frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    label_frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    _, directories, _ = next(walk(PATHNAME))
    i = 0
    sample_index = 0
    for directory in directories:
        _, _, filenames = next(walk(PATHNAME + '/' + directory))
        temp_frames = np.zeros((DATA_LENGTH, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
        for filename in filenames:
            image = imageio.imread(PATHNAME + '/' + directory + '/' + filename, as_gray=True)
            image = np.reshape(image, (GRID_SIZE,GRID_SIZE,1))
            filename_sections = filename.split('_')
            filename_index, _ = filename_sections[-1].split('.')
            temp_frames[int(filename_index)] = image/255
        
        # start at index one since frame 0 is empty
        for shift_index in range(1, DATA_LENGTH - n_frames):
            frames[sample_index] = temp_frames[shift_index:shift_index+n_frames]
            label_frames[sample_index] = temp_frames[shift_index+n_frames]
            sample_index += 1
            if sample_index >= n_samples:
                return frames, label_frames
        i += 1


past_frames, label_frames = load_scenarios(n_samples=1000)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/20210601-163046/cp-0000.ckpt.index"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a basic model instance
seq = tf.keras.Sequential(
    [
        tf.keras.Input(
            shape=(10, GRID_SIZE, GRID_SIZE, 1), dtype="float32"
        ),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
        ),
    ]
)

def only_bright_pixels_custom(y_true, y_pred):
    cv2.threshold(y_true, 200, 255, cv2.TRESH_BINARY)[1]
    cv2.threshold(y_pred, 200, 255, cv2.TRESH_BINARY)[1]
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=1)

seq.compile(loss="mean_squared_logarithmic_error", optimizer="adadelta", metrics=['mse'])


print("past frames", past_frames[:900].shape)
print("label frames", label_frames[:900].shape)
seq.summary()

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Loads the weights
seq.load_weights(latest)

# Re-evaluate the model
print("Starting evaluation... ")

test_index = 30
test_frames = past_frames[np.newaxis, test_index]
print(test_frames.shape)
plt.subplot(1, 4, 1)
plt.imshow(test_frames[0, 7])
plt.subplot(1, 4, 2)
plt.imshow(test_frames[0, 8])
plt.subplot(1, 4, 3)
plt.imshow(test_frames[0, 9])
predict_frame = seq.predict(test_frames)
print(predict_frame.shape)
plt.subplot(1, 4, 4)
plt.imshow(predict_frame[0, 0])
plt.show()
