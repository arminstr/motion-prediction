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
    frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    label_frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    _, directories, _ = next(walk(PATHNAME))
    i = 0
    sample_index = 0
    for directory in directories:
        _, _, filenames = next(walk(PATHNAME + '/' + directory))
        temp_frames = np.zeros((DATA_LENGTH, GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
        for filename in filenames:
            image = imageio.imread(PATHNAME + '/' + directory + '/' + filename)
            filename_sections = filename.split('_')
            filename_index, _ = filename_sections[-1].split('.')
            temp_frames[int(filename_index)] = image/255
        
        # start at index one since frame 0 is empty
        for shift_index in range(1, DATA_LENGTH - n_frames):
            frames[sample_index] = temp_frames[shift_index:shift_index+n_frames]
            label_frames[sample_index] = temp_frames[shift_index+1:shift_index+n_frames+1]
            sample_index += 1
            if sample_index >= n_samples:
                return frames, label_frames
        i += 1


past_frames, label_frames = load_scenarios(n_samples=1200)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/cp-0021.ckpt.index"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a basic model instance
seq = tf.keras.Sequential(
    [
        tf.keras.Input(
            shape=(10, GRID_SIZE, GRID_SIZE, 4), dtype="float32"
        ),
        layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=4, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
)

seq.compile(loss="binary_crossentropy", optimizer="adadelta", metrics='accuracy')

print("past frames", past_frames[:1000].shape)
print("label frames", label_frames[:1000].shape)
seq.summary()

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# Loads the weights
seq.load_weights(latest)

# Re-evaluate the model
print("Starting evaluation... ")
loss, acc = seq.evaluate(
    past_frames[1101:1111], 
    label_frames[1101:1111],
    verbose=1,
    batch_size=1
    )
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


test_index = 100
test_frames = past_frames[np.newaxis, test_index]
print(test_frames.shape)
for frame in test_frames[0]:
    plt.imshow(frame)
    plt.show()
test_frames = seq.predict(test_frames)
print(test_frames.shape)
for frame in test_frames[0]:
    plt.imshow(frame)
    plt.show()
# print(seq.predict(test_frame))
