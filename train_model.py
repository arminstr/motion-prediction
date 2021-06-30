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
import datetime

import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


from convert_single_scenario import tf_example_scenario

PATHNAME_TRAINING = '/media/dev/data/waymo_motion/training'
PATHNAME_VALIDATION = '/media/dev/data/waymo_motion/validation'
GRID_SIZE = 128
DATA_LENGTH = 90

tf.keras.backend.clear_session()
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])


seq = tf.keras.Sequential(
    [
        tf.keras.Input(
            shape=(10, GRID_SIZE, GRID_SIZE, 1), dtype="float32"
        ),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(5, 5), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=128, kernel_size=(1, 1), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
)


def only_bright_pixels_custom(y_true, y_pred):
 
    threshold = tf.reduce_max(y_true) / 2
    mask_vehicle = tf.math.greater(y_true, threshold)
    mask_map = tf.math.less(y_true, threshold)
    # mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=len(mask.shape))
    # print("mask:", mask)
    mask_vehicle = tf.cast(mask_vehicle, dtype=tf.float32)
    mask_map = tf.cast(mask_map, dtype=tf.float32)
    y_true_masked_vehicle = mask_vehicle * y_true
    y_pred_masked_vehicle =  mask_vehicle * y_pred

    # set the true value to zero. this way all other pixels then the vehicles will go black
    y_true_masked_map = mask_map * 0.0
    y_pred_masked_map =  mask_map * y_pred

    squared_difference_vehicle = tf.square(y_true_masked_vehicle - y_pred_masked_vehicle)
    squared_difference_map = tf.square(y_true_masked_map - y_pred_masked_map)

    return tf.reduce_mean(squared_difference_vehicle, axis=1) * 0.9 + tf.reduce_mean(squared_difference_map, axis=1) * 0.1

seq.compile(loss=only_bright_pixels_custom, optimizer="adadelta", metrics=['mse'])

def load_scenarios(n_samples, path_name):
    """
    load scenarios that were saved using convert_training_data.py
    """
    n_frames=10
    frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    label_frames = np.zeros((n_samples, n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    _, directories, _ = next(walk(path_name))
    i = 0
    sample_index = 0
    for directory in directories:
        if 'static_' in directory: 
            _, _, filenames = next(walk(path_name + '/' + directory))
            temp_frames = np.zeros((DATA_LENGTH, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
            for filename in filenames:
                image = imageio.imread(path_name + '/' + directory + '/' + filename, as_gray=True)
                image = np.reshape(image, (GRID_SIZE, GRID_SIZE,1))
                filename_sections = filename.split('_')
                filename_index, _ = filename_sections[-1].split('.')
                temp_frames[int(filename_index)] = image/255

            # start at index one since frame 0 is empty
            # for shift_index in range(1, DATA_LENGTH - n_frames):
            frames[sample_index] = temp_frames[0:n_frames]
            label_frames[sample_index] = temp_frames[1:n_frames+1]
            # label_frame = np.zeros((n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
            # for j in range(n_frames):
            #     label_frame[j] = temp_frames[14 + j * 5]
            # label_frames[sample_index] = label_frame

            sample_index += 1
            if sample_index >= n_samples:
                return frames, label_frames
            i += 1

epochs = 200
batch_size = 1
n_samples_train = 1000
n_samples_val = 150

past_frames_train, label_frames_train = load_scenarios(n_samples_train, PATHNAME_TRAINING)
past_frames_validation, label_frames_validation = load_scenarios(n_samples_val, PATHNAME_VALIDATION)

print("past frames", past_frames_train[:n_samples_train].shape)
print("label frames", label_frames_train[:n_samples_train].shape)
seq.summary()

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a callback that saves the model's weights
seq_callbacks = [
    tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path, 
                                        verbose=0, 
                                        save_weights_only=True, 
                                        save_freq=batch_size*10),
    tf.keras.callbacks.TensorBoard(     log_dir=log_dir, 
                                        update_freq=batch_size,
                                        histogram_freq=1),
]

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

seq.save_weights(checkpoint_path.format(epoch=0))

seq.fit(
    past_frames_train,
    label_frames_train,
    shuffle=True,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data = (past_frames_validation, label_frames_validation),
    # validation_split=0.1,
    callbacks=seq_callbacks
)
