"""
This is used to train the rainbow motion prediction cnn.
Created in Mai 2021
Author: Armin Straller
Email: armin.straller@hs-augsburg.de
"""
import time
from os import walk
import numpy as np
import pylab as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from convert_single_scenario import tf_example_scenario

PATHNAME = 'data'

seq = tf.keras.Sequential(
    [
        tf.keras.Input(
            shape=(10, 256, 256, 1)
        ),
        layers.ConvLSTM2D(
            filters=256, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=256, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=256, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=256, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
)
seq.compile(loss="binary_crossentropy", optimizer="adadelta")

def get_scenarios_from_folder(path):
    """
    Uses the single scenario converter to convert all files in a directory.
    """
    _, _, filenames = next(walk(path))
    scenario_converter = tf_example_scenario(256, 0.5)
    grid_streams = {}
    for filename in filenames:
        start_time = time.time()
        grid_streams[path] = (scenario_converter.load('data/' + filename))
        print(">> Time to open " + filename + ":    %s s"\
        % (time.time() - start_time))
    return grid_streams, len(grid_streams)

grid_streams_dict, n_samples = get_scenarios_from_folder(PATHNAME)
print("Received ", n_samples, " samples.")
