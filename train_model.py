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
from data_generator import DataDirectoryStorage, DataGenerator

PATHNAME_TRAINING = '/media/dev/data/waymo_motion/training'
PATHNAME_VALIDATION = '/media/dev/data/waymo_motion/validation'

# Size of the pixel grid for data rasterization
GRID_SIZE = 128

# DATA_LENGTH = 90

# The parameter TIME_STEPS_100MS sets the amount of time steps used for training. 
# By increasing this parameter the model sees more data during training. 
# More data should increase the models performance to predict vehicle movement over a wider varity of scenarios and points in time. 
# A value of 30 means that the model sees the first 3 seconds of each scenario during training.
TIME_STEPS_100MS = 30

# The parameter NUMBER_OF_FRAMES sets the amount of frames used for the cnn.
# Since only one second of history is provided by waymo, the parameter is set to 10 frames each representing 100 ms of historical data.
NUMBER_OF_FRAMES = 10

# -----------------------
# Currently only trainable with batch size one in Nvidia 1060 with 6GB of graphics memory.
# This is due to the memory required for intermediate calculations e.g. layer activation outputs.
# The required amount of memory increases linearly with batch size.

# there are two main solutions to this problem. 
# - Increasing Amount of Memory available by using graphics card with more memory available
# - splitting up the data into smaller batches and train those on different GPUs in parallel or 
#   sequentially on one GPU

# References:
# https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce
# https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa

# Gradient Accumulation
# This is a technique using smaller batches and a sequential feed to the GPU.
# The linked article describes an implementation of gradient accumulation for keras models using runai.
# https://towardsdatascience.com/how-to-easily-use-gradient-accumulation-in-keras-models-fa02c0342b60
# STEPS defines the number of steps the gradients are accumulated over
# -----------------------

batch_size = 1
GRADIENT_ACCUMULATION_STEPS = 128

# training 100 epochs this would take ~ 100 * 10h on my setup
epochs = 100

# init tf session
tf.keras.backend.clear_session()
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])

# Custom Train Step Class from 
# https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
# This enables gradient accumulation
class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

# Model 
input = tf.keras.Input(shape=(NUMBER_OF_FRAMES, GRID_SIZE, GRID_SIZE, 1), dtype="float32")
base_maps = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(5, 5), padding="same", return_sequences=True)(input)
base_maps = tf.keras.layers.BatchNormalization()(base_maps)
base_maps = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True)(base_maps)
base_maps = tf.keras.layers.BatchNormalization()(base_maps)
base_maps = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 1), padding="same", return_sequences=True)(base_maps)
base_maps = tf.keras.layers.BatchNormalization()(base_maps)
base_maps = tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(base_maps)
custom_model = CustomTrainStep(n_gradients=GRADIENT_ACCUMULATION_STEPS, inputs=[input], outputs=[base_maps])

# custom loss function focusing on bright pixels which represent objects
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

# Compile the custom model
custom_model.compile(loss=only_bright_pixels_custom, optimizer="adadelta", metrics=['mse'])
# print the model summary
custom_model.summary()

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a callback that saves the model's weights
custom_model_callbacks = [
    tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path, 
                                        verbose=0, 
                                        save_weights_only=True, 
                                        save_freq=GRADIENT_ACCUMULATION_STEPS*batch_size),
    tf.keras.callbacks.TensorBoard(     log_dir=log_dir, 
                                        update_freq=GRADIENT_ACCUMULATION_STEPS*batch_size,
                                        histogram_freq=1),
]

# saving weights
custom_model.save_weights(checkpoint_path.format(epoch=0))

# generate data ids for data generation the format is defined in DataGenerator
data_info_training = DataDirectoryStorage(PATHNAME_TRAINING, 'static_tfrecord-', TIME_STEPS_100MS, NUMBER_OF_FRAMES)
training_generator = DataGenerator(data_info_training, (NUMBER_OF_FRAMES, GRID_SIZE, GRID_SIZE, 1), batch_size)

data_info_validation = DataDirectoryStorage(PATHNAME_VALIDATION, 'static_tfrecord-', TIME_STEPS_100MS, NUMBER_OF_FRAMES)
validation_generator = DataGenerator(data_info_validation, (NUMBER_OF_FRAMES, GRID_SIZE, GRID_SIZE, 1), batch_size)

# Train model on dataset
custom_model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=custom_model_callbacks,
    use_multiprocessing=True,
    workers=6)