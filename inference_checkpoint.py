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
import scipy.signal
import cv2 as cv
from PIL import Image

import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from convert_single_scenario import tf_example_scenario

scenario_name = 'tfrecord-00001-of-00150'

PATHNAME = '/media/dev/data/waymo_motion/validation'
GRID_SIZE = 128
DATA_LENGTH = 90

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Create a basic model instance
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
    # y_true = y_true[0]
    # print(y_true)
    # y_pred = y_pred[0]
    # print(y_pred)

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

    return tf.reduce_mean(squared_difference_vehicle, axis=1) * 0.5 + tf.reduce_mean(squared_difference_map, axis=1) * 0.5

seq.compile(loss=only_bright_pixels_custom, optimizer="adadelta", metrics=['mse'])

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/dynamic-map-200-epochs-1e-1s-custom-loss-kernel-531/cp-0200.ckpt.index"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
# Loads the weights
seq.load_weights(latest)
seq.summary()

# Re-evaluate the model
print("Starting evaluation... ")

def postprocess_objects(frame):
    reduced_max = tf.reduce_max(frame)
    reduced_mean = tf.reduce_mean(frame)
    threshold_objects = reduced_max * 0.5 # (reduced_max + reduced_mean) / 2
    mask_objects = tf.math.greater(frame, threshold_objects)
    mask_objects = tf.cast(mask_objects, dtype=tf.float32)
    return mask_objects * reduced_max

def postprocess_map(frame):
    reduced_max = tf.reduce_max(frame)
    reduced_mean = tf.reduce_mean(frame)
    threshold_map = (reduced_max + reduced_mean) / 2
    # threshold_map = reduced_max * threshold
    mask_map = tf.math.less(frame, threshold_map)
    mask_map = tf.cast(mask_map, dtype=tf.float32)
    return mask_map * frame

def cross_image(im1, im2):
    """
    Uses fftconvolve of the two map reductions to calculate the global shift
    https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
    """
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = tf.cast(im1, dtype=tf.float32)
    im2_gray = tf.cast(im2, dtype=tf.float32)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    # TODO: try full mode
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
def optical_flow(im1, im2):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.1,
                        minDistance = 3,
                        blockSize = 2 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    im1_gray = tf.cast(im1, dtype=tf.float32)
    old_gray = im1_gray.numpy()
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # convert old image back to uint8
    im1_gray = tf.cast(im1, dtype=tf.uint8)
    old_gray = im1_gray.numpy()
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_gray)

    im2_gray = tf.cast(im2, dtype=tf.uint8)
    frame_gray = im2_gray.numpy()

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame_gray = cv.circle(frame_gray,(int(a),int(b)),5,color[i].tolist(),-1)
    
    img = cv.add(frame_gray,mask)
    # cv.imshow('frame',img)
    # # p0 = good_new.reshape(-1,1,2)
    # print("p0:", p0)
    # print("p1:", p1)
    # print("diff:", p0-p1)
    return img

dot_img_file = 'images/model.png'
tf.keras.utils.plot_model(seq, to_file=dot_img_file, show_shapes=True)

def last_int(x):
    y = x.split("00150_")
    z = y[1].split(".")
    return(int(z[0]))

def load_scenario(n_frames=10):
    """
    load scenarios that were saved using convert_training_data.py
    """
    frames = np.zeros((1, n_frames, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    sample_index = 0

    _, _, filenames = next(walk(PATHNAME + '/static_' + scenario_name))
    temp_frames = np.zeros((DATA_LENGTH, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    
    for filename in sorted(filenames, key = last_int)  :
        image = imageio.imread(PATHNAME + '/static_' + scenario_name + '/' + filename, as_gray=True)
        image = np.reshape(image, (GRID_SIZE,GRID_SIZE,1))
        filename_sections = filename.split('_')
        filename_index, _ = filename_sections[-1].split('.')
        temp_frames[int(filename_index)] = image/255
    
    frames[0] = temp_frames[0:n_frames]

    return frames

def predict_future_frame(frames):
    pred_frames = seq.predict(frames)

    FILEPATH = PATHNAME + '/validation_tfexample.' + scenario_name

    scenario_converter = tf_example_scenario(128, 1)
    map = scenario_converter.load_map_at(FILEPATH, [0.0, 0.0,0.0], 0)
    map = np.expand_dims(map, axis=2)/255

    objects = postprocess_objects(pred_frames[0, 9]).numpy()

    # use the section below to mask out objetcs from the map
    threshold = tf.reduce_max(objects) / 2
    mask_objects = tf.math.less(objects, threshold)
    mask_objects = tf.cast(mask_objects, dtype=tf.float32)
    map_masked = mask_objects * map

    prediction = np.add(map_masked, objects)
    # CONTROL: uncommenting the line below disables the prediction data improvement
    # prediction = pred_frames[0, 9]
    return prediction

# TODO: visualization with comparison to reference images

predicted_frames = np.zeros((30, 128, 128, 1))
input_frames = load_scenario()
updated_input = input_frames
# Predict 30 Frames
for j in range(10):
    # predict single frame and add it to the predicted_frames

    predicted_frames[j] = predict_future_frame(updated_input[np.newaxis, 0])
    updated_input = np.delete(updated_input, 0, 1)
    predicted_frame = np.array([predicted_frames[j]])
    updated = np.concatenate((updated_input[0], predicted_frame), axis=0)
    updated_input = np.zeros((1, 10, 128, 128, 1))
    updated_input[0] = updated

for i in range(1, 10):
    plt.subplot(1, 10, i)
    plt.imshow(updated_input[0][i])
plt.show()