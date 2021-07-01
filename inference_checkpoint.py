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

scenario_name = 'tfrecord-00002-of-00150'

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

    return tf.reduce_mean(squared_difference_vehicle, axis=1) * 0.9 + tf.reduce_mean(squared_difference_map, axis=1) * 0.1

seq.compile(loss=only_bright_pixels_custom, optimizer="adadelta", metrics=['mse'])

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/static-map-100-epochs-1e-1s/cp-0100.ckpt.index"
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
    threshold_objects = (reduced_max + reduced_mean) / 2
    mask_objects = tf.math.greater(frame, threshold_objects)
    mask_objects = tf.cast(mask_objects, dtype=tf.float32)
    return mask_objects * 0.845

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
    plt.subplot(1, 5, 1)
    plt.imshow(frames[0, 6])
    plt.subplot(1, 5, 2)
    plt.imshow(frames[0, 7])
    plt.subplot(1, 5, 3)
    plt.imshow(frames[0, 8])
    plt.subplot(1, 5, 4)
    plt.imshow(frames[0, 9])
    plt.show()

    corr_img = np.zeros((4, 64, 64, 1))
    corr_img[0] = cross_image(postprocess_map(pred_frames[0, 9])[0:64, 0:64], postprocess_map(frames[0, 9])[0:64, 0:64])
    corr_img[1]  = cross_image(postprocess_map(pred_frames[0, 9])[64:128, 0:64], postprocess_map(frames[0, 9])[64:128, 0:64])
    corr_img[2]  = cross_image(postprocess_map(pred_frames[0, 9])[64:128, 64:128], postprocess_map(frames[0, 9])[64:128, 64:128])
    corr_img[3]  = cross_image(postprocess_map(pred_frames[0, 9])[0:64, 64:128], postprocess_map(frames[0, 9])[0:64, 64:128])

    unravel_max = np.zeros((4, 3))
    for i in range(0, 4):
        unravel_max[i] = np.unravel_index(np.argmax(corr_img[i]), corr_img[i].shape)

    center_offset = 32
    center = np.array((center_offset, center_offset, 0))
    radius_from_center = np.sqrt(center_offset*center_offset+center_offset*center_offset)

    unravel_max = (unravel_max - center) * radius_from_center
    map_offset =  (unravel_max.sum(axis=0)/radius_from_center) / 4

    map_shift_angle = np.arctan2(radius_from_center * (1 + unravel_max.sum(axis=0)[1]), radius_from_center * (1 + unravel_max.sum(axis=0)[0]))/4

    # print("offset: ", map_offset) # dividing this by four since we are using 4 quadrants
    # print("shift angle: ", map_shift_angle)

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

    plt.subplot(1, 5, 5)
    plt.imshow(prediction)
    return prediction

input_frames = load_scenario()
frames = input_frames[np.newaxis, 0]

pred = np.zeros((70, 128, 128, 1))
pred[0] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:9] = input_frames[0][1:10]
frames[9] = pred[0]

frames = np.expand_dims(frames, axis=0)
pred[1] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:8] = input_frames[0][2:10]
frames[8:10] = pred[0:2]

frames = np.expand_dims(frames, axis=0)
pred[2] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:7] = input_frames[0][3:10]
frames[7:10] = pred[0:3]

frames = np.expand_dims(frames, axis=0)
pred[3] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:6] = input_frames[0][4:10]
frames[6:10] = pred[0:4]

frames = np.expand_dims(frames, axis=0)
pred[4] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:5] = input_frames[0][5:10]
frames[5:10] = pred[0:5]

frames = np.expand_dims(frames, axis=0)
pred[5] = predict_future_frame(frames)

frames = np.zeros((10, 128, 128, 1))
frames[0:4] = input_frames[0][6:10]
frames[4:10] = pred[0:6]

frames = np.expand_dims(frames, axis=0)
pred[6] = predict_future_frame(frames)

# TODO: visualization with comparison to refernce images