# Tutorial for Data Generation during fit found on:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import tensorflow as tf
from os import walk
import imageio
import matplotlib.pyplot as plt

'''
- First create a map of all valid input and matching label data.
  This should be directories of the image data in the right order.
  therfore the dimensions are the following for 3 second input:
  [{'input': ['dir-1', ... 'dir-10'], 'label': ['dir-1', ... 'dir-10']} ...]
- Each time the Data Generator is called batch size number of files are picked from the training and validation directory pool.
'''
GRID_SIZE = 128
class DataDirectoryStorage:
    """
    Stores the path information about the data.

    directory_searchphrase: The directory searchphrase has to be a string that is included in
    the pathname of training and validation data.
    It also has to end directly in front of the numeric iterator in the dir naming.
    """

    def __init__(self, path, directory_searchphrase, future_time_steps = 30, time_series_length = 10):
        self.path = path
        self.future_time_steps = future_time_steps
        self.time_series_length = time_series_length
        self.data_directory = []
        self.init_data_directory(directory_searchphrase)
        
    
    def init_data_directory(self, directory_searchphrase):
        '''
        Stores all the data directories in order
        '''
        def dir_sorting_helper(x):
            '''
            Helps sorting the directorys in numerical order according to their scenario number.
            '''
            if directory_searchphrase in x:
                y = x.replace(directory_searchphrase, '')
                z = y.split("-")
                return(int(z[0]))
            else:
                return -1
        def file_sorting_helper(x):
            '''
            Helps sorting the filenames in numerical order according to their timestep related number.
            '''
            if '.png' in x:
                y = x.replace('.png', '')
                z = y.split("_")
                return(int(z[2]))
            else:
                return -1

        _, dirs, _ = next(walk(self.path))
        for dir in sorted(dirs, key = dir_sorting_helper):
            if directory_searchphrase in dir:
                _, _, files = next(walk(self.path + '/' + dir))
                files = sorted(files, key = file_sorting_helper)
                file_paths = []
                for file in files:
                    file_paths.append(self.path + '/' + dir + '/' + file)
                for future_timestep in range(1, self.future_time_steps+1):
                    input = []
                    label = []
                    for step in range(0, self.time_series_length):
                        input.append(file_paths[future_timestep+step])
                        label.append(file_paths[future_timestep+step+1])
                    self.data_directory.append({'input': input, 'label': label})
        # print('first:', self.data_directory[0])
        # print('time_series_length:', self.data_directory[self.future_time_steps-1])
        # print(len(self.data_directory))

    @property
    def data_directory(self):
        return self.__data_directory
    
    @data_directory.setter
    def data_directory(self, data_directory):
        self.__data_directory = data_directory


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_info, dim, batch_size, shuffle=True):
        'Initialization'
        self.dim = dim
        self.data_dirs = data_info.data_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_dirs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_dirs_temp = [self.data_dirs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_dirs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_dirs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_dirs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))

        def image_from_path_helper(path):
            image = imageio.imread(path, as_gray=True)
            image = np.reshape(image, (GRID_SIZE, GRID_SIZE,1))
            return image/255

        # Generate data
        for i, data_dir_info in enumerate(data_dirs_temp):
            # Store sample
            ids = ['input', 'label']
            for j, data_path in enumerate(data_dir_info['input']):
                X[i,j] = image_from_path_helper(data_path)
            for j, data_path in enumerate(data_dir_info['label']):
                y[i,j] = image_from_path_helper(data_path)
        return X, y