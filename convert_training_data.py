"""
This is used to convert the data from the tfrecord files.
Created in Mai 2021
Author: Armin Straller
Email: armin.straller@hs-augsburg.de
"""
import time
import os
from os import walk
import matplotlib.pyplot as plt

from convert_single_scenario import tf_example_scenario

def get_scenarios_from_folder(path):
    """
    Uses the single scenario converter to convert all files in a directory.
    """
    _, _, filenames = next(walk(path))
    scenario_converter = tf_example_scenario(128, 1)
    grid_streams = {}
    access_rights = 0o755
    print(path)
    for filename in filenames:
        start_time = time.time()
        _, file_ending = filename.split('.')
        file_path = path + '/' + file_ending
        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path, access_rights)
                grid_streams[file_ending] = (scenario_converter.load(path + '/' + filename))
                i = 0
                for stream in grid_streams[file_ending]:
                    plt.imsave(file_path + '/static_' + file_ending + '_' + str(i) + '.png', stream)
                    i += 1
        except OSError:
            print ("Creation of the directory %s failed" % file_path)
        else:
            print ("Successfully created the directory %s" % file_path)
        print(">> Time to open " + filename + ":    %s s"\
        % (time.time() - start_time))
    return grid_streams, len(grid_streams)

PATHNAME = '/media/dev/data/waymo_motion/training'
grid_streams_dict, n_samples = get_scenarios_from_folder(PATHNAME)
print(PATHNAME)
print("Received ", n_samples, " samples.")

PATHNAME = '/media/dev/data/waymo_motion/validation'
grid_streams_dict, n_samples = get_scenarios_from_folder(PATHNAME)
print(PATHNAME)
print("Received ", n_samples, " samples.")

