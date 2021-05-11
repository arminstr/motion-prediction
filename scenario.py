# 
# Created by Armin Straller 
#


import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

from enum import Enum

# Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
class StateType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4
    def __int__(self):
        return self.value

# [0, 19]. LaneCenter-Freeway = 1, LaneCenter-SurfaceStreet = 2, LaneCenter-BikeLane = 3, 
# RoadLine-BrokenSingleWhite = 6, RoadLine-SolidSingleWhite = 7, RoadLine-SolidDoubleWhite = 8, 
# RoadLine-BrokenSingleYellow = 9, RoadLine-BrokenDoubleYellow = 10, Roadline-SolidSingleYellow = 11, 
# Roadline-SolidDoubleYellow=12, RoadLine-PassingDoubleYellow = 13, RoadEdgeBoundary = 15, 
# RoadEdgeMedian = 16, StopSign = 17, Crosswalk = 18, 
# SpeedBump = 19, other values are unknown types and should not be present.
class RoadGraphType(Enum):
    UNSET = 0
    LANE_CENTER_FREEWAY = 1
    LANE_CENTER_SURFACE_STREET = 2
    LANE_CENTER_BIKE_LANE = 3
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEEDBUMP = 19
    GENERIC_MAP_INFO = 20

    def __int__(self):
        return self.value

class EgoVehicle:
    global_x = 0
    global_y = 0
    yaw = 0
    init = False
    def __init__(self, x, y, yaw):
        self.global_x = x
        self.global_y = y
        self.yaw = yaw
    def __repr__(self):
        return "EgoVehicle"
    def __str__(self):
        return "EgoVehicle: " + str(self.global_x) + ", " + str(self.global_y) + ", " + str(self.yaw)

class Scenario:
    # Example field definition
    __scenario_features = {
        'scenario/id':
            tf.io.FixedLenFeature([1], tf.string, default_value=None),
    }
    # Roadgraph field definition
    __roadgraph_features = {
        'roadgraph_samples/dir':
            tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
        'roadgraph_samples/id':
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        'roadgraph_samples/type':
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        'roadgraph_samples/valid':
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        'roadgraph_samples/xyz':
            tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    }
    # Features of other agents.
    __state_features = {
        'state/id':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/tracks_to_predict':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/current/bbox_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/height':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/timestamp_micros':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/valid':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/vel_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/width':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/z':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/bbox_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/height':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/timestamp_micros':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/valid':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/vel_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/width':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/z':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/past/bbox_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/height':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/timestamp_micros':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/valid':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/vel_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/width':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/z':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    }
    # Traffic light features
    __traffic_light_features = {
        'traffic_light_state/current/state':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/valid':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/x':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/y':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/z':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/past/state':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/valid':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/x':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/y':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/z':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    }
    __features_description = {}
    __features_description.update(__scenario_features)
    __features_description.update(__roadgraph_features)
    __features_description.update(__state_features)
    __features_description.update(__traffic_light_features)

    __d = {}
    __p = {}
    __scenarioId = ''

    def __init__(self, data):
        self.__d = data
        self.__p = tf.io.parse_single_example(self.__d, self.__features_description)
        self.__scenarioId = self.__getScenarioIdFromData()
    
    def __getScenarioIdFromData(self):
        return self.__p['scenario/id'].numpy()[0].decode('utf-8')

    def get_scenario_id(self):
        return self.__scenarioId

    def get_parsed_data(self):
        return self.__p