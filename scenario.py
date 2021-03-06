"""
This script contains classes for scenario represenation.
Created in Mai 2021
Author: Armin Straller
"""

from enum import Enum
import tensorflow as tf

class StateType(Enum):
    """
    StateType describes the type of an obstacle
    Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
    """
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4
    def __int__(self):
        return self.value

class RoadGraphType(Enum):
    """
    RoadGraphType describes the type of an road element
    [0, 19]. LaneCenter-Freeway = 1, LaneCenter-SurfaceStreet = 2,
    LaneCenter-BikeLane = 3, RoadLine-BrokenSingleWhite = 6,
    RoadLine-SolidSingleWhite = 7, RoadLine-SolidDoubleWhite = 8,
    RoadLine-BrokenSingleYellow = 9, RoadLine-BrokenDoubleYellow = 10,
    Roadline-SolidSingleYellow = 11, Roadline-SolidDoubleYellow=12,
    RoadLine-PassingDoubleYellow = 13, RoadEdgeBoundary = 15,
    RoadEdgeMedian = 16, StopSign = 17, Crosswalk = 18,
    SpeedBump = 19, other values are unknown types and should not be present.
    """
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
    """ Stores the information of the ego vehicle. """
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
        return "EgoVehicle: " + str(self.global_x) + ", " + \
        str(self.global_y) + ", " + str(self.yaw)

class Scenario:
    """
    Stores the information of the entire scenario.
    This includes the features as defined by waymo tf_example proto.
    """
    # Example field definition
    __scenario_features = {
        'scenario/id':
            tf.io.FixedLenFeature([1], tf.string, default_value=None),
    }
    # Roadgraph field definition
    __roadgraph_features = {
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
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/valid':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/width':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/bbox_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/valid':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/width':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/past/bbox_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/valid':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/width':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    }

    __features_description = {}
    __features_description.update(__scenario_features)
    __features_description.update(__roadgraph_features)
    __features_description.update(__state_features)

    __d = {}
    __p = {}
    __scenario_id = ''

    def __init__(self, data):
        self.__d = data
        self.__p = tf.io.parse_single_example(self.__d, self.__features_description)
        self.__scenario_id = self.__get_scenario_id_from_data()

    def __get_scenario_id_from_data(self):
        return self.__p['scenario/id'].numpy()[0].decode('utf-8')

    def get_scenario_id(self):
        """ Returns the identifier of the scenario. """
        return self.__scenario_id

    def get_parsed_data(self):
        """ Returns the parsed data. """
        return self.__p
