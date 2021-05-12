# 
# Created by Armin Straller 
#

import math as m
import os
import uuid
import time
import sys

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

from scenario import RoadGraphType, Scenario, StateType, EgoVehicle

FILENAME = 'data/uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000'

GRID_SIZE = 256
GRID_RESOLUTION = 0.5 # meters
GRID_RESOLUTION_INV = (1/GRID_RESOLUTION)

grid = np.zeros((GRID_SIZE, GRID_SIZE))
grid.fill(0)
ego_vehicle = EgoVehicle(0,0,0)

def gridElementContainedIn2DTriangle(v1, v2, v3, x, y):
    isContained = False
    p0x, p0y = v1[0], v1[1]
    p1x, p1y = v2[0], v2[1]
    p2x, p2y = v3[0], v3[1]

    Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
    s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*x + (p0x - p2x)*y)
    t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*x + (p1x - p0x)*y)

    if s>0 and t>0 and 1-s-t>0:
        isContained = True
    
    return isContained

# returns the vehicle grid elements 
def getGridElements(x, y, bbox_yaw, length, width):
    gridElements = []

    # delta x and y in length axis
    dlx = m.cos(bbox_yaw)*length/2
    dly = m.sin(bbox_yaw)*length/2
    # delta x and y in with axis
    dwx = m.cos(bbox_yaw - m.pi/2)*width/2 
    dwy = m.sin(bbox_yaw - m.pi/2)*width/2

    boundingBox = np.array(   [[x + dlx + dwx, y + dly + dwy],
                            [x + dlx - dwx, y + dly - dwy],
                            [x - dlx + dwx, y - dly + dwy],
                            [x - dlx - dwx, y - dly - dwy]
                            ])

    # get min max values from bounding box
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for i in range(0, 4):
        if(boundingBox[i][0] > max_x):
            max_x = boundingBox[i][0]
            if max_x > 0:
                max_x = m.ceil(max_x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
            if max_x < 0:
                max_x = m.floor(max_x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
        if(boundingBox[i][1] > max_y):
            max_y = boundingBox[i][1]
            if max_y > 0:
                max_y = m.ceil(max_y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
            if max_y < 0:
                max_y = m.floor(max_y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV

        if(boundingBox[i][0] < min_x):
            min_x = boundingBox[i][0]
            if min_x > 0:
                min_x = m.floor(min_x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
            if min_x < 0:
                min_x = m.ceil(min_x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
        if(boundingBox[i][1] < min_y):
            min_y = boundingBox[i][1]
            if min_y > 0:
                min_y = m.floor(min_y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
            if min_y < 0:
                min_y = m.ceil(min_y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
    
    for x in np.arange(min_x, max_x, GRID_RESOLUTION):
        for y in np.arange(min_y, max_y, GRID_RESOLUTION):
            # TODO: Doing this with Triangles leaves some holes in the Grid.
            #       Suggested fix: Do it using the rectangle.
            bT1 = gridElementContainedIn2DTriangle( boundingBox[0], boundingBox[2], boundingBox[3], x, y)
            bT2 = gridElementContainedIn2DTriangle( boundingBox[0], boundingBox[1], boundingBox[3], x, y)
            if bT1 or bT2:
                gridElements.append((x, y))
    return gridElements

def getGridElement(x, y):
    gridElements = []
    if x > 0:
        x = m.ceil(x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
    if x < 0:
        x = m.floor(x * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV

    if y > 0:
        y = m.ceil(y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
    if y < 0:
        y = m.floor(y * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV
    
    gridElements.append((x, y))
    return gridElements

def convertGlobalToVehicleGridFrame(elements):
    e_ret = []
    for e in elements:
        dgx = e[0] - ego_vehicle.global_x
        dgy = e[1] - ego_vehicle.global_y
        e = (   m.sin(ego_vehicle.yaw - m.atan2(dgy , dgx)) * m.sqrt(m.pow(dgx, 2) + m.pow(dgy, 2)), 
                m.cos(ego_vehicle.yaw - m.atan2(dgy , dgx)) * m.sqrt(m.pow(dgx, 2) + m.pow(dgy, 2)) )
        e = ( int(GRID_SIZE/2 + round(e[0] * GRID_RESOLUTION_INV)), int (GRID_SIZE/2 + round(e[1] * GRID_RESOLUTION_INV) ))
        # e = ( int(GRID_SIZE/2 + round(e[0] * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV) , int (GRID_SIZE/2 + round(e[1] * GRID_RESOLUTION_INV) / GRID_RESOLUTION_INV) )
        if e[0] < GRID_SIZE and e[1] < GRID_SIZE and e[0] >= 0 and e[1] >= 0:
            e_ret.append(e)
    return e_ret

def addStateToGrid(x, y, bbox_yaw, length, width, status_id):
    # do a rasterization for the object state
    if length < GRID_RESOLUTION * 2:
        length = GRID_RESOLUTION * 2
    if width < GRID_RESOLUTION * 2:
        width = GRID_RESOLUTION * 2
    elements = getGridElements(x, y, bbox_yaw, length, width)
    elements = convertGlobalToVehicleGridFrame(elements)
    
    for e in elements:
        grid[int(e[0])][int(e[1])] = status_id

def addRoadGraphSampleToGrid(xyz, sample_type):
    # do a rasterization for the RoadGraphSample
    elements = getGridElement(xyz[0], xyz[1])
    elements = convertGlobalToVehicleGridFrame(elements)
    for e in elements:
        grid[int(e[0])][int(e[1])] = sample_type

def evaluatePast(scenario, ref):
    if ego_vehicle.init:
        i = 0
        for valid in scenario.get_parsed_data()['state/past/valid'].numpy():    
            j = 0
            for vI in valid:
                if vI == 1:
                    status = 100 + int(StateType(scenario.get_parsed_data()['state/type'].numpy()[i]))*10 + j
                    addStateToGrid  (   scenario.get_parsed_data()['state/past/x'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/past/y'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/past/bbox_yaw'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/past/length'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/past/width'].numpy()[i][j],
                                        status
                                    )
                j += 1
            i += 1
    else:
        if ref == "future":
            evaluateFutureEgoPos(scenario)
        elif ref == "past":
            evaluatePastEgoPos(scenario)
        else:
            sys.exit('Wrong Time Reference Provided! Exiting!')
        
        evaluatePast(scenario, ref)

def evaluateFuture(scenario, ref):
    if ego_vehicle.init:
        i = 0
        for valid in scenario.get_parsed_data()['state/future/valid'].numpy():    
            j = 0
            for vI in valid:
                if vI == 1:
                    status = 100 + int(StateType(scenario.get_parsed_data()['state/type'].numpy()[i]))*10 + j
                    addStateToGrid  (   scenario.get_parsed_data()['state/future/x'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/future/y'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/future/bbox_yaw'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/future/length'].numpy()[i][j],
                                        scenario.get_parsed_data()['state/future/width'].numpy()[i][j],
                                        status
                                    )
                j += 1
            i += 1
    else:
        if ref == "future":
            evaluateFutureEgoPos(scenario)
        elif ref == "past":
            evaluatePastEgoPos(scenario)
        else:
            sys.exit('Wrong Time Reference Provided! Exiting!')

        evaluateFuture(scenario, ref)

def evaluateFutureEgoPos(scenario):
    initialized_ego_vehicle = False
    i = 0
    for valid in scenario.get_parsed_data()['state/future/valid'].numpy():
        j = 0
        for vI in valid:
            if vI == 1:
                if scenario.get_parsed_data()['state/is_sdc'].numpy()[i] == 1:
                    ego_vehicle.global_x = scenario.get_parsed_data()['state/future/x'].numpy()[i][j]
                    ego_vehicle.global_y = scenario.get_parsed_data()['state/future/y'].numpy()[i][j]
                    ego_vehicle.yaw = scenario.get_parsed_data()['state/future/bbox_yaw'].numpy()[i][j]
                    initialized_ego_vehicle = True
            j += 1
        i += 1
    ego_vehicle.init = initialized_ego_vehicle

def evaluatePastEgoPos(scenario):
    initialized_ego_vehicle = False
    i = 0
    for valid in scenario.get_parsed_data()['state/past/valid'].numpy():
        j = 0
        for vI in valid:
            if vI == 1:
                if scenario.get_parsed_data()['state/is_sdc'].numpy()[i] == 1:
                    ego_vehicle.global_x = scenario.get_parsed_data()['state/past/x'].numpy()[i][j]
                    ego_vehicle.global_y = scenario.get_parsed_data()['state/past/y'].numpy()[i][j]
                    ego_vehicle.yaw = scenario.get_parsed_data()['state/past/bbox_yaw'].numpy()[i][j]
                    initialized_ego_vehicle = True
            j += 1
        i += 1
    ego_vehicle.init = initialized_ego_vehicle

def evaluateMap(scenario, ref):
    if ego_vehicle.init:
        i = 0
        for valid in scenario.get_parsed_data()['roadgraph_samples/valid'].numpy():   
            if valid == 1:
                roadgraph_type = 10 + int(RoadGraphType(scenario.get_parsed_data()['roadgraph_samples/type'].numpy()[i]))
                roadgraph_id = int(scenario.get_parsed_data()['roadgraph_samples/id'].numpy()[i])
                addRoadGraphSampleToGrid(   scenario.get_parsed_data()['roadgraph_samples/xyz'].numpy()[i],
                                            roadgraph_type
                                        )
            i += 1
    else:
        if ref == "future":
            evaluateFutureEgoPos(scenario)
        elif ref == "past":
            evaluatePastEgoPos(scenario)
        else:
            sys.exit('Wrong Time Reference Provided! Exiting!')
        
        evaluateMap(scenario, ref)

def evaluateAll(scenario, ref):
    start_time = time.time()
    evaluateMap(scenario, ref)
    print("--- Map time:        %s s ---" % (time.time() - start_time))
    start_time = time.time()
    evaluatePast(scenario, ref)
    print("--- Past time:       %s s ---" % (time.time() - start_time))
    start_time = time.time()
    evaluateFuture(scenario, ref)
    print("--- Future time:     %s s ---" % (time.time() - start_time))
    
    
def main():
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    scenario = Scenario(data)
    print("Scenario ID: " + scenario.get_scenario_id())
    
    ref = "past"
    start_time = time.time()
    evaluateAll(scenario, ref)
    print("--- Overall time:    %s s ---" % (time.time() - start_time))

    plt.imshow(grid, cmap='nipy_spectral', interpolation='none')
    plt.show()

if __name__ == "__main__":
    main()




