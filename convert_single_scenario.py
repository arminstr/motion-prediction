"""
This script converts a single tf_example motion scenario into a grid representation.
Created in Mai 2021
Author: Armin Straller
Email: armin.straller@hs-augsburg.de
"""

import math as m
import time
import sys

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from scenario import RoadGraphType, Scenario, StateType, EgoVehicle

class tf_example_scenario:
    """
    Wrapper class for loading and converting a tf_example to data grid representation.
    """
    __n_samples = 90
    def __init__(self, grid_size, grid_resolution):
        """ Initialize tf_example_scenario """
        self.__grid_size = grid_size
        self.__grid_resolution = grid_resolution
        self.__grid_resolution_inv = 1 / self.__grid_resolution
        self.__ego_vehicle = EgoVehicle(0,0,0)
        self.__grids = np.zeros((self.__n_samples, self.__grid_size, self.__grid_size))

    def load(self, path):
        """ Loading the data from the tfrecord """
        dataset = tf.data.TFRecordDataset(path, compression_type='')
        data = next(dataset.as_numpy_iterator())
        scenario = Scenario(data)
        for time in range(0, self.__n_samples):
            self.__grids[time] = np.zeros((self.__grid_size, self.__grid_size))
            self.__evaluate_all(scenario, time)
        return self.__grids
    
    def load_map_at(self, path, offset, angle):
        """ Loading the data from the tfrecord at a specific vehicle pos """
        dataset = tf.data.TFRecordDataset(path, compression_type='')
        data = next(dataset.as_numpy_iterator())
        scenario = Scenario(data)
        timestep = 9
        self.__grids[timestep] = np.zeros((self.__grid_size, self.__grid_size))
        self.__evaluate_past_ego_pos(scenario, timestep)
        self.__ego_vehicle.global_x += offset[0]
        self.__ego_vehicle.global_y += offset[1]
        self.__ego_vehicle.yaw += angle
        i = 0
        for valid in scenario.get_parsed_data()['roadgraph_samples/valid'].numpy():
            if valid == 1:
                roadgraph_type = 10 + int(RoadGraphType(
                                                scenario.get_parsed_data()['roadgraph_samples/type']
                                                    .numpy()[i]))
                self.__add_road_graph_sample_to_grid(   
                                                scenario.get_parsed_data()['roadgraph_samples/xyz']
                                                        .numpy()[i],
                                                roadgraph_type,
                                                timestep
                                                    )
            i += 1
        return self.__grids[timestep]

    def __evaluate_past_ego_pos(self, scenario, timestep):
        """ Evaluates the ego vehicle pos at the provided timestep [0, 9] """
        i = 0
        for valid_all in scenario.get_parsed_data()['state/past/valid'].numpy():
            j = 0
            for valid_single_state in valid_all:
                if valid_single_state == 1 and j == timestep:
                    if scenario.get_parsed_data()['state/is_sdc'].numpy()[i] == 1:
                        self.__ego_vehicle.global_x =  scenario.get_parsed_data()['state/past/x']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.global_y =  scenario.get_parsed_data()['state/past/y']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.yaw =       scenario.get_parsed_data()['state/past/bbox_yaw']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.init = True
                j += 1
            i += 1

    def __evaluate_future_ego_pos(self, scenario, timestep):
        """ Evaluates the ego vehicle pos at the timestep[0, 79] of the future timeframe """
        i = 0
        for valid_all in scenario.get_parsed_data()['state/future/valid'].numpy():
            j = 0
            for valid_single_state in valid_all:
                if valid_single_state == 1 and j == timestep:
                    if scenario.get_parsed_data()['state/is_sdc'].numpy()[i] == 1:
                        self.__ego_vehicle.global_x =  scenario.get_parsed_data()['state/future/x']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.global_y =  scenario.get_parsed_data()['state/future/y']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.yaw =       scenario.get_parsed_data()['state/future/bbox_yaw']\
                                                        .numpy()[i][j]
                        self.__ego_vehicle.init = True
                j += 1
            i += 1

    def __add_road_graph_sample_to_grid(self, xyz, sample_type, timestep):
        """
        Do a rasterization for the RoadGraphSample.
        The timestep has to be provided in gobal time[0, 89].
        """
        elements = self.__get_grid_element(xyz[0], xyz[1])
        elements = self.__convert_global_to_vehicle_frame(elements)
        for single_element in elements:
            self.__grids[timestep][int(single_element[0])][int(single_element[1])] = sample_type

    def __evaluate_map(self, scenario, timestep):
        """
        Evaluates the map defined by scenario.
        The timestep has to be provided in gobal time[0, 89].
        """
        if self.__ego_vehicle.init:
            i = 0
            for valid in scenario.get_parsed_data()['roadgraph_samples/valid'].numpy():
                if valid == 1:
                    roadgraph_type = 10 + int(RoadGraphType(
                                                    scenario.get_parsed_data()['roadgraph_samples/type']
                                                        .numpy()[i]))
                    self.__add_road_graph_sample_to_grid(   
                                                    scenario.get_parsed_data()['roadgraph_samples/xyz']
                                                            .numpy()[i],
                                                    roadgraph_type,
                                                    timestep
                                                        )
                i += 1
        else:
            self.__evaluate_past_ego_pos(scenario, 9)

            self.__evaluate_map(scenario, timestep)
    
    def __evaluate_past(self, scenario, timestep):
        """ Evaluates the past 10 time steps """
        if self.__ego_vehicle.init and timestep < 10:
            i = 0
            for valid_all in scenario.get_parsed_data()['state/past/valid'].numpy():
                j = 0
                for valid_single_state in valid_all:
                    if valid_single_state == 1 and j == timestep:
                        status = 100 + j + 10 * int(StateType(
                                                scenario.get_parsed_data()['state/type']
                                                        .numpy()[i]))
                        self.__add_state_to_grid(       scenario.get_parsed_data()['state/past/x']
                                                        .numpy()[i][j],
                                                scenario.get_parsed_data()['state/past/y']
                                                        .numpy()[i][j],
                                                scenario.get_parsed_data()['state/past/bbox_yaw']
                                                        .numpy()[i][j],
                                                scenario.get_parsed_data()['state/past/length']
                                                        .numpy()[i][j],
                                                scenario.get_parsed_data()['state/past/width']
                                                        .numpy()[i][j],
                                                status,
                                                timestep
                                            )
                    j += 1
                i += 1
        else:
            self.__evaluate_past_ego_pos(scenario, 9)
            if timestep < 10:
                self.__evaluate_past(scenario, timestep)

    def __evaluate_future(self, scenario, timestep):
        """ Evaluates the future 80 time steps """
        if self.__ego_vehicle.init and timestep > 9:
            timestep = timestep - 10
            i = 0
            for valid_all in scenario.get_parsed_data()['state/future/valid'].numpy():
                j = 0
                for valid_single_state in valid_all:
                    if valid_single_state == 1 and j == timestep:
                        status = 100 + j + 10 * int(StateType(
                                            scenario    .get_parsed_data()['state/type']
                                                        .numpy()[i]))
                        self.__add_state_to_grid  (   scenario    .get_parsed_data()['state/future/x']
                                                        .numpy()[i][j],
                                            scenario    .get_parsed_data()['state/future/y']
                                                        .numpy()[i][j],
                                            scenario    .get_parsed_data()['state/future/bbox_yaw']
                                                        .numpy()[i][j],
                                            scenario    .get_parsed_data()['state/future/length']
                                                        .numpy()[i][j],
                                            scenario    .get_parsed_data()['state/future/width']
                                                        .numpy()[i][j],
                                            status,
                                            timestep + 10
                                        )
                    j += 1
                i += 1
        else:
            self.__evaluate_past_ego_pos(scenario, 9)
            if timestep > 9:
                self.__evaluate_future(scenario, timestep)

    def __evaluate_all(self, scenario, timestep):
        """ Evaluates all time steps and the map """
        self.__evaluate_map(scenario, timestep)
        self.__evaluate_past(scenario, timestep)
        self.__evaluate_future(scenario, timestep)
    
    def __check_grid_coordinate_contained_in_triangle(
        self,
        point_one,
        point_two,
        point_three,
        coord_x,
        coord_y
    ):
        """
        Checks if the corrdinate coord_x, coord_y is in the triangle
        described by point_one, point_two and point_three
        """
        b_is_contained = False
        p0x, p0y = point_one[0], point_one[1]
        p1x, p1y = point_two[0], point_two[1]
        p2x, p2y = point_three[0], point_three[1]

        area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
        len_s = 1/(2*area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*coord_x + (p0x - p2x)*coord_y)
        len_t = 1/(2*area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*coord_x + (p1x - p0x)*coord_y)

        if len_s>0 and len_t>0 and 1-len_s-len_t>0:
            b_is_contained = True

        return b_is_contained

    def __get_grid_elements(
        self,
        coord_x,
        coord_y,
        bbox_yaw,
        length,
        width
    ):
        """
        Finds and outputs the grid elements that are part of the passed bounding box.
        """
        grid_elements = []

        # delta x and y in length axis
        d_length = ( m.cos(bbox_yaw)*length/2, m.sin(bbox_yaw)*length/2 )
        # delta x and y in with axis
        d_width = ( m.cos(bbox_yaw - m.pi/2)*width/2, m.sin(bbox_yaw - m.pi/2)*width/2 )

        bounding_box = np.array([   [   coord_x + d_length[0] + d_width[0],
                                        coord_y + d_length[1] + d_width[1]],
                                    [   coord_x + d_length[0] - d_width[0],
                                        coord_y + d_length[1] - d_width[1]],
                                    [   coord_x - d_length[0] + d_width[0],
                                        coord_y - d_length[1] + d_width[1]],
                                    [   coord_x - d_length[0] - d_width[0],
                                        coord_y - d_length[1] - d_width[1]]
                                ])

        # get min max values from bounding box
        min_bb = ( float('inf'), float('inf') )
        max_bb = ( float('-inf'), float('-inf') )

        for i in range(0, 4):
            # check max for boundig box x
            if bounding_box[i][0] > max_bb[0] and max_bb[0] > 0:
                max_bb = ( m.ceil(bounding_box[i][0] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv, max_bb[1] )
            if bounding_box[i][0] > max_bb[0] and max_bb[0] < 0:
                max_bb = ( m.floor(bounding_box[i][0] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv, max_bb[1] )
            # check max for boundig box y
            if bounding_box[i][1] > max_bb[1] and max_bb[1] > 0:
                max_bb = ( max_bb[0], m.ceil(bounding_box[i][1] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv )
            if bounding_box[i][1] > max_bb[1] and max_bb[1] < 0:
                max_bb = ( max_bb[0], m.floor(bounding_box[i][1] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv )

            # check min for boundig box x
            if bounding_box[i][0] < min_bb[0] and min_bb[0] > 0:
                min_bb = ( m.floor(bounding_box[i][0] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv, min_bb[1] )
            if bounding_box[i][0] < min_bb[0] and min_bb[0] < 0:
                min_bb = ( m.ceil(bounding_box[i][0] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv, min_bb[1] )
            # check min for boundig box y
            if bounding_box[i][1] < min_bb[1] and min_bb[1] > 0:
                min_bb = ( min_bb[0], m.floor(bounding_box[i][1] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv )
            if bounding_box[i][1] < min_bb[1] and min_bb[1] < 0:
                min_bb = ( min_bb[0], m.ceil(bounding_box[i][1] * self.__grid_resolution_inv) /
                    self.__grid_resolution_inv )

        for x_it in np.arange(min_bb[0], max_bb[0], self.__grid_resolution):
            for y_it in np.arange(min_bb[1], max_bb[1], self.__grid_resolution):
                # TODO: Doing this with Triangles leaves some holes in the Grid.
                #       Suggested fix: Do it using the rectangle.
                b_triangle_one = self.__check_grid_coordinate_contained_in_triangle(
                                                        bounding_box[0],
                                                        bounding_box[2],
                                                        bounding_box[3],
                                                        x_it, y_it)
                b_triangle_two = self.__check_grid_coordinate_contained_in_triangle(
                                                        bounding_box[0],
                                                        bounding_box[1],
                                                        bounding_box[3],
                                                        x_it, y_it)
                if b_triangle_one or b_triangle_two:
                    grid_elements.append((x_it, y_it))

        return grid_elements

    def __get_grid_element(self, coord_x, coord_y):
        """
        Finds and outputs the grid element that matches the passed position.
        Returns an array to match the output of get_grid_elements
        """
        grid_elements = []
        if coord_x > 0:
            coord_x = m.ceil(coord_x * self.__grid_resolution_inv) / self.__grid_resolution_inv
        if coord_x < 0:
            coord_x = m.floor(coord_x * self.__grid_resolution_inv) / self.__grid_resolution_inv

        if coord_y > 0:
            coord_y = m.ceil(coord_y * self.__grid_resolution_inv) / self.__grid_resolution_inv
        if coord_y < 0:
            coord_y = m.floor(coord_y * self.__grid_resolution_inv) / self.__grid_resolution_inv

        grid_elements.append((coord_x, coord_y))
        return grid_elements

    def __convert_global_to_vehicle_frame(self, elements):
        """
        Transforms the passed elements into the vehicle coordinate frame
        specified in global variable ego_vehicle.
        """
        e_ret = []
        for single_element in elements:
            dgx = single_element[0] - self.__ego_vehicle.global_x
            dgy = single_element[1] - self.__ego_vehicle.global_y
            single_element = (  m.sin(self.__ego_vehicle.yaw - m.atan2(dgy , dgx)) *
                                m.sqrt(m.pow(dgx, 2) + m.pow(dgy, 2)),
                                m.cos(self.__ego_vehicle.yaw - m.atan2(dgy , dgx)) *
                                m.sqrt(m.pow(dgx, 2) + m.pow(dgy, 2)))
            single_element = (   int(self.__grid_size/2 + round(single_element[0] * self.__grid_resolution_inv)),
                    int (self.__grid_size/2 + round(single_element[1] * self.__grid_resolution_inv)))
            # check if the current element matches the grid size
            if single_element[0] < self.__grid_size and single_element[1] < self.__grid_size \
                and single_element[0] >= 0 and single_element[1] >= 0:
                e_ret.append(single_element)
        return e_ret

    def __add_state_to_grid(self, coord_x, coord_y, bbox_yaw, length, width, status_id, timestep):
        """ Do a rasterization for the object state """
        if length < self.__grid_resolution * 2:
            length = self.__grid_resolution * 2
        if width < self.__grid_resolution * 2:
            width = self.__grid_resolution * 2
        elements = self.__get_grid_elements(coord_x, coord_y, bbox_yaw, length, width)
        elements = self.__convert_global_to_vehicle_frame(elements)
        for single_element in elements:
            self.__grids[timestep][int(single_element[0])][int(single_element[1])] = status_id
