#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processes images to create data to train the regional u-net
"""

import numpy as np
import skimage
from skimage import io
import math
import random

class Masked_Image:
    
    """An image focused on a single item, with a foreground mask for that item. 
    The methods below pre-process this image to train a foreground detector."""
    
    def __init__(self, image_file, predicted_foreground_file, float_foreground_file, correct_foreground_file):
        
        # Loading original image
        self.image = io.imread(image_file)
        
        # Handles black and white images
        if len(np.shape(self.image)) == 2:
            self.image = np.expand_dims(self.image, 2)
            self.image = self.image * np.ones((1,1,3))
            
        self.height = len(self.image)
        self.width = len(self.image[0])
        if np.amax(self.image) > 1.0:
            self.image = self.image/255.0
        
        # Loading predicted foreground
        self.predicted_foreground = io.imread(predicted_foreground_file)
        if len(np.shape(self.predicted_foreground))> 2:
            self.predicted_foreground = self.predicted_foreground[:, :, 0].astype(int)

        self.predicted_foreground = skimage.transform.resize(self.predicted_foreground, (self.height, self.width))
        if np.amax(self.predicted_foreground) > 1.0:
            self.predicted_foreground = self.predicted_foreground/np.amax(self.predicted_foreground)
        self.predicted_foreground = np.rint(self.predicted_foreground)
        
        # Loading predicted foreground float mask
        self.float_foreground = io.imread(float_foreground_file)
        if len(np.shape(self.float_foreground))> 2:
            self.float_foreground = self.float_foreground[:, :, 0]

        self.float_foreground = skimage.transform.resize(self.float_foreground, (self.height, self.width))
        if np.amax(self.float_foreground) > 1.0:
            self.float_foreground = self.float_foreground/np.amax(self.float_foreground)
        
        # Loading correct foreground
        self.correct_foreground = io.imread(correct_foreground_file)
        if len(np.shape(self.correct_foreground))> 2:
            self.correct_foreground = self.correct_foreground[:, :, 0].astype(int)
        self.correct_foreground = skimage.transform.resize(self.correct_foreground, (self.height, self.width))
        if np.amax(self.correct_foreground) > 1.0:
            self.correct_foreground = self.correct_foreground/np.amax(self.correct_foreground)
        self.correct_foreground = np.rint(self.correct_foreground)
        
    
    def add_padding(self, h_buffer, v_buffer):
        """Adds padding to the image and its foreground mask."""
        
        # Padding the image
        left_image_padding = np.ones((self.height, math.floor(h_buffer), 3))
        right_image_padding = np.ones((self.height, math.ceil(h_buffer), 3))

        self.image = np.concatenate((left_image_padding, np.concatenate((self.image, right_image_padding), axis=1)), axis=1)
        
        top_image_padding = np.ones((math.floor(v_buffer), self.width + 2*h_buffer, 3))
        bottom_image_padding = np.ones((math.ceil(v_buffer), self.width + 2*h_buffer, 3))

        self.image = (np.concatenate((top_image_padding, np.concatenate((self.image, bottom_image_padding), axis=0)), axis=0))

        # Padding the predicted foreground mask
        left_foreground_padding = np.ones((self.height, math.floor(h_buffer)))
        right_foreground_padding = np.ones((self.height, math.ceil(h_buffer)))

        self.predicted_foreground = np.concatenate((left_foreground_padding, np.concatenate((self.predicted_foreground, right_foreground_padding), axis=1)), axis=1)

        top_foreground_padding = np.ones((math.floor(v_buffer), self.width + 2*h_buffer))
        bottom_foreground_padding = np.ones((math.ceil(v_buffer), self.width + 2*h_buffer))

        self.predicted_foreground = np.concatenate((top_foreground_padding, np.concatenate((self.predicted_foreground, bottom_foreground_padding), axis=0)), axis=0)

        # Padding the float foreground mask

        self.float_foreground = np.concatenate((left_foreground_padding, np.concatenate((self.float_foreground, right_foreground_padding), axis=1)), axis=1)
        self.float_foreground = np.concatenate((top_foreground_padding, np.concatenate((self.float_foreground, bottom_foreground_padding), axis=0)), axis=0)


        # Padding the correct foreground mask

        self.correct_foreground = np.concatenate((left_foreground_padding, np.concatenate((self.correct_foreground, right_foreground_padding), axis=1)), axis=1)
        self.correct_foreground = np.concatenate((top_foreground_padding, np.concatenate((self.correct_foreground, bottom_foreground_padding), axis=0)), axis=0)

        self.height = len(self.image)
        self.width = len(self.image[0])
    
    
    def find_boundary(self):
        """Finds the boundary between foreground and background. Only uses lateral adjacencies."""
    
        self.foreground = np.reshape(self.predicted_foreground[:,:], (self.height, self.width))
        
        # Creates copies of the foreground shifted laterally
        foreground_left = self.predicted_foreground[:, :self.width-1]
        foreground_right = self.predicted_foreground[:, 1:]
        foreground_top = self.predicted_foreground[:self.height-1, :]
        foreground_bottom = self.predicted_foreground[1:, :]
        
        # Finds horizontal and vertical boundary points
        horizontal_difference = (foreground_left != foreground_right)
        horizontal_difference.astype(float)
        vertical_difference = (foreground_top != foreground_bottom)
        vertical_difference.astype(float)
        
        # Combines the horizontal and vertical differences to create the lateral boundary
        right_difference = np.concatenate((horizontal_difference, np.zeros((self.height, 1))), axis=1)
        left_difference = np.concatenate((np.zeros((self.height, 1)), horizontal_difference), axis=1)
        top_difference = np.concatenate((vertical_difference, np.zeros((1, self.width))), axis=0)
        bottom_difference = np.concatenate((np.zeros((1, self.width)), vertical_difference), axis=0)

        lateral_boundary = np.logical_or(np.logical_or(right_difference, left_difference), np.logical_or(top_difference, bottom_difference))

        return lateral_boundary
    
    
    def list_boundary(self):
        """Finds a list of all boundary coordinates."""
        
        boundary = self.find_boundary()
        return np.argwhere(boundary > 0)
    
    
    def find_subregion(self, row, col, dist):
        """Creates an array of shape (2dist, 2dist, 5), consisting of all predicted foreground, image, 
        and correct foreground values in a square region centered at (row, col)."""
        
        subregion = np.zeros((2*dist, 2*dist, 5))
        
        subregion[:,:,0] = self.float_foreground[row-dist:row+dist, col-dist:col+dist]
        subregion[:,:,1:4] = self.image[row-dist:row+dist, col-dist:col+dist, :]
        subregion[:,:,4] = self.correct_foreground[row-dist:row+dist, col-dist:col+dist]
        
        # Randomly flips some subregions horizontally
        coin = random.randrange(2)
        if  coin == 1:
            np.flip(subregion, 1)
        
        return subregion
    
    
    def create_subregion_dataset(self, dist, num_samples):
        """Creates a dataset consisting of subregions centered at the boundary. 
        If there are more than num_samples points on the boundary, then num_samples
        boundary points are chosen randomly."""
        
        # Creates the list of boundary points to be used for this dataset
        boundary_list = self.list_boundary()
        np.random.shuffle(boundary_list)
        if len(boundary_list) > num_samples:
            sample_points = boundary_list[:num_samples]
        else:
            sample_points = boundary_list
        
        # Creates arrays storing the data for each subregion
        input_array = np.zeros((num_samples, 2*dist, 2*dist, 4))
        output_array = np.zeros((num_samples, 2*dist, 2*dist, 1))
        
        for index in range(len(sample_points)):
            point = sample_points[index]
            point_row = point[0]
            point_col = point[1]
            subregion = self.find_subregion(point_row, point_col, dist)
            input_array[index, :,:,:] = subregion[:,:,:4]
            output_array[index, :,:,:] = subregion[:,:,4:]
        
        return (input_array, output_array)
