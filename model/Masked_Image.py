#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processes images to create data to train the u-net
"""

import numpy as np
import skimage
from skimage import io
import math
import random

class Masked_Image:
    
    """An image focused on a single item, with a correct foreground mask as well as
    a predicted foreground mask for that item. 
    The methods below pre-process this image to train a foreground detector."""
    
    def __init__(self, image_file, predicted_foreground_file, correct_foreground_file):
        
        # Loads original image
        self.image = io.imread(image_file)
        
        # Handles black and white images
        if len(np.shape(self.image)) == 2:
            self.image = np.expand_dims(self.image, 2)
            self.image = self.image * np.ones((1,1,3))
            
        self.height = len(self.image)
        self.width = len(self.image[0])
        
        # Ensures the image is formatted using floats in [0,1]
        if np.amax(self.image) > 1.0:
            self.image = self.image/255.0
        
        # Loads predicted foreground
        self.predicted_foreground = io.imread(predicted_foreground_file)
        
        # Ensured the predicted foreground is formatted as a 2D array
        if len(np.shape(self.predicted_foreground))> 2:
            self.predicted_foreground = self.predicted_foreground[:, :, 0].astype(int)
        
        # Ensures the predicted foreground is the same size as the image
        self.predicted_foreground = skimage.transform.resize(self.predicted_foreground, (self.height, self.width))
        
        # Ensures the predicted foreground is an array of 1s and 0s
        if np.amax(self.predicted_foreground) > 1.0:
            self.predicted_foreground = self.predicted_foreground/np.amax(self.predicted_foreground)
        self.predicted_foreground = np.rint(self.predicted_foreground)
        
        # Loads correct foreground
        self.correct_foreground = io.imread(correct_foreground_file)
        
        # Ensures the correct foreground is formatted as a 2D array
        if len(np.shape(self.correct_foreground))> 2:
            self.correct_foreground = self.correct_foreground[:, :, 0].astype(int)

        # Ensures the correct foreground is the same size as the image
        self.correct_foreground = skimage.transform.resize(self.correct_foreground, (self.height, self.width))
        
        # Ensures the correct foreground is an array of 1s and 0s
        if np.amax(self.correct_foreground) > 1.0:
            self.correct_foreground = self.correct_foreground/np.amax(self.correct_foreground)
        self.correct_foreground = np.rint(self.correct_foreground)
        
    
    def add_padding(self, h_buffer, v_buffer):
        """Adds padding to the image and its foreground mask."""
        
        # Pads the image
        left_image_padding = np.ones((self.height, math.floor(h_buffer), 3))
        right_image_padding = np.ones((self.height, math.ceil(h_buffer), 3))

        self.image = np.concatenate((left_image_padding, np.concatenate((self.image, right_image_padding), axis=1)), axis=1)
        
        top_image_padding = self.image[:1, :, :]*np.ones((math.floor(v_buffer), self.width + 2*h_buffer, 3))
        bottom_image_padding = self.image[-1:, :, :]*np.ones((math.ceil(v_buffer), self.width + 2*h_buffer, 3))

        self.image = (np.concatenate((top_image_padding, np.concatenate((self.image, bottom_image_padding), axis=0)), axis=0))

        # Pads the predicted foreground mask
        left_foreground_padding = np.ones((self.height, math.floor(h_buffer)))
        right_foreground_padding = np.ones((self.height, math.ceil(h_buffer)))

        self.predicted_foreground = np.concatenate((left_foreground_padding, np.concatenate((self.predicted_foreground, right_foreground_padding), axis=1)), axis=1)

        top_foreground_padding = np.ones((math.floor(v_buffer), self.width + 2*h_buffer))
        bottom_foreground_padding = np.ones((math.ceil(v_buffer), self.width + 2*h_buffer))

        self.predicted_foreground = np.concatenate((top_foreground_padding, np.concatenate((self.predicted_foreground, bottom_foreground_padding), axis=0)), axis=0)

        # Pads the correct foreground mask
        self.correct_foreground = np.concatenate((left_foreground_padding, np.concatenate((self.correct_foreground, right_foreground_padding), axis=1)), axis=1)
        self.correct_foreground = np.concatenate((top_foreground_padding, np.concatenate((self.correct_foreground, bottom_foreground_padding), axis=0)), axis=0)

        # Updates the height and width of the image
        self.height = len(self.image)
        self.width = len(self.image[0])
    
    
    def find_boundary(self):
        """Finds the boundary between foreground and background. Only uses lateral adjacencies."""
        
        # Creates arrays representing the foreground mask shifted in all four directions
        foreground_left = self.predicted_foreground[:, :self.width-1]
        foreground_right = self.predicted_foreground[:, 1:]
        foreground_top = self.predicted_foreground[:self.height-1, :]
        foreground_bottom = self.predicted_foreground[1:, :]
        
        # Creates arrays marking the horizontal and vertical transitions between foreground and background
        horizontal_difference = (foreground_left != foreground_right)
        vertical_difference = (foreground_top != foreground_bottom)
        
        horizontal_difference = horizontal_difference.astype(float)
        vertical_difference = vertical_difference.astype(float)
        
        # Creates arrays marking the points where a move right/left/up/down results in
        # crossing between foreground and background
        right_difference = np.concatenate((horizontal_difference, np.zeros((self.height, 1))), axis=1)
        left_difference = np.concatenate((np.zeros((self.height, 1)), horizontal_difference), axis=1)
        top_difference = np.concatenate((vertical_difference, np.zeros((1, self.width))), axis=0)
        bottom_difference = np.concatenate((np.zeros((1, self.width)), vertical_difference), axis=0)
        
        # Takes the union of all four sets marked above, obtaining the boundary between foreground and background
        lateral_boundary = np.logical_or(np.logical_or(right_difference, left_difference), np.logical_or(top_difference, bottom_difference))

        return lateral_boundary
    
    
    def list_boundary(self):
        """Finds a list of all boundary coordinates."""
        
        boundary = self.find_boundary()
        return np.argwhere(boundary > 0)
    
    
    def find_subregion(self, row, col, dist):
        """Creates an array of shape (2dist + 1, 2dist + 1, 4), consisting of all foreground and image
        values in a square region centered at (row, col)."""
        
        # Ensures there is room to create the desired subregion
        assert row >= dist and col >= dist and row + dist <= self.height and col + dist <= self.width
        
        # Finds the subregion centered at [row, col]
        subregion = np.zeros((2*dist, 2*dist, 5))
        
        subregion[:,:,0] = self.predicted_foreground[row-dist:row+dist, col-dist:col+dist]
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
        
        # Creates a list of all boundary points, then shuffles it
        boundary_list = self.list_boundary()
        np.random.shuffle(boundary_list)
        
        # Creates a list of samples by shortening the list of boundary points, if necessary
        if len(boundary_list) > num_samples:
            sample_points = boundary_list[:num_samples]
        else:
            sample_points = boundary_list
        
        # Creates input and output arrays. 
        # Each sample in the input array consists of a given region in the image and predicted mask. 
        # The corresponding sample in the output array consists of the same region in the correct mask. 
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
