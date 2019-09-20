#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image pre-processing for foreground detection
"""

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import math

class Masked_Image:
    
    """An image focused on a single item, with a foreground mask for that item. 
    The methods below pre-process this image to train a foreground detector."""
    
    def __init__(self, image_file, foreground_file):
        
        # Loading original image
        self.image = io.imread(image_file)[:,:,0:3]
        self.height = len(self.image)
        self.width = len(self.image[0])
        if np.amax(self.image) > 1.0:
            self.image = self.image/np.amax(self.image)
        
        # Loading original foreground
        self.foreground = io.imread(foreground_file)
        if np.amax(self.foreground) > 1.0:
            self.foreground = self.foreground/np.amax(self.foreground)
    
    
    def resize(self, max_dim):
        """Resizes the image and its foreground mask to have a greater dimension of max_dim."""
        
        if self.height > self.width:
            self.image = skimage.transform.resize(self.image, (max_dim, int(self.width * max_dim/self.height)))
            self.foreground = skimage.transform.resize(self.foreground, (max_dim, int(self.width * max_dim/self.height)))
        else:
            self.image = skimage.transform.resize(self.image, (int(self.height * max_dim/self.width), max_dim))
            self.foreground = skimage.transform.resize(self.foreground, (int(self.height * max_dim/self.width), max_dim))
        self.foreground.astype('int')
        self.foreground.astype('float')
        
        self.height = len(self.image)
        self.width = len(self.image[0])
        
        
    def add_padding(self, h_buffer, v_buffer):
        """Adds padding to the image and its foreground mask."""
        
        """The following code pads the image."""
        left_image_padding = self.image[:, 0:1, :]*np.ones((1, math.floor(h_buffer), 3))
        right_image_padding = self.image[:, self.width-1:self.width, :]*np.ones((1, math.ceil(h_buffer), 3))

        self.image = np.concatenate((left_image_padding, np.concatenate((self.image, right_image_padding), axis=1)), axis=1)

        top_image_padding = self.image[0:1, :, :]*np.ones((math.floor(v_buffer), 1, 3))
        bottom_image_padding = self.image[self.height-1:self.height, :, :]*np.ones((math.ceil(v_buffer), 1, 3))

        self.image = (np.concatenate((top_image_padding, np.concatenate((self.image, bottom_image_padding), axis=0)), axis=0))

        """The following code pads the foreground mask."""
        left_foreground_padding = self.foreground[:, 0:1, :]*np.ones((1, math.floor(h_buffer), 1))
        right_foreground_padding = self.foreground[:, self.width-1:self.width, :]*np.ones((1, math.ceil(h_buffer), 1))

        self.foreground = np.concatenate((left_foreground_padding, np.concatenate((self.foreground, right_foreground_padding), axis=1)), axis=1)

        top_foreground_padding = self.foreground[0:1, :]*np.ones((math.floor(v_buffer), 1, 1))
        bottom_foreground_padding = self.foreground[self.height-1:self.height, :]*np.ones((math.ceil(v_buffer), 1, 1))

        self.foreground = np.concatenate((top_foreground_padding, np.concatenate((self.foreground, bottom_foreground_padding), axis=0)), axis=0)

        self.height = len(self.image)
        self.width = len(self.image[0])
    
    
    def smooth_image(self, st_dev):
        """Smooths the image."""
        
        self.image = skimage.filters.gaussian(self.image, sigma = st_dev, multichannel=True)


    def horiz_deriv(self, initial_image):
        """Calculates the horizontal derivative for an image. This includes the local change in 
        the 'red' parameter as x increases, for example."""
  
        height = len(initial_image)
        width = len(initial_image[0])
  
        im_shifted_left = initial_image[:, :width-2, :]
        im_shifted_right = initial_image[:, 2:, :]
  
        return (im_shifted_right - im_shifted_left)[1:height-1, :, :]


    def vert_deriv(self, initial_image):
        """Calculates the vertical derivative for an image"""
  
        height = len(initial_image)
        width = len(initial_image[0])
  
        im_shifted_up = initial_image[:height-2, :, :]
        im_shifted_down = initial_image[2:, :, :]
  
        return (im_shifted_down - im_shifted_up)[:, 1:width-1, :]
    

    def first_and_second_deriv(self):
        """Calculates the first and second derivatives for the image"""
        
        h_deriv = self.horiz_deriv(self.image)
        v_deriv = self.vert_deriv(self.image)
  
        hh_deriv = self.horiz_deriv(h_deriv)
        hv_deriv = self.vert_deriv(h_deriv)
        vh_deriv = self.horiz_deriv(v_deriv)
        vv_deriv = self.vert_deriv(v_deriv)
  
        cropped_h_deriv = h_deriv[1:-1, 1:-1, :]
        cropped_v_deriv = v_deriv[1:-1, 1:-1, :]
  
        """The following code defines an array to store all first and second derivatives, 
        along with the cropped original image."""
        
        all_derivs = np.zeros(([self.height - 4, self.width - 4, 3, 7]))
        """There are 3 color channels and 7 derivatives, including the zeroth derivative."""
  
        all_derivs[:,:,:,0] = self.image[2:-2, 2:-2, :]
        all_derivs[:,:,:,1] = cropped_h_deriv
        all_derivs[:,:,:,2] = cropped_v_deriv
        all_derivs[:,:,:,3] = hh_deriv
        all_derivs[:,:,:,4] = hv_deriv
        all_derivs[:,:,:,5] = vh_deriv
        all_derivs[:,:,:,6] = vv_deriv
  
        return all_derivs
    

    def normalize_derivs(self, derivatives):
        """Normalizes an array along the first two axes."""

        max1 = np.amax(derivatives, axis = 0)
        max2 = np.amax(max1, axis = 0)
  
        min1 = np.amin(derivatives, axis = 0)
        min2 = np.amin(min1, axis = 0)
  
        dist = (max2 - min2)/2
        mid = (max2 + min2)/2
  
        return (derivatives - mid)/dist
    
    
    def paint_derivatives(self, deriv_index):
        """Creates images of the derivatives of the original image."""
        
        image_derivs = self.first_and_second_deriv()
        image_derivs = self.normalize_derivs(image_derivs)

        plt.imshow((image_derivs[:,:,:,deriv_index]-np.amin(image_derivs[:,:,:,deriv_index]))/(np.amax(image_derivs[:,:,:,deriv_index] - np.amin(image_derivs[:,:,:,deriv_index]))))
        plt.show()
    
    
    def find_boundary(self):
        """Finds the boundary between foreground and background."""
    
        self.foreground = np.reshape(self.foreground[:,:,0:1], (self.height, self.width))

        foreground_left = self.foreground[:, :self.width-1]
        foreground_right = self.foreground[:, 1:]
        foreground_top = self.foreground[:self.height-1, :]
        foreground_bottom = self.foreground[1:, :]

        horizontal_difference = (foreground_left != foreground_right)
        vertical_difference = (foreground_top != foreground_bottom)
        vertical_difference.astype(float)
        
        right_difference = np.concatenate((horizontal_difference, np.zeros((self.height, 1))), axis=1)
        left_difference = np.concatenate((np.zeros((self.height, 1)), horizontal_difference), axis=1)
        top_difference = np.concatenate((vertical_difference, np.zeros((1, self.width))), axis=0)
        bottom_difference = np.concatenate((np.zeros((1, self.width)), vertical_difference), axis=0)

        foreground_top_left = self.foreground[:self.height-1, :self.width-1]
        foreground_top_right = self.foreground[:self.height-1, 1:]
        foreground_bottom_left = self.foreground[1:, :self.width-1]
        foreground_bottom_right = self.foreground[1:, 1:]

        lateral_boundary = np.logical_or(np.logical_or(right_difference, left_difference), np.logical_or(top_difference, bottom_difference))

        downward_difference = (foreground_top_left != foreground_bottom_right)

        upward_difference = (foreground_bottom_left != foreground_top_right)
        top_left_difference = np.concatenate((np.concatenate((downward_difference, np.zeros((self.height-1, 1))), axis=1), np.zeros((1, self.width))), axis = 0)
        top_right_difference = np.concatenate((np.concatenate((np.zeros((self.height-1, 1)), upward_difference), axis=1), np.zeros((1, self.width))), axis = 0)
        bottom_left_difference = np.concatenate((np.zeros((1, self.width)), np.concatenate((upward_difference, np.zeros((self.height-1, 1))), axis=1)), axis=0)
        bottom_right_difference = np.concatenate((np.zeros((1, self.width)), np.concatenate((np.zeros((self.height-1, 1)), downward_difference), axis=1)), axis=0)

        diagonal_boundary = np.logical_or(np.logical_or(top_left_difference, top_right_difference), np.logical_or(bottom_left_difference, bottom_right_difference))

        return np.logical_or(lateral_boundary, diagonal_boundary)
    
    
    def create_sample_set(self):
        """Creates the full set of samples for the image."""
        """The image should already be padded and blurred, if applicable."""
    
        cropped_foreground = self.foreground[2:-2,2:-2]
        boundary = self.find_boundary()
        image_derivs = self.first_and_second_deriv()
        image_derivs = self.normalize_derivs(image_derivs)
        
        """There are 3 color channels and 7 derivatives including the original image, 
        hence 21 features"""
        
        input_dim = 21
        buffer = 2

        total_samples = (self.height - 2*buffer)*(self.width - 2*buffer)

        foreground_samples = np.zeros((total_samples, input_dim))
        background_samples = np.zeros((total_samples, input_dim))
        boundary_samples = np.zeros((total_samples, input_dim))

        all_samples = np.zeros((total_samples, input_dim))

        total_foreground = 0
        total_background = 0
        total_boundary = 0

        current_sample = 0
        
        for row in range(self.height-2*buffer):
            for col in range(self.width-2*buffer):
    
                sample = image_derivs[row, col, :, :]
                unrolled_sample = sample.reshape((21))
    
                all_samples[current_sample, :] = unrolled_sample
                current_sample += 1
    
                if boundary[row][col]:
                    boundary_samples[total_boundary, :] = unrolled_sample
                    total_boundary += 1

                elif cropped_foreground[row][col][0] == 1.0:
                    foreground_samples[total_foreground, :] = unrolled_sample
                    total_foreground += 1

                else:
                    background_samples[total_background, :] = unrolled_sample
                    total_background += 1
                    
        foreground_samples = foreground_samples[:total_foreground, :]
        background_samples = background_samples[:total_background, :]
        boundary_samples = boundary_samples[:total_boundary, :]
        
        return (foreground_samples, background_samples, boundary_samples)
    
    
    def create_train_and_cv(self, foreground_samples, background_samples, boundary_samples):
        """Creates random training and cross-validation sets."""
        
        np.random.shuffle(foreground_samples)
        np.random.shuffle(background_samples)
        np.random.shuffle(boundary_samples)
        
        cv_num_samples = 100
        """This is the number of cross-validation samples of each type (foreground, background, boundary)."""
        
        foreground_train_num = len(foreground_samples) - cv_num_samples
        background_train_num = len(background_samples) - cv_num_samples
        boundary_train_num = len(boundary_samples) - cv_num_samples
        
        foreground_inputs_train = foreground_samples[:boundary_train_num, :]
        background_inputs_train = background_samples[:boundary_train_num, :]
        boundary_inputs_train = boundary_samples[:boundary_train_num, :]
        
        foreground_inputs_cv = foreground_samples[foreground_train_num:, :]
        background_inputs_cv = background_samples[background_train_num:, :]
        boundary_inputs_cv = boundary_samples[boundary_train_num:, :]
        
        foreground_outputs_train = np.zeros((3, boundary_train_num))
        foreground_outputs_cv = np.zeros((3, cv_num_samples))
        background_outputs_train = np.zeros((3, boundary_train_num))
        background_outputs_cv = np.zeros((3, cv_num_samples))
        boundary_outputs_train = np.zeros((3, boundary_train_num))
        boundary_outputs_cv = np.zeros((3, cv_num_samples))

        foreground_outputs_train[0, :] = 1.0
        foreground_outputs_cv[0, :] = 1.0
        background_outputs_train[2, :] = 1.0
        background_outputs_cv[2, :] = 1.0
        boundary_outputs_train[1, :] = 1.0
        boundary_outputs_cv[1, :] = 1.0
        
        train_inputs = np.concatenate((np.concatenate((foreground_inputs_train, boundary_inputs_train), axis=0), 
                                       background_inputs_train), axis=0)
        train_outputs = np.concatenate((np.concatenate((foreground_outputs_train, boundary_outputs_train), axis=1), 
                                        background_outputs_train), axis=1)

        cv_inputs = np.concatenate((np.concatenate((foreground_inputs_cv, boundary_inputs_cv), axis=0), 
                                    background_inputs_cv), axis=0)
        cv_outputs = np.concatenate((np.concatenate((foreground_outputs_cv, boundary_outputs_cv), axis=1), 
                                     background_outputs_cv), axis=1)
        
        train_outputs = train_outputs.transpose()
        cv_outputs = cv_outputs.transpose()
        
        return (train_inputs, train_outputs, cv_inputs, cv_outputs)
    
    
    def process_image(self):
        """Begins with the original image and foregroud mask, and creates training and cv sets."""
        
        self.resize(500)
        self.add_padding((504 - fancy_plate.width)/2, (504 - fancy_plate.height)/2)
        original_image = np.copy(self.image[2:-2, 2:-2, :])
        self.smooth_image(5)
        
        samples = self.create_sample_set()
        
        return (original_image, self.create_train_and_cv(samples[0], samples[1], samples[2]))