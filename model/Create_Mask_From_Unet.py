#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses a regional U-Net to refine a predicted segmentation mask

@author: tova
"""

from keras.models import load_model

import Masked_Image as process
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import io

# Loads the u-net weights
unet = load_model('../data/model_weights/unet_16filters_32_dist_float_15epochs.h5')

# The u-net takes square regions of size (2*buffer)x(2*buffer) as input
buffer = 32

# Since each boundary region will have its own predictions from the u-net, 
# we need to take a weighted average of these predictions. 
# We use a gaussian function with its maximum in the center of the region to create these weights. 
# Sigma is the standard deviation for this gaussian. 
sigma = 8


def create_gaussian(dist, sigma):
    """Creates a gaussian filter of size (2*dist)x(2*dist), with standard deviation sigma."""
    
    gauss_array = np.zeros((2*dist, 2*dist))
    
    for row in range(2*dist):
        for col in range(2*dist):
            radius_squared = math.sqrt((row-dist)**2 + (col-dist)**2)
            gaussian = math.exp(-radius_squared/(2*sigma))
            gauss_array[row][col] = gaussian
            
    return gauss_array


def find_iou(correct_file, predicted_mask):
    """Finds the intersection over union for a predicted mask, given the correct mask."""
    
    # Loads the correct mask
    correct_mask = io.imread(correct_file)
    
    # Ensures the correct mask is a 2D array of floats
    if len(np.shape(correct_mask)) > 2:
        correct_mask = correct_mask[:,:,0]
    correct_mask = correct_mask.astype(float)
    
    # Scales the correct mask and predicted mask so that they are both arrays of 1s and 0s
    if np.amax(correct_mask) > 0.0:
        correct_mask = correct_mask/np.amax(correct_mask)
    if np.amax(predicted_mask) > 0.0:
        predicted_mask = predicted_mask/np.amax(predicted_mask)
    
    # Calculates the intersection over union
    intersection = np.multiply(correct_mask, predicted_mask)
    union = np.maximum(correct_mask, predicted_mask)
    
    if np.sum(intersection) == 0.0:
        return 0.0
    else:
        return np.sum(intersection)/np.sum(union)
    

def create_mask(prefix):
    """Uses the u-net to create a new mask for the image represented by the prefix."""
    
    # Creates paths for the image
    image_file = "../data/test/images/" + prefix + "_image.jpg"
    tuned_prediction_file = "../data/test/tuned_masks/" + prefix + "_.jpg"
    first_prediction_file = "../data/test/first_masks/" + prefix + "_.jpg"
    float_prediction_file = "../data/test/float_masks/" + prefix + "_.jpg"
    correct_mask_file = "../data/test/masks/" + prefix + "_mask.png"
    
    # Processes the image and adds padding
    motorcycle = process.Masked_Image(image_file, tuned_prediction_file, float_prediction_file, tuned_prediction_file)
    motorcycle.add_padding(2*buffer, 2*buffer)

    # Creates a list of all boundary points in the image
    boundary_list = motorcycle.list_boundary()
    
    # Initializes an array to store the information for the new predicted mask
    height = motorcycle.height
    width = motorcycle.width
    prediction_array = np.zeros((height, width, 2))
    # For each row and column, the first entry stores the sum of all weighted predictions for that point, 
    # while the second entry stores the sum of all weights associated with those predictions. 
    # This allows us to take a weighted average after obtaining all the data, 
    # by dividing the first entry by the second entry. 
    
    # Creates an array to store the input data for each region centered at a boundary point
    region_array = np.zeros((len(boundary_list), 2*buffer, 2*buffer, 4))
    boundary_index = 0

    for boundary_point in boundary_list:
        row = boundary_point[0]
        col = boundary_point[1]
    
        region = motorcycle.find_subregion(row, col, buffer)
        region_array[boundary_index, :, :, :] = region[:,:,:4]
    
        boundary_index += 1
    
    # Uses the u-net to make new predictions for each region, and combines them to make a new mask
    total_prediction = unet.predict(region_array)
    boundary_index = 0

    for boundary_point in boundary_list:
        row = boundary_point[0]
        col = boundary_point[1]
    
        local_prediction = total_prediction[boundary_index, :, :, 0]
    
        prediction_array[row-buffer:row+buffer, col-buffer:col+buffer,0] += local_prediction * gaussian
        prediction_array[row-buffer:row+buffer, col-buffer:col+buffer,1] += gaussian
    
        boundary_index += 1
        
    # Scales the prediction array so it doesn't overwhelm the initial prediction
    prediction_array = prediction_array / 20
    
    # Incorporates the initial predicted foreground into the prediction array
    prediction_array[:,:,0] += motorcycle.float_foreground
    prediction_array[:,:,1] += np.ones((height, width))
    
    # Sets all 0s equal to 1 in the second entry in dimension 2, to avoid dividing by 0 in the next step
    prediction_array[:,:,1] += (prediction_array[:,:,1] == 0.0)
    
    # Divides the sum of all weighted predictions by the sum of all weights, 
    # obtaining a weighted average
    new_data = prediction_array[:,:,0]/prediction_array[:,:,1]
    
    # Rounds the new prediction to the nearest integer, creating a mask
    new_prediction = np.round(new_data)[2*buffer:height-2*buffer, 2*buffer:width-2*buffer]
    
    # Creates an image of the new mask
    plt.imshow(new_prediction)
    plt.show
    
    # Loads the previous predicted masks (with and without fine tuning) for comparison
    tuned_predicted_mask = io.imread(tuned_prediction_file)
    first_predicted_mask = io.imread(first_prediction_file)
    
    # Prints the IoUs for all three masks
    print("The IoU for the initial prediction was:")
    print(find_iou(correct_mask_file, first_predicted_mask))
    print("The IoU for the fine tuned prediction was:")
    print(find_iou(correct_mask_file, tuned_predicted_mask))
    print("The IoU for the u-net's prediction is:")
    print(find_iou(correct_mask_file, new_prediction))
    print()
    print("Here is the u-net's prediction:")
    
    return new_prediction


# Creates the gaussian filter
gaussian = create_gaussian(buffer, sigma)

# Creates a new mask for an image. Change the prefix to see a different image. 
# Use a prefix in data/test/first_masks/ 
prefix = "t_571"
new_prediction = create_mask(prefix)
