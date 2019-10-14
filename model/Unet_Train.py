#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains a U-Net to predict masks in subregions of an image with an initial predicted mask
"""
import os
import Masked_Image as process
import numpy as np

# Imports an open source u-net implementation, available here: 
# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py
import custom_unet as unet

# Loads images for the training set
image_dir_train = "../data/train/images/"
predicted_dir_train = "../data/train/tuned_masks/"
correct_dir_train = "../data/train/correct_masks/"

train_list = os.listdir(predicted_dir_train)
np.random.shuffle(train_list)

num_train_images = len(train_list)

# Loads images for the validation set
image_dir_val = "Train_CV_Test/val/images/"
predicted_dir_val = "Train_CV_Test/val/tuned_masks/"
correct_dir_val = "Train_CV_Test/val/correct_masks/"

val_list = os.listdir(predicted_dir_val)
np.random.shuffle(val_list)

num_val_images = len(val_list)

# Suffixes for file names
image_suffix = "image.jpg"
correct_suffix = "mask.png"

# The unet will take (2*dist)x(2*dist) squares as input
dist = 32

# The number of sample points used per image (use fewer if there aren't enough boundary points)
num_sample_points = 50

# Initializing the arrays for the training set and validation set
train_in = np.zeros((num_sample_points*num_train_images, 2*dist, 2*dist, 4))
train_out = np.zeros((num_sample_points*num_train_images, 2*dist, 2*dist, 1))

val_in = np.zeros((num_sample_points*num_val_images, 2*dist, 2*dist, 4))
val_out = np.zeros((num_sample_points*num_val_images, 2*dist, 2*dist, 1))

# We will log the number of samples in the training and validation set, and shorten the above arrays if necessary
total_train = 0
total_val = 0


# Creating the training set
for predicted_name in train_list:
    
    # Finding the file paths for the image, correct mask, and predicted mask
    prefix = predicted_name[:-4]
    image_path = image_dir_train + prefix + image_suffix
    predicted_path = predicted_dir_train + predicted_name
    correct_path = correct_dir_train + prefix + correct_suffix
    
    # Creating the array of samples for this particular image
    motorcycle = process.Masked_Image(image_path, predicted_path, correct_path)
    motorcycle.add_padding(2*dist, 2*dist)
    (in_array, out_array) = motorcycle.create_subregion_dataset(dist, num_sample_points)
    
    num_new_points = len(in_array)
    
    # Recording the samples for this image in the larger array of all training samples
    train_in[total_train:total_train+num_new_points, :, :, :] = in_array
    train_out[total_train:total_train+num_new_points, :, :, :] = out_array
    
    total_train += num_new_points

# Shortening the training arrays
train_in = train_in[:total_train, :, :, :]
train_out = train_out[:total_train, :, :, :]

# The output array should consist of 1s and 0s
train_out = train_out.astype(int)


# Creating the validation set
for predicted_name in val_list:
    
    # Finding the file paths for the image, correct mask, and predicted mask
    prefix = predicted_name[:-4]
    image_path = image_dir_val + prefix + image_suffix
    predicted_path = predicted_dir_val + predicted_name
    correct_path = correct_dir_val + prefix + correct_suffix
    
    # Creating the array of samples for this particular image
    motorcycle = process.Masked_Image(image_path, predicted_path, correct_path)
    motorcycle.add_padding(2*dist, 2*dist)
    (in_array, out_array) = motorcycle.create_subregion_dataset(dist, num_sample_points)
    
    num_new_points = len(in_array)
    
    # Recording the samples for this image in the larger array of all training samples
    val_in[total_val:total_val+num_new_points, :, :, :] = in_array
    val_out[total_val:total_val+num_new_points, :, :, :] = out_array
    
    total_val += num_new_points

# Shortening the validation arrays
val_in = val_in[:total_val, :, :, :]
val_out = val_out[:total_val, :, :, :]

# The output array should consist of 1s and 0s
val_out = val_out.astype(int)

# Creating and compiling the u-net. There are 4 input channels - 3 for the image, 1 for the mask. 
model = unet.custom_unet((2*dist, 2*dist, 4), filters = 16)

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Training, evaluating, and saving the u-net. 
model.fit(train_in, train_out, epochs=30)
print(model.evaluate(val_in, val_out))

model.save('../data/model_weights/unet.h5')
