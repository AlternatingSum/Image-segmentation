#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains a regional U-Net to predict masks in subregions of an image with an initial predicted mask
"""

import os
import Masked_image_motorcycle as process
import numpy as np
import custom_unet as unet

# Paths for the training set
image_dir_train = "Train_CV_Test/train/Images/"
predicted_dir_train = "Train_CV_Test/train/Predicted_Masks_Fine_Tuned/"
float_dir_train = "Train_CV_Test/train/Float_Masks/"
correct_dir_train = "Train_CV_Test/train/Masks/"

# Chooses a random order for the training set, and finds its length
train_list = os.listdir(float_dir_train)
np.random.shuffle(train_list)

num_train_images = len(train_list)


# Paths for the validation set
image_dir_val = "Train_CV_Test/val/Images/"
predicted_dir_val = "Train_CV_Test/val/Predicted_Masks_Fine_Tuned/"
float_dir_val = "Train_CV_Test/val/Float_Masks/"
correct_dir_val = "Train_CV_Test/val/Masks/"

# Chooses a random order for the validation set, and finds its length
val_list = os.listdir(float_dir_val)
np.random.shuffle(val_list)

num_val_images = len(val_list)


# Defines suffixes for the image files and ground truth mask files
image_suffix = "image.jpg"
correct_suffix = "mask.png"

# Dist is half the side length of the regions, and num_sample_points is the 
# maximum number of boundary points used for each image
dist = 32
num_sample_points = 50


# Creates blank arrays for the training set and validation set
train_in = np.zeros((num_sample_points*num_train_images, 2*dist, 2*dist, 4))
train_out = np.zeros((num_sample_points*num_train_images, 2*dist, 2*dist, 1))

val_in = np.zeros((num_sample_points*num_val_images, 2*dist, 2*dist, 4))
val_out = np.zeros((num_sample_points*num_val_images, 2*dist, 2*dist, 1))

total_train = 0
total_val = 0


# Builds the training set
for predicted_name in train_list:
    
    if predicted_name[0] != "m":
        prefix = predicted_name[:-4]
        image_path = image_dir_train + prefix + image_suffix
        predicted_path = predicted_dir_train + predicted_name
        float_path = float_dir_train + predicted_name
        correct_path = correct_dir_train + prefix + correct_suffix
    
        motorcycle = process.Masked_Image(image_path, predicted_path, float_path, correct_path)
        motorcycle.add_padding(2*dist, 2*dist)
        (in_array, out_array) = motorcycle.create_subregion_dataset(dist, num_sample_points)
    
        num_new_points = len(in_array)
    
        train_in[total_train:total_train+num_new_points, :, :, :] = in_array
        train_out[total_train:total_train+num_new_points, :, :, :] = out_array
    
        total_train += num_new_points

train_in = train_in[:total_train, :, :, :]
train_out = train_out[:total_train, :, :, :]
train_out = train_out.astype(int)


# Builds the validation set
for predicted_name in val_list:
    
    if predicted_name[0] != "m":
        prefix = predicted_name[:-4]
        image_path = image_dir_val + prefix + image_suffix
        predicted_path = predicted_dir_val + predicted_name
        float_path = float_dir_val + predicted_name
        correct_path = correct_dir_val + prefix + correct_suffix
    
        motorcycle = process.Masked_Image(image_path, predicted_path, float_path, correct_path)
        motorcycle.add_padding(2*dist, 2*dist)
        (in_array, out_array) = motorcycle.create_subregion_dataset(dist, num_sample_points)
    
        num_new_points = len(in_array)
    
        val_in[total_val:total_val+num_new_points, :, :, :] = in_array
        val_out[total_val:total_val+num_new_points, :, :, :] = out_array
    
        total_val += num_new_points

val_in = val_in[:total_val, :, :, :]
val_out = val_out[:total_val, :, :, :]
val_out = val_out.astype(int)


# Defines and trains the model
model = unet.custom_unet((2*dist, 2*dist, 4), filters = 16)

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_in, train_out, epochs=15)
print(model.evaluate(val_in, val_out))

model.save('unet_16filters_32_dist_float_15epochs.h5')
