#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialist_tester.py

Given an image and a specialist network for that image, this program visualizes
the networks predictions (classifying pixels as foreground, background, or boundary).

"""

from tensorflow import keras
import numpy as np

import Creator_class
import Open_Images_pre_processing as process

mask_folder = 'Train_CV_Test/Training_set/Masks/'
image_folder = 'Train_CV_Test/Training_set/Images/'
weights_folder = 'Train_CV_Test/Training_set/Weights/'

# Change the prefix below to see the prediction for a different image
prefix = 't_6'

mask_suffix = '_mask.png'
image_suffix = '_image.jpg'
weights_suffix = '.txt'

mask_path = mask_folder + prefix + mask_suffix
image_path = image_folder + prefix + image_suffix
weights_path = weights_folder + prefix + weights_suffix

dog = process.Masked_Image(image_path, mask_path)

data = dog.process_image()
dog_samples = data[1][4]


def read_weights(file_name):
    
    file = open(file_name, 'r')
    weight_string = file.readlines()[0]
    weight_list = weight_string.split()
    weights = np.zeros((len(weight_list)))
    
    for index in range(len(weight_list)):
        weights[index] = float(weight_list[index])
    
    return weights
    

model = keras.Sequential()
model.add(keras.layers.Dense(21, input_dim = 21, activation = 'relu'))
model.add(keras.layers.Dense(21, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'relu'))
model.add(keras.layers.Dense(3, activation = 'softmax'))

dog_specialist = Creator_class.Specialist(model)
dog_specialist.update_params(read_weights(weights_path))
dog_specialist.predict_segmentation(dog_samples, dog.height-4, dog.width-4)
