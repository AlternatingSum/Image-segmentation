#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialist_trainer.py

Given folders containing images and corresponding segmentation masks, 
this programs trains a specialist network for each image, meant to 
classify its pixels as foreground, background, or boundary. 

"""

import Open_Images_pre_processing as process

from tensorflow import keras

import numpy as np
import os

# TA: I'm using global variables below. Any tips for reworking this code to avoid them? 
image_folder = 'Train_CV_Test/Training_set/Images/'
mask_folder = 'Train_CV_Test/Training_set/Masks/'
weights_folder = 'Train_CV_Test/Training_set/Weights/'
num_params = 1177 # This is the number of parameters in each specialist network

def train_specialist(prefix):
    """Given an image with a mask, this trains a specialist network for that image. 
    The specialist network takes a pixel's colors and color derivatives as input, 
    and predicts whether the pixel is foreground, background, or boundary.
    The image file is identified by a prefix."""
    
    image_file = image_folder + prefix + '_image.jpg'
    mask_file = mask_folder + prefix + '_mask.png'
    
    # Given an image with a mask, the following code creates data sets for the pixels in that image
    dog = process.Masked_Image(image_file, mask_file)
    data = dog.process_image()

    train_in = data[1][0]
    train_out = data[1][1]
    cv_in = data[1][2]
    cv_out = data[1][3]

    # Builds and trains the specialist network
    model = keras.Sequential()
    model.add(keras.layers.Dense(21, activation = 'relu'))
    model.add(keras.layers.Dense(21, activation = 'relu'))
    model.add(keras.layers.Dense(10, activation = 'relu'))
    model.add(keras.layers.Dense(3, activation = 'softmax'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train_in, train_out, epochs=60)

    (loss, accuracy) = model.evaluate(cv_in, cv_out)
    
    # Now that the specialist network is trained, the following code records the network weights, 
    # reformatting them into a vector
    specialist_weights = np.zeros((num_params))
    current_param = 0
        
    for current_layer in model.layers:
        current_weights = current_layer.get_weights()
        for current_array in current_weights: 
            current_size = np.size(current_array)
            array_weights = np.reshape(current_array, current_size)
            specialist_weights[current_param:current_param + current_size] = array_weights
            current_param += current_size
            
    # This function returns the specialist network's weights, as well as its
    # performance on its cross validation set
    
    return (specialist_weights, accuracy)


def write_weights(weights, accuracy, prefix):
    """Given the weights of a neural network and that network's performance on a CV set, 
    this writes both pieces of information to a text file identified by a prefix."""
    
    new_file_path = weights_folder + prefix + '.txt'
    new_file = open(new_file_path,'w+')
    for weight in weights:
        new_file.write(str(weight))
        new_file.write(' ')
    new_file.write('\n')
    new_file.write(str(accuracy))
    new_file.close()

def train_specialists():
    """Trains a specialist for each image, and records its weights and accuracy on its CV set."""
    
    mask_list = os.listdir(mask_folder)

    for current_mask in mask_list:

        # Extracts the prefix for the image, to identify it. 
        character = current_mask.find('m')
        current_prefix = current_mask[:character-1]
        
        # Trains a specialist for the image, then records its weights and accuracy. 
        (dog_weights, dog_accuracy) = train_specialist(current_prefix)
        write_weights(dog_weights, dog_accuracy, current_prefix)
