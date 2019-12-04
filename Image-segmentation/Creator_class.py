#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creator_class.py

Defines classes used to create specialist networks (each one specialized to a particular image), 
and creator classes which create them. 

"""

import numpy as np
import matplotlib.pyplot as plt

class Creator:
    """A creator network takes an image as input, and creates a specialist network
    tailored to that image as output. 
    The creator structure and specialist structure should both be keras sequential models. 
    The number of outputs for the creator structure should be the same as the number of 
    parameters for the specialist structure."""
    
    def __init__(self, creator_structure, specialist_structure):
    
        self.creator = creator_structure
        self.specialist = Specialist(specialist_structure)
    
        self.creator.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
        
        
    def train(self, train_inputs, train_outputs, num_epochs):
        """Trains the creator model.
        The inputs should be images, and the outputs should be vectors."""
        
        self.creator.fit(train_inputs, train_outputs, epochs=num_epochs)
    
    
    def evaluate_creator(self, cv_inputs, cv_outputs):
        """Evaluates the performance of the creator model on a cross-validation dataset.
        The inputs should be images, and the outputs should be vectors."""
        
        self.creator.evaluate(cv_inputs, cv_outputs)
    
    
    def create(self, creator_input):
        """Uses the input (an image) to create a specialist network for that image."""
        
        specialist_params = self.creator.predict(creator_input, steps=1)
        specialist_params = np.reshape(specialist_params, (np.size(specialist_params)))
        
        self.specialist.update_params(specialist_params)
    
    
    def evaluate_specialist(self, creator_input, specialist_inputs, specialist_outputs):
        """Creates a specialist network for an image, and then evaluates its accuracy."""
        
        self.create(creator_input)
        self.specialist.evaluate(specialist_inputs, specialist_outputs)


class Specialist:
    """A specialist network is a neural network tailored to a particular image.
    The structure (specialist_structure) should be a keras sequential network."""
    
    def __init__(self, specialist_structure):
        
        self.specialist = specialist_structure
        
        self.specialist.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

        self.num_params = self.specialist.count_params()
    
    
    def export_params(self):
        """Return a vector listing all the model's parameters"""
        
        specialist_weights = np.zeros((self.num_params))
        current_param = 0
        
        for current_layer in self.specialist.layers:
            current_weights = current_layer.get_weights()
            for current_array in current_weights: 
                current_size = np.size(current_array)
                array_weights = np.reshape(current_array, current_size)
                specialist_weights[current_param:current_param + current_size] = array_weights
                current_param += current_size
        
        return specialist_weights
    
    
    def update_params(self, new_params):
        """Updates the model parameters to match the entries in new_params, which is a vector."""
        
        current_param = 0
    
        for layer_index in range(len(self.specialist.layers)):
            current_layer = self.specialist.get_layer(index = layer_index)
            current_weights = current_layer.get_weights()
            new_weights = []
        
            for current_array in current_weights:
                
                current_shape = np.shape(current_array)
                current_size = np.size(current_array)
                array_weights = new_params[current_param:current_param + current_size]
                current_param += current_size
                new_array = np.reshape(array_weights, current_shape)
                new_weights.append(new_array)
            
            current_layer.set_weights(new_weights)
    
    
    def evaluate(self, inputs, outputs):
        """Evaluates the specialist's accuracy on a set of inputs, given a set of desired outputs."""
        
        self.specialist.evaluate(inputs, outputs)
    
    
    def predict_segmentation(self, all_inputs, height, width):
        """Creates an image showing the predicted segmentation."""
        
        prediction = self.specialist.predict(all_inputs)
        predicted_image = np.reshape(prediction, (height, width, 3))
        
        plt.imshow(predicted_image)
        plt.show
