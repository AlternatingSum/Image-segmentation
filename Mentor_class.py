# -*- coding: utf-8 -*-
"""
Mentor class
"""

from tensorflow import keras
import numpy as np


class Mentor:
    """This creates a neural network (mentor) capable of creating other neural networks (students), 
    with each student tailored to a specific input to the mentor network."""

    def __init__(self, mentor, student):
    
        self.mentor = mentor
        self.student = student
        self.student_weights = []
        self.student_shape = []
        self.student_sizes = []
        self.num_student_params = 0
    
        keras.optimizers.Adam(lr=1.0)
    
        self.mentor.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    
        self.student.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    
        """The following code records the shapes and sizes of the weight arrays in the student model."""
        for current_layer in self.student.layers:
            current_weights = current_layer.get_weights()
            self.student_weights.append(current_weights)
            current_shapes = []
            current_sizes = []
        
            for current_array in current_weights:
                current_shapes.append(np.shape(current_array))
                current_sizes.append(np.size(current_array))
                self.num_student_params += np.size(current_array)
            self.student_shape.append(current_shapes)
            self.student_sizes.append(current_sizes)
  
    
    def teach(self, mentor_input):
        """The mentor input takes an input and produces a student network."""
    
        student_params = self.mentor.predict(mentor_input, steps=1)
        student_params = np.reshape(student_params, (np.size(student_params)))
        param_index = 0
    
        """The following code reshapes the mentor's output to form the weights in the student model."""
        for layer_index in range(len(self.student.layers)):
            current_layer = self.student.get_layer(index = layer_index)
            current_shapes = self.student_shape[layer_index]
            current_sizes = self.student_sizes[layer_index]
            current_weights = []
        
            for array_index in range(len(current_shapes)):
                new_param_index = param_index + current_sizes[array_index]
                array_weights = student_params[param_index:new_param_index]
                param_index = new_param_index
                new_array = np.reshape(array_weights, current_shapes[array_index])
                current_weights.append(new_array)
            
            current_layer.set_weights(current_weights)


    def predict(self, mentor_input, student_inputs):
        """The mentor network takes an input and produces a student network. 
        The student network then takes in its own batch of inputs and produces new outputs."""
    
        self.teach(mentor_input)
    
        return self.student.predict(student_inputs, steps=1)


    def record_student_weights(self):
        """Records the current weights for the student network, reshaping them into a single vector."""
        student_weights = np.zeros((self.num_student_params))
        current_param = 0
        
        for current_layer in self.student.layers:
            current_weights = current_layer.get_weights()
            for current_array in current_weights: 
                current_size = np.size(current_array)
                array_weights = np.reshape(current_array, current_size)
                student_weights[current_param:current_param + current_size] = array_weights
                current_param += current_size
    
        return student_weights


    def gradient_descent_step(self, mentor_input, student_inputs, student_outputs):
        """This performs one step of gradient descent to update the mentor parameters."""
        
        """Uses the mentor's output to create the student weights."""
        self.teach(mentor_input)
        student_weights = self.record_student_weights()
        
        """Performs one step of gradient descent on the student."""
        self.student.fit(student_inputs, student_outputs, steps_per_epoch=1, epochs=1)
        student_weights = self.record_student_weights()
        student_weights = np.reshape(student_weights, (1, self.num_student_params))
        
        """Uses the updated studnet weights to perform one step of gradient descent on the mentor."""
        self.mentor.fit(mentor_input, student_weights, steps_per_epoch=1, epochs=1)