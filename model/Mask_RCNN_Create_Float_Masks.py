"""
This program runs Mask R-CNN's inference (after fine-tuning) on cropped images of motorcycles, 
and produces probabilistic masks. 

This code was based on the following program: 
    https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py

Here is the copyright information for the original program:

Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import mrcnn.model as modellib

# Import COCO config
from mrcnn import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


def find_motorcycle_mask(mask_results):
    # Given the results from Mask R-CNN on a cropped motorcycle image, find the mask for the central motorcycle. 
    # This also returns a boolean indicating whether the model detected a motorcycle or not. 
    
    # Get the predicted bounding boxes, masks, and object ids from the results. 
    boxes = mask_results[0]['rois']
    masks = mask_results[0]['masks']
    ids = mask_results[0]['class_ids']
    
    # This handles the case where only one mask is detected
    if len(np.shape(masks)) == 2:
        if ids[0] == 4: # This indicates that the model predicts the object is a motorcycle
            return (masks, True)
        else:
            return (masks, False)
    
    # This handles the case where no masks are detected
    if len(masks[0][0]) == 0:
        return (masks, False)
    
    # Initializes motorcycle_mask to be the first mask detected
    motorcycle_mask = masks[:,:,0]
    
    # This variable will change to True if a motorcycle is detected
    found_motorcycle = False
    
    # This will store the largest area of a motorcycle bounding box. 
    # Since the model may find multiple motorcycles, this program returns the motorcycle mask
    # with the largest bounding box. 
    motorcycle_area = 0.0
    
    for index in range(len(boxes)):
        if ids[index] == 4: # This indicates that the model predicts the object is a motorcycle
            box = boxes[index]

            y_min = box[0]
            x_min = box[1]
            y_max = box[2]
            x_max = box[3]
            
            area = (x_max - x_min)*(y_max - y_min)
            if area >= motorcycle_area:
                motorcycle_area = area
                motorcycle_mask = masks[:,:,index]
                found_motorcycle = True
                
    return (motorcycle_mask, found_motorcycle)


def mask_motorcycle(current_prefix):
    # Given the file name for an image of a motorcycle and a foreground mask, resize the image and 
    # then use Mask R-CNN to predict a probabilistic mask for the motorcycle. 
    
    image_folder = '../data/test/images/'
    current_image = current_prefix + 'image.jpg'
    image_path = image_folder + current_image
    
    # Load a motorcycle image
    image = io.imread(image_path)

    # Run detection. I've modified the method model.detect to take an additional argument is_bool. 
    # Is is_bool is set to True the model returns a boolean mask, else it returns a probabilistic mask. 
    results = model.detect([image], verbose=1, is_bool=False)
    
    (predicted_mask, found_motorcycle) = find_motorcycle_mask(results)
    
    if found_motorcycle:
        # Ensures the mask uses floats in [0,1]
        if np.amax(predicted_mask) > 0.0:
            predicted_mask / 255.0
        
        plt.imshow(predicted_mask)
        plt.show()
    else:
        print("No motorcycle detected.")
    
    return found_motorcycle

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir="../data/model_weights/", config=config)

# Local path to weights files
COCO_MODEL_PATH = "../data/model_weights/mask_rcnn_coco.h5"
FINE_TUNED_MODEL_PATH = "../data/model_weights/mask_rcnn_motorcycle_0003.h5"

# Load fine tuned weights
# Change FINE_TUNED_MODEL_PATH to COCO_MODEL_PATH to use coco weights instead
model.load_weights(FINE_TUNED_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# The following code creates a float mask for a motorcycle image. 
# Change this to another prefix in data/test/images/ to see a different example. 
prefix = 't_20_'
mask_motorcycle(prefix)
