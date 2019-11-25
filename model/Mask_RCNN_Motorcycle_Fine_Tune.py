#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program fine-tunes Mask R-CNN on motorcycle images. 

This code was based on the following program: 
    https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py

Here is the copyright information for the original program:

Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
"""

import os
import numpy as np
import imageio

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

# Path to COCO trained weights file
COCO_MODEL_PATH = "../data/model_weights/mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = "../data/model_weights/fine_tuned_model"

############################################################
#  Configurations
############################################################


class NewMotorcycleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "motorcycle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class NewMotorcycleDataset(utils.Dataset):

    def load_motorcycle(self, dataset_dir, subset):
        """Load a subset of the NewMotorcycle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        subset is the name of a subfolder. 
        """

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

        # Add all COCO classes. 
        for index in range(1,len(class_names)):
            self.add_class("coco", index, class_names[index])

        # Train or validation dataset?
        print(subset)
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        image_list = os.listdir(dataset_dir + "/images")
        
        for image_name in image_list:
            image_path = dataset_dir + "/images/" + image_name
            character = image_name.find('i')
            prefix = image_name[:character]
            image = imageio.imread(image_path)
            im_height = len(image)
            im_width = len(image[0])
            self.add_image(
                "coco",
                image_id = subset + "_" + prefix,
                path = image_path, height = im_height,
                width = im_width)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        image_id is the prefix, with the string "train_" or "val_" at the beginning. 
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Determines whether the mask is in the training or validation set
        name = self.image_info[image_id]["id"]
        if name[0] == "t":
            mask_dir = "../data/train/masks/"
            prefix = name[6:]
        if name[0] == "v":
            mask_dir = "../data/val/masks/"
            prefix = name[4:]
        
        mask_file_name = prefix + "mask.png"
        print(mask_file_name)
        mask_path = mask_dir + mask_file_name
        mask = imageio.imread(mask_path)
        if len(np.shape(mask)) == 2:
            mask = np.expand_dims(mask, axis = 2)

        # Return mask, and array of class IDs of each instance. Since each image 
        # has only one mask which is a motorcycle, we return the array [4]. 
        return mask.astype(np.bool), np.array([4])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = NewMotorcycleDataset()
    dataset_train.load_motorcycle("../data", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NewMotorcycleDataset()
    dataset_val.load_motorcycle("../data", "val")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='heads')

############################################################
#  Training
############################################################
    
# Directory to save logs and trained model
config = NewMotorcycleConfig()

model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

train(model)
