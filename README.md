# Image-segmentation
The goal of this project is to create a model capable of segmenting an image with known bounding boxes, optimizing for high accuracy (measured as IoU).

# State of the art - Mask R-CNN
Mask R-CNN(https://github.com/matterport/Mask_RCNN) is a state of the art model which creates segmentation masks for a variety of object classes, even in scenes with many objects. However, these masks are not always precise. For example, Mask R-CNN correctly identifies this as an image of a motorcycle:

![Motorcycle](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/t_125_image.jpg?raw=true)

But its segmentation mask, while approximately correct, would benefit from improved precision: 

![Motorcycle mask](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/t_125_.jpg?raw=true)

# Proposed approach: 
## Step 1: Fine tuning Mask R-CNN


## Step 2: Training a U-Net to refine small regions of proposed segmentation masks

