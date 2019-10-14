# Image-segmentation
The goal of this project is to create a model capable of segmenting an image with known bounding boxes, optimizing for high accuracy (measured as IoU).

# State of the art - Mask R-CNN
[Mask R-CNN](https://github.com/matterport/Mask_RCNN) is a state of the art model which creates segmentation masks for a variety of object classes, even in scenes with many objects. However, these masks are not always precise. For example, Mask R-CNN correctly identifies this as an image of a motorcycle, but its segmentation mask, while approximately correct, would benefit from improved precision: 

![Motorcycle mask](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/First%20approximation.png?raw=true)

# Proposed approach: 
## Step 1: Fine tuning Mask R-CNN
The first step to improving Mask R-CNN's precision is to use transfer learning, beginning with its COCO trained weights, but subsequently training it on a small but carefully annotated set of images. 

## Step 2: Training a U-Net to refine small regions of proposed segmentation masks
Once Mask R-CNN has been fine tuned for precise segmentation for a particular image class, the next step is to train a u-net for that class. This u-net takes a small region of the original image as input, along with the same small region of the predicted segmentation mask, and produces a new prediction for the segmentation mask in that region. 
