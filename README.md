# Image segmentation: Regional U-Nets for improving precision
This project uses u-nets to improve the precision of an existing instance segmentation model. 

# State of the art - Mask R-CNN
[Mask R-CNN](https://github.com/matterport/Mask_RCNN) is a state of the art model which creates segmentation masks for a variety of object classes, even in scenes with many objects. However, these masks are not always precise. For example, Mask R-CNN correctly identifies this as an image of a motorcycle, but its segmentation mask, while approximately correct, would benefit from improved precision: 

![Motorcycle mask](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/First%20approximation.png?raw=true)

# Approach: 
## Step 1: Fine tuning Mask R-CNN
The first step to improving Mask R-CNN's precision was to use transfer learning, beginning with its COCO trained weights, but subsequently training it on a small but carefully annotated set of images. I used images of motorcycles from [Open Images](https://opensource.google/projects/open-images-dataset) datasat. Howeer, I only used images from its cross validation and test sets, because their masks are hand drawn and more accurate than those in the training set. 

This fine tuning resulted in an increase of mean IoU from 0.679 to 0.712 on my test set of 57 motorcycle images. (I began with a test set of 70 images, but Mask R-CNN only identified 57 of them as containing motorcycles, before fine tuning. I used this set of 57 throughout for consistency.)

![Fine tuning](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/Progression%20fine%20tuned.png?raw=true)

## Step 2: Training a U-Net to refine small regions of proposed segmentation masks
Once Mask R-CNN has been fine tuned for precise segmentation for a particular image class, the next step is to train a u-net for that class. This u-net takes a 64x64 region of the original image as input, along with the same small region of the predicted segmentation mask, and produces a new prediction for the segmentation mask in that region. 

![Regional U-Net](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/Regional%20u-net.png?raw=true)

To create this u-net I used [this](https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py) open source implementation. The architecture is similar to that of ResNet 18, but with fewer channels, in my case: 

![U-Net architecture](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/U-net%20diagram.png?raw=true)

After training the u-net, I refined each proposed mask by considering all 64x64 squares centered on the boundary of the mask, feeding them all to the u-net, and taking a weighted average of their predictions along with the predictions of the mask itself. 

# Results
The approach described above further improved the mean IoU on my test set, from 0.712 to 0.732. Here is a nice example of this increased precision: 

![Post u-net](https://github.com/AlternatingSum/Image-segmentation/blob/master/static/After%20u-net.png?raw=true)
