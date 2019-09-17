# Image-segmentation
The goal of this project is to create a model capable of segmenting an image with known bounding boxes, optimizing for high accuracy (measured as IoU).

# Proposed approach: 
## A neural network for each image
Suppose we have an image, like this clownfish, and wish to identify the foreground pixels: 

![Clownfish](https://i.imgur.com/ZDNLZAi.png)

If we have a random assortment of pixels labelled as foreground / background / boundary, then we can train a relatively small neural network to classify the rest of the pixels as foreground / background / boundary based only on local information. This neural network would take a small square of pixels as input, and use this to classify the pixel in the center. 

![Small neural network](https://i.imgur.com/hI258Cf.png)

This neural network is not especially helpful on its own, however. It's tailored to a specific image and doesn't generalize to other images. 

## A neural network that creates other neural networks
To create a model capable of processing images it's never seen before, I propose the following approach: Train a convolutional neural network which takes an image as output, and produces a small neural network - tailored to that image - as output. 

![Large neural network](https://i.imgur.com/CqWhMaA.png)

I would then use the smaller neural network tailored to a specific image to classify pixels as foreground, background, or boundary. 
