# Image Segmentation using U-Net
The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf).

## Overview

Face segmentation represents an active area of research within the biometric community in particular and the computer vision community in general. Over the last two decades, methods for face segmentation have received increasing attention due to their diverse applications in several human-face image analysis tasks. Although many algorithms have been developed to address the problem, face segmentation is still a challenge not being completely solved, particularly for
images taken in wild, unconstrained conditions.

![eyes_lips_segmentation.png](https://preview.redd.it/1f58zpq0afs41.png?width=694&format=png&auto=webp&s=e36b0ba7557de7700740056fa4c6a540db222c6a)

In this project, I created a segmentation model capable of segmenting eyes and lips from the provided facial image. The system utilises U-Net Architecture trained on 8,000 images from CelebAMask Dataset.

The goal is to create an intelligent system to segment eyes and lips by training a segmentation model on the CelebAMask Dataset.

### Data

The original model was trained on dataset from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), but I used this architecture to do the lips and eyes segmentation from [CelebAMask Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Evaluation Metric

Pixelwise Accuracy is a common metric used for these kinds of problems and is easiest to understand conceptually. It is the percentage of pixels in your image that are classified correctly. While it is easy to understand, it is not the best metric for segmentation problems. The maximum area of the ground truth(about 95%) is covered with black pixels. So if the model classifies all pixels as that class, 95% of pixels are classified accurately while the other 5% are not. As a result,
although the accuracy is 95%, the model is returning a completely useless prediction.

![iou.png](https://pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)

Here we are using Intersection over Union (Jaccard index) to evaluate our model accuracy. It is a standard and reliable metric for segmentation tasks. It is defined as the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth.This metric ranges from 0–1 (0–100%) with 0 signifying no overlap and 1 signifying perfectly overlapping segmentation. For multi-class segmentation, the mean IoU of the image is calculated by taking the IoU of each class and averaging them.

## Methodology

The entire task was segregated in the following stages:
1. Pre- Processing
2. Model Building
3. Training
4. Accuracy Evaluation
5. Fine-Tuning
6. Final Evaluation on Test Images

## Preprocessing

The preprocessing done in the notebook consists of the following steps:
1. The list of images is randomized
2. The images were divided into a training set and a validation set with following distribution:  
    a. **Training Set - 8100 instances**  
    b. **Testing Set - 1000 instances**  
    c. **Validation Set - 9000 instances**  
3. As we are using the [sigmoid](https://keras.io/api/layers/activations/#sigmoid-function) in our final layer, the images and mask were normalised with [min-max normalisation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) approach.
4. The mask for upper lip, lower lip, left eye and right eye were fused and converted to a single channel, to be interpreted as grayscale.
5. All the training images were resized to (128,128,3) and masks to (128,128,1). As the training resources were limited, this step was undertaken to decrease the number of parameters in the model.
6. The input images and masks were stored as a numpy array in batch to avoid the preprocessing steps for each run.
7. The preprocessing steps were common for the entire dataset so a parallelised approach was used. As the task was compute heavy [ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html) was utilized.

## Model Architecture

![unet.png](https://miro.medium.com/max/1838/1*f7YOaE4TWubwaFF7Z1fzNw.png)

## Model Building and Training

The model was trained on the preprocessed training data. This was done in a Colab notebook and can be further divided into the following steps:

1. Load the training and validation images into memory, preprocess them as described in the previous section and dump them in numpy array after creating the batches.
2. Define the network architecture and training parameters
3. Define the loss function(Jaccard Loss here), accuracy(Jackard Coeff here)
4. Train the network, logging the validation/training loss and the training/validation IOU.
5. Plot the logged values using tensorboard
6. Create model checkpoints and If the accuracy is not high enough, return to step 4.
7. Save and freeze the trained network

## Accuracy Evaluation

During development, a validation set was used to evaluate the model.

The training was done for 80 epochs and following were the IOU scores at each checkpoints:

| Epoch  | IOU |
| ------------- | ------------- |
| 20  | 0.201  |
| 40  | 0.285 |
| 50  | 0.725 |
| 70  | 0.741 |

## Improvement

1. Currently the model was trained on 8100 images due to the lack of resources to handle 30,000 images. For improvement the model can be built on a bigger machine to train on all the images for higher epochs.
2. Currently the model was trained on keras which inhibited the usage of generators to create a data loader. The code can be moved to stack tensorflow or pytorch to implement generators which will decrease the computational resource requirement and will optimize
the preprocessing time..
3. Keras batch training doesn’t support implementation of callbacks such as weight decay, early stopping etc. Moving the training pipeline to pytorch or tensorflow stack will be helpful to fine tune more hyperparameters.
