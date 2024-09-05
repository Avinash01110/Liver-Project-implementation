Liver Tumor Segmentation using Deep Learning

This repository contains code for liver tumor segmentation using deep learning techniques. The segmentation models are implemented using the MONAI framework and FastAI, with architectures such as UNet, V-Net, AH-Net, and FastAI-based UNet.

Overview

The main files in this repository are Jupyter notebooks named after each architecture, such as unet\_monai.ipynb, vnet\_monai.ipynb, ahnet\_monai.ipynb, and fastai\_unet.ipynb. Each notebook contains the implementation of the liver tumor segmentation model with the respective architecture. These notebooks include dataset loading, model definition, training loop, evaluation, and inference functions.

Code Explanation

Imports

The code begins with importing necessary libraries:

monai, torch, numpy, albumentations, nibabel, matplotlib.pyplot: Libraries used for medical image analysis, deep learning tasks, numerical computations, data augmentation, and visualization.

fastai: Used for implementing UNet models for segmentation tasks.

Dataset Class

Each notebook contains a CustomSegmentationDataset class responsible for loading and preprocessing the liver tumor segmentation dataset. It handles loading CT images and corresponding tumor masks from NIfTI files, applying transformations, and preprocessing the data for training and evaluation.

Model Definition

The liver tumor segmentation models are implemented using different architectures such as MONAI’s UNet, V-Net, and AH-Net, along with FastAI’s UNet. These architectures are initialized with the appropriate number of classes (tumor and background) and input channels.

Training Loop

The train function implements the training loop for each architecture. It iterates over the training data, computes the loss (typically Dice Loss or Cross Entropy Loss), performs backpropagation, and updates model parameters using optimizers like Adam. The training process is repeated for multiple epochs to optimize model performance.

Evaluation

Each notebook includes evaluation metrics such as Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and pixel accuracy to assess the model's performance on the validation set. These metrics are calculated using a Metrics class that compares the predicted masks with the ground truth masks.

Inference

After training, the models can be used for inference on new CT images to generate tumor segmentation masks. The inference function takes a DataLoader containing test images as input and outputs the predicted masks using the trained model.

Results

The models implemented in this repository, including UNet, V-Net, AH-Net, and FastAI-based UNet, provide efficient segmentation of liver tumors. Each architecture demonstrates varying levels of accuracy and computational efficiency, making them suitable for different use cases depending on the available resources and desired precision.
