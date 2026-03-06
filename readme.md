# CNN Image Classification Project

## Overview

This project implements an image classification system using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

The model is trained to classify images into **7 categories**:

* Bike
* Cars
* Cats
* Dogs
* Flowers
* Horses
* Human

## Features

* Image preprocessing and normalization
* Data augmentation for better generalization
* CNN architecture with multiple convolution layers
* Batch normalization and dropout to prevent overfitting
* Training with validation dataset
* Model evaluation using accuracy and loss metrics

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

## Project Structure

```
CNN/
│
├── data/
│   └── data_set/
│       ├── bike/
│       ├── cars/
│       ├── cats/
│       ├── dogs/
│       ├── flowers/
│       ├── horses/
│       └── human/
│
├── main.py
├── image_classifier.h5
└── README.md
```

## Model Architecture

The CNN model consists of:

* Convolution layers for feature extraction
* MaxPooling layers for dimensionality reduction
* BatchNormalization for stable training
* Dropout layers to reduce overfitting
* Fully connected Dense layers
* Softmax output layer for multi-class classification

## Training

The model is trained with:

* Image augmentation
* Adam optimizer
* Categorical crossentropy loss
* 30 training epochs

## Results

The model outputs:

* Test Accuracy
* Test Loss

## Run the Project

Install dependencies:

```bash
pip install tensorflow opencv-python numpy
```

Run the training script:

```bash
python main.py
```

## Author

**Ashik Shyjin**
