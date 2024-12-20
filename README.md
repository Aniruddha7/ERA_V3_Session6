# MNIST Classification Model

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. It includes two primary components:

1. **MNIST Model (`MNIST.py`)**: A CNN for classifying MNIST digits with Batch Normalization, Dropout, and Fully Connected layers.
2. **Test Suite (`test_model.py`)**: A testing script to validate the model architecture and check its hyperparameters.

## Table of Contents

- [Model Overview](#model-overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Resulting Model](#resulting-model)
- [File Structure](#file-structure)

## Model Overview

The model consists of several convolutional layers designed to classify MNIST digits. The key features of the architecture include:

- **Convolutional Layers**: Several layers of convolutional operations with Batch Normalization and ReLU activation functions.
- **Pooling Layers**: MaxPooling layers to reduce spatial dimensions.
- **Dropout Layers**: Dropout is used to prevent overfitting.
- **Fully Connected Layer**: After convolutional layers, the output is flattened and passed through a fully connected layer to output class probabilities.

### Model Components

- **Layer 1**: Conv -> ReLU -> BatchNorm -> MaxPool -> 1x1 Conv
- **Layer 2**: Conv -> ReLU -> BatchNorm -> MaxPool -> 1x1 Conv
- **Layer 3**: Conv -> ReLU -> BatchNorm
- **Layer 4**: Conv -> ReLU -> BatchNorm
- **Layer 5**: 1x1 Conv to reduce channels before FC layer
- **FC Layer**: Fully connected layer to output 10 classes.

## Requirements

The following packages are required to run this project:

- `torch`
- `torchvision`
- `pytest`
- `tqdm`
- `torchsummary`
- `logging`

You can install them by running:

```bash
pip install -r requirements.txt
```


## Training the Model
The model is trained for 17 epochs using the MNIST dataset, which is automatically downloaded and preprocessed. You can run the training by executing the following:
```
python MNIST.py
```

This will:

Train the model for 17 epochs (adjustable).
Apply data augmentations like random rotation and affine transformations.
Save the best model based on validation accuracy.
The saved model will be named ```mnist_best_model_<timestamp>.pth.```

### Training logs with validation accuracy of 99.48% at 14th epoch
![](https://github.com/Aniruddha7/ERA_V3_Session6/blob/main/train_logs.jpg)

## Testing the Model
To validate the model's architecture and hyperparameters, use the ```test_model.py script```. This script performs the following tests:

Test 1: Verifies that the model has fewer than 20,000 parameters.
Test 2: Checks if Batch Normalization, Dropout, and Fully Connected layers are present in the model.
Test 3: Verifies that the model was trained with fewer than 20 epochs.
Run the test suite with:

```
    pytest test_model.py
```
The tests will log the results, including the number of parameters, the presence of Batch Normalization, Dropout, and FC layers, and the number of training epochs.

## Resulting Model
Once training is completed, the best model (in terms of accuracy) will be saved with the following details:

Model state dictionary (model_state_dict).
Test accuracy achieved on the validation dataset.
Epoch number where the best accuracy was achieved.


