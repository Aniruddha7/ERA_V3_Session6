import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytest
from MNIST import Net
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_architecture():
    model = Net()  # Only create the model, don't run any forward passes
    
    # Test number of parameters
    num_params = count_parameters(model)
    assert num_params < 20000, f"Model has {num_params} parameters, should be less than 20000"
    logging.info(f"Model parameter count test passed: {num_params:,} parameters")

    # Check if Batch Normalization is used
    batchnorm_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    assert len(batchnorm_layers) > 0, "Batch normalization is not used in the model"

    # Check if Dropout is used
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert len(dropout_layers) > 0, "Dropout is not used in the model"

    # Check if Fully Connected layer is used
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(fc_layers) > 0, "Fully connected layer is not used in the model"

checkpoint_path = 'mnist_best_model_20241221_012216.pth'

def test_check_number_of_epochs():
    checkpoint = torch.load(checkpoint_path) 
    num_epochs = checkpoint.get('epoch', None)
    assert num_epochs is not None, "Number of epochs not found in the checkpoint file"
    assert num_epochs <= 20, f"Number of epochs {num_epochs} exceeds the limit of 20"

