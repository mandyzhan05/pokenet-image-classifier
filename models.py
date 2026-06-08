# Imports
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes = 1000):
    '''
    Phase 1: Load pre-trained ResNet50 and adapt for Pokemon classification

    Args:
    num_classes (int): Number of Pokémon classes to output
    '''
    # Load a pre-trained ResNet50 
    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    
    # Freeze all pre-trained layers
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    # Unfreeze the final FC layer for fine-tuning, outputting num_classes
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model

def unfreeze_model(model):
    '''
    Phase 2: Transition to full fine-tuning

    Args:
    model: ResNet50 model returned by get_model()
    '''
    # Unfreeze all layers for fine-tuning
    for parameter in model.parameters():
        parameter.requires_grad = True
        
    return model

def get_device():
    '''
    Returns the available device (GPU if available, otherwise CPU)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return device