import torch.nn as nn
from torchvision import models

def build_resnet_baseline():
    #Initializes a custom 1-channel ResNet18 trained from scratch.
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model

def build_resnet_transfer():
    #Initializes a pre-trained RGB ResNet18 for transfer learning.
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model

def build_vit_champion():
    #Initializes the Vision Transformer (ViT-B/16) for global context.
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads.head = nn.Linear(model.heads.head.in_features, 3)
    return model