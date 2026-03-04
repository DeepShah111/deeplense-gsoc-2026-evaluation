import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetBaseline(nn.Module):
    """
    Custom 1-channel ResNet18 trained from scratch for baseline evaluation.
    """
    def __init__(self, num_classes=3):
        super(ResNetBaseline, self).__init__()
        self.model = models.resnet18(weights=None)
        # Modify for 1-channel (Grayscale) input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNetTransfer(nn.Module):
    """
    Pre-trained RGB ResNet18 utilizing ImageNet weights for transfer learning.
    """
    def __init__(self, num_classes=3):
        super(ResNetTransfer, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ViTChampion(nn.Module):
    """
    Vision Transformer (ViT-B/16) leveraging global attention mechanisms.
    """
    def __init__(self, num_classes=3):
        super(ViTChampion, self).__init__()
        self.model = models.vit_b_16(weights='IMAGENET1K_V1')
        # ViT classification head modification
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    


class DeepLenseEnsemble(nn.Module):
    """
    Fuses a ResNet (Texture Expert) and ViT (Context Expert) for robust predictions.
    Both models dynamically accept 224x224 high-resolution inputs.
    """
    def __init__(self, resnet_model, vit_model):
        super(DeepLenseEnsemble, self).__init__()
        self.resnet = resnet_model
        self.vit = vit_model
        
        # Freeze the base models to prevent retraining during inference
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Passes the 224x224 input to both experts and averages their probability distributions.
        """
        # ViT Path (Global Context)
        vit_outputs = self.vit(x)
        vit_probs = F.softmax(vit_outputs, dim=1)
        
        # ResNet Path (Local Textures)
        resnet_outputs = self.resnet(x)
        resnet_probs = F.softmax(resnet_outputs, dim=1)
        
        # Average Ensemble Fusion
        ensemble_probs = (vit_probs + resnet_probs) / 2.0
        return ensemble_probs