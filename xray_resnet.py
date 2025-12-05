import torch.nn as nn
from torchvision import models


def get_xray_model(num_classes=1):
    """
    Returns a ResNet-18 model fine-tuned for binary classification.
    Output: one logit (will pass through sigmoid later).
    """
    # Load ResNet-18 pre-trained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace last fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
