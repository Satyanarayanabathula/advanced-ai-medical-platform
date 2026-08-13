import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 2


def create_model():
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model