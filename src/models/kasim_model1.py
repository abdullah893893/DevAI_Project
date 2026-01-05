# src/models/kasim_model1.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class KasimResNet18Transfer(nn.Module):
    """
    Kasım - Model 1: Transfer Learning (ResNet18 pretrained) - CIFAR10 uyumlu
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
        else:
            self.model = resnet18(weights=None)

        # CIFAR-10 uyarlaması
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # Son katman
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in self.model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x):
        return self.model(x)


def create_model():
    return KasimResNet18Transfer(
        num_classes=10,
        pretrained=True,
        freeze_backbone=True
    )
