# src/models/kasim_model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Block
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)   # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class KasimAttentionCNN(nn.Module):
    """
    Kasım - Model 2: Attention CNN (SE-Block) - CIFAR10
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=16)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=16)

        self.pool = nn.MaxPool2d(2, 2)  # 32->16->8
        self.drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 16x16

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool(x)  # 8x8

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def create_model():
    return KasimAttentionCNN(num_classes=10)
