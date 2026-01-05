# src/models/abdullah_model1.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbdullahSimpleCNN(nn.Module):
    """
    Abdullah - Model 1: Basit CNN (PyTorch)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=0)   # 32x32 -> 30x30
        self.pool  = nn.MaxPool2d(2, 2)               # 30x30 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, 3)             # 15x15 -> 13x13
        self.conv3 = nn.Conv2d(64, 64, 3)             # 13x13 -> 11x11

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x

def create_model():
    return AbdullahSimpleCNN(num_classes=10)
