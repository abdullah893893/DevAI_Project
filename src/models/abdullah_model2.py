# src/models/abdullah_model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbdullahAdvancedCNN(nn.Module):
    """
    Abdullah - Model 2: Gelişmiş CNN (BatchNorm + Dropout) (PyTorch)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)   # 32x32
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)  # 32x32
        self.b2 = nn.BatchNorm2d(32)
        self.p1 = nn.MaxPool2d(2, 2)               # 16x16
        self.d1 = nn.Dropout(0.25)

        # Block 2
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)  # 16x16
        self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)  # 16x16
        self.b4 = nn.BatchNorm2d(64)
        self.p2 = nn.MaxPool2d(2, 2)               # 8x8
        self.d2 = nn.Dropout(0.25)

        # Block 3
        self.c5 = nn.Conv2d(64, 128, 3, padding=1) # 8x8
        self.b5 = nn.BatchNorm2d(128)
        self.d3 = nn.Dropout(0.25)

        # FC
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.b6  = nn.BatchNorm1d(256)
        self.d4  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = self.p1(x)
        x = self.d1(x)

        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        x = self.p2(x)
        x = self.d2(x)

        x = F.relu(self.b5(self.c5(x)))
        x = self.d3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.b6(self.fc1(x)))
        x = self.d4(x)
        x = self.fc2(x)  # logits
        return x

def create_model():
    return AbdullahAdvancedCNN(num_classes=10)
