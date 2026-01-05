# src/models/cuneyd_model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CuneydHybridCNNLSTM(nn.Module):
    """
    Cüneyd - Model 2: Hybrid CNN-LSTM
    CNN feature map -> sequence (H*W) -> LSTM -> classifier
    """
    def __init__(self, num_classes=10, lstm_hidden=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # 16x16

            nn.Conv2d(32, 64, 3, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # 8x8

            nn.Conv2d(64, 128, 3, padding=1),# 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # sequence length = 8*8 = 64, feature dim = 128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = self.features(x)                 # (B, 128, 8, 8)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, 128)
        x = x.view(b, h*w, c)                   # (B, 64, 128)

        out, _ = self.lstm(x)                # (B, 64, hidden)
        last = out[:, -1, :]                 # (B, hidden)

        last = self.dropout(last)
        logits = self.fc(last)
        return logits


def create_model():
    return CuneydHybridCNNLSTM(num_classes=10, lstm_hidden=128)
