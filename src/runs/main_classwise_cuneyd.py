# src/main_classwise_cuneyd.py
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
import pandas as pd

from src.models.cuneyd_model1 import create_model as create_cuneyd_m1
from src.models.cuneyd_model2 import create_model as create_cuneyd_m2


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_classwise(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())

    return classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )


def run_classwise(model_name, model_fn, dataset, device, epochs=1):
    print(f"\n=== {model_name} | Class-wise ===")
    y = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    reports = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"Fold {fold}/5")

        train_loader = DataLoader(Subset(dataset, tr), batch_size=16, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(dataset, va), batch_size=16, shuffle=False, num_workers=0)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        reports.append(eval_classwise(model, val_loader, device))

    # fold ortalaması
    metrics = {}
    for cls in CLASS_NAMES:
        metrics[cls] = {
            "precision": float(np.mean([r[cls]["precision"] for r in reports])),
            "recall":    float(np.mean([r[cls]["recall"] for r in reports])),
            "f1-score":  float(np.mean([r[cls]["f1-score"] for r in reports])),
        }

    df = pd.DataFrame(metrics).T
    csv_name = f"classwise_{model_name}.csv"
    df.to_csv(csv_name, index=True)
    print(df.head())
    print(f"\n✅ CSV kaydedildi: {csv_name}")
    return df


def main():
    print("=== DEVAI (PyTorch) | Cüneyd Class-wise ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = get_dataset()

    run_classwise("Cuneyd_Model1_ResidualCNN", create_cuneyd_m1, ds, device, epochs=1)
    run_classwise("Cuneyd_Model2_HybridCNNLSTM", create_cuneyd_m2, ds, device, epochs=1)


if __name__ == "__main__":
    main()
