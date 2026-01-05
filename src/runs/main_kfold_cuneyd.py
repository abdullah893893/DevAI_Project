# src/main_kfold_cuneyd.py
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import datasets, transforms

from src.models.cuneyd_model1 import create_model as create_cuneyd_m1
from src.models.cuneyd_model2 import create_model as create_cuneyd_m2


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
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, p, r, f1


def run_kfold(name, model_fn, dataset, device, epochs=1):
    print(f"\n=== {name} | StratifiedKFold ===")
    y = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n--- Fold {fold}/5 ---")
        train_loader = DataLoader(Subset(dataset, tr), batch_size=16, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(dataset, va), batch_size=16, shuffle=False, num_workers=0)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        acc, p, r, f1 = evaluate(model, val_loader, device)
        print(f"Fold {fold} | Acc={acc:.4f} | F1={f1:.4f}")
        results.append([acc, p, r, f1])

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)

    print(f"\n>>> {name} | mean ± std")
    print(f"Acc : {mean[0]:.4f} ± {std[0]:.4f}")
    print(f"P   : {mean[1]:.4f} ± {std[1]:.4f}")
    print(f"R   : {mean[2]:.4f} ± {std[2]:.4f}")
    print(f"F1  : {mean[3]:.4f} ± {std[3]:.4f}")


def main():
    print("=== DEVAI (PyTorch) | Cüneyd K-Fold ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = get_dataset()
    run_kfold("Cüneyd - Model1 (Residual CNN)", create_cuneyd_m1, ds, device, epochs=1)
    run_kfold("Cüneyd - Model2 (Hybrid CNN-LSTM)", create_cuneyd_m2, ds, device, epochs=1)


if __name__ == "__main__":
    main()
