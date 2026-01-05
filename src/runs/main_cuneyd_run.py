# src/main_cuneyd_run.py
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import load_cifar10
from src.models.cuneyd_model1 import create_model as create_cuneyd_m1
from src.models.cuneyd_model2 import create_model as create_cuneyd_m2


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, p, r, f1


def run_experiment(name, model_fn, train_loader, test_loader, device, epochs=3, lr=1e-3):
    print(f"\n--- {name} ---")
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, p, r, f1 = evaluate(model, test_loader, device)
        print(f"Epoch {ep}/{epochs} | loss={loss:.4f} | acc={acc:.4f} | P={p:.4f} R={r:.4f} F1={f1:.4f}")


def main():
    print("=== DEVAI (PyTorch) | Cüneyd ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, test_loader = load_cifar10(batch_size=32, num_workers=0)

    run_experiment("Cüneyd - Model1 (Residual CNN)", create_cuneyd_m1, train_loader, test_loader, device, epochs=3)
    run_experiment("Cüneyd - Model2 (Hybrid CNN-LSTM)", create_cuneyd_m2, train_loader, test_loader, device, epochs=3)


if __name__ == "__main__":
    main()
