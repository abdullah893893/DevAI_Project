# src/main_kasim_run.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import load_cifar10
from src.models.kasim_model1 import create_model as create_kasim_m1
from src.models.kasim_model2 import create_model as create_kasim_m2


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
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
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

    return acc, p, r, f1


def main():
    print("=== DEVAI (PyTorch) | Kasım ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, test_loader = load_cifar10(batch_size=64, num_workers=0)

    run_experiment("Kasım - Model1 (ResNet18 Transfer)", create_kasim_m1, train_loader, test_loader, device, epochs=3, lr=1e-3)
    run_experiment("Kasım - Model2 (SE-Attention CNN)", create_kasim_m2, train_loader, test_loader, device, epochs=3, lr=1e-3)


if __name__ == "__main__":
    main()
