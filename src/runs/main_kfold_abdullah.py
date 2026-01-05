import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torchvision import datasets, transforms

from src.models.abdullah_model1 import create_model as create_abdullah_m1
from src.models.abdullah_model2 import create_model as create_abdullah_m2


def get_cifar10_train_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return ds


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
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, p, r, f1


def run_stratified_kfold(model_name, model_fn, dataset, device, n_splits=5, epochs=2, batch_size=64):
    print(f"\n=== {model_name} | StratifiedKFold={n_splits} ===")

    y = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=0)

        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            acc, p, r, f1 = evaluate(model, val_loader, device)
            print(f"Fold {fold} Epoch {ep}/{epochs} | loss={loss:.4f} | acc={acc:.4f} | F1={f1:.4f}")

        fold_metrics.append([acc, p, r, f1])

    fold_metrics = np.array(fold_metrics)
    mean = fold_metrics.mean(axis=0)
    std = fold_metrics.std(axis=0)

    print(f"\n>>> {model_name} | K-Fold mean ± std")
    print(f"Accuracy : {mean[0]:.4f} ± {std[0]:.4f}")
    print(f"Precision: {mean[1]:.4f} ± {std[1]:.4f}")
    print(f"Recall   : {mean[2]:.4f} ± {std[2]:.4f}")
    print(f"F1-macro : {mean[3]:.4f} ± {std[3]:.4f}")

    return mean, std


def main():
    print("=== DEVAI (PyTorch) | Abdullah K-Fold ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = get_cifar10_train_dataset()

    # CPU hızlı olsun diye epochs=2 öneriyorum
    run_stratified_kfold("Abdullah - Model1 (SimpleCNN)", create_abdullah_m1, ds, device, n_splits=5, epochs=2)
    run_stratified_kfold("Abdullah - Model2 (AdvancedCNN)", create_abdullah_m2, ds, device, n_splits=5, epochs=2)


if __name__ == "__main__":
    main()
