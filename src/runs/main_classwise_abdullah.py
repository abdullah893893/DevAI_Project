import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
import pandas as pd

from src.models.abdullah_model1 import create_model as create_abdullah_m1
from src.models.abdullah_model2 import create_model as create_abdullah_m2


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
def evaluate_classwise(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    return report


def run_classwise(model_name, model_fn, dataset, device, epochs=2):
    print(f"\n=== {model_name} | Class-wise Evaluation ===")

    y = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"Fold {fold}/5")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True, num_workers=0)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=64, shuffle=False, num_workers=0)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        report = evaluate_classwise(model, val_loader, device)
        reports.append(report)

    # Fold ortalaması
    class_metrics = {}
    for cls in CLASS_NAMES:
        class_metrics[cls] = {
            "precision": np.mean([r[cls]["precision"] for r in reports]),
            "recall":    np.mean([r[cls]["recall"] for r in reports]),
            "f1-score":  np.mean([r[cls]["f1-score"] for r in reports]),
        }

    df = pd.DataFrame(class_metrics).T
    df.to_csv(f"classwise_{model_name.replace(' ', '_')}.csv")
    print(df)

    return df


def main():
    print("=== DEVAI (PyTorch) | Class-wise Abdullah ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = get_dataset()

    run_classwise("Abdullah_Model1_SimpleCNN", create_abdullah_m1, dataset, device)
    run_classwise("Abdullah_Model2_AdvancedCNN", create_abdullah_m2, dataset, device)


if __name__ == "__main__":
    main()
