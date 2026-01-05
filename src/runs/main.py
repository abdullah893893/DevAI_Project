# src/main.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import load_cifar10
from src.models.abdullah_model1 import create_model as create_abdullah_m1
from src.models.abdullah_model2 import create_model as create_abdullah_m2


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
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
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, p, r, f1


def run_experiment(model_name, model_fn, train_loader, test_loader, device, epochs=5, lr=1e-3):
    print(f"\n--- {model_name} ---")

    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, p, r, f1 = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss={loss:.4f} | "
            f"acc={acc:.4f} | "
            f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
        )

    return {
        "Model": model_name,
        "Acc": acc,
        "P": p,
        "R": r,
        "F1": f1
    }


def main():
    print("=== DEVAI (PyTorch) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Windows için en stabil ayar
    train_loader, test_loader = load_cifar10(batch_size=64, num_workers=0)

    results = []

    # Abdullah Model 1
    results.append(
        run_experiment(
            "Abdullah - Model1 (SimpleCNN)",
            create_abdullah_m1,
            train_loader,
            test_loader,
            device,
            epochs=5
        )
    )

    # Abdullah Model 2
    results.append(
        run_experiment(
            "Abdullah - Model2 (AdvancedCNN)",
            create_abdullah_m2,
            train_loader,
            test_loader,
            device,
            epochs=5
        )
    )

    print("\n=== ÖZET (Abdullah) ===")
    for r in results:
        print(
            f"{r['Model']} | "
            f"Acc={r['Acc']:.4f} | "
            f"P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f}"
        )


if __name__ == "__main__":
    main()
