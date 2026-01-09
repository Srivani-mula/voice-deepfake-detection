import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import ASVspoofDataset
from model import CNNClassifier


# =====================================================
# MAIN FUNCTION (IMPORTANT FOR WINDOWS)
# =====================================================
def main():

    # ------------------------
    # Device
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # Datasets
    # ------------------------
    train_dataset = ASVspoofDataset(split="train")
    val_dataset = ASVspoofDataset(split="dev")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # IMPORTANT for Windows
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # ------------------------
    # Class Weights (IMBALANCE FIX)
    # ------------------------
    train_labels = [label.item() for _, label in train_dataset]
    class_counts = np.bincount(train_labels)

    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights:", class_weights)

    # ------------------------
    # Model
    # ------------------------
    model = CNNClassifier().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------
    # Training Loop
    # ------------------------
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Val F1: {f1:.4f}"
        )

    # ------------------------
    # Confusion Matrix (FINAL)
    # ------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Spoof", "Bonafide"],
        yticklabels=["Spoof", "Bonafide"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – ASVspoof LA (Dev)")
    plt.tight_layout()
    plt.show()

    # ------------------------
    # Save Model
    # ------------------------
    torch.save(model.state_dict(), "cnn_asvspoof.pth")
    print("Model saved as cnn_asvspoof.pth")


# =====================================================
# ENTRY POINT (REQUIRED ON WINDOWS)
# =====================================================
if __name__ == "__main__":
    main()
