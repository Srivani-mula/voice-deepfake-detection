import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ASVspoofDataset
from model import CNN
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import CNNClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_asvspoof.pth", map_location=device))
model.eval()

# Load validation data
val_dataset = ASVspoofDataset(split="dev")
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Bonafide", "Spoof"]))
