import torch
from model import CNNClassifier
from model import CNNClassifier

model = CNNClassifier()
model.load_state_dict(torch.load("cnn_asvspoof.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 64, 100)
output = model(dummy_input)

print("Model output:", output.item())
