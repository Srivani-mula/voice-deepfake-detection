import torch
import numpy as np
from features import extract_logmel

def predict_audio(model, audio_path):
    # Extract features
    features = extract_logmel(audio_path)

    # Shape: (1, 1, time, mel)
    features = torch.tensor(features).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    label = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(real_prob, fake_prob) * 100

    return label, confidence, probs
