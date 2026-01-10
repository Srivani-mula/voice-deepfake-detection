import streamlit as st
import torch
import numpy as np
import librosa
import tempfile
import os

from model import CNNClassifier
from features import extract_logmel

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Voice Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Voice Deepfake Detection")
st.write("Upload an audio file to check whether it is **Real (Bonafide)** or **Fake (Spoof)**")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    model = CNNClassifier()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "cnn_asvspoof.pth")

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file cnn_asvspoof.pth not found")
        st.stop()

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "ogg", "aac", "m4a"]
)

# ----------------------------------
# PROCESS AUDIO
# ----------------------------------
if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load audio
        audio, sr = librosa.load(tmp_path, sr=16000)

        # Feature extraction (IMPORTANT)
        features = extract_logmel(audio, sr, max_len=100)  # (T, F)

        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)

        # Add batch & channel dimensions -> (1, 1, T, F)
        features = features.unsqueeze(0).unsqueeze(0)
        st.write("Feature shape:", features.shape)

        # Model inference
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

        # Label mapping (MATCH TRAINING)
        label = "Fake (Spoof)" if pred == 0 else "Real (Bonafide)"

        # Display result
        if pred == 0:
            st.error(f"🚨 {label} — Confidence: {confidence:.2f}%")
        else:
            st.success(f"✅ {label} — Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        # Cleanup
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
