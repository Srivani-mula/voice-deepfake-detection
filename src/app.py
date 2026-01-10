import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from model import CNNClassifier
from features import extract_logmel

# PAGE CONFIG
st.set_page_config(
    page_title="Voice Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

# CUSTOM CSS
st.markdown(
    """
<style>
body {
    background-color: #f5f7fb;
}
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #0B1C2D !important;
    font-weight: 700;
}
.result-box {
    padding: 1.5rem;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
.real {
    background-color: #e6f4ea;
    color: #1e7f43;
}
.fake {
    background-color: #fdecea;
    color: #b71c1c;
}
</style>
""",
    unsafe_allow_html=True,
)

# LOAD MODEL
@st.cache_resource
def load_model():
    model = CNNClassifier()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cnn_asvspoof.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()


# PREDICTION FUNCTION
def predict_audio(wav_path):
    features = extract_logmel(wav_path)  # (64, 100)
    features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    label = "Bonafide (Real)" if np.argmax(probs) == 1 else "Spoof (Fake)"
    confidence = float(np.max(probs)) * 100
    return label, confidence


# UI
st.title("🎙️ Voice Deepfake Detection")
st.caption("Upload an audio file to check whether it is Real (Bonafide) or Fake (Spoof).")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "aac", "m4a"],
)

# HANDLE UPLOAD
if uploaded_file is not None:
    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load audio
        audio, sr = librosa.load(tmp_path, sr=16000)

        # Extract features
        features = extract_logmel(audio, sr)
        features = torch.tensor(features).unsqueeze(0)
        features = torch.tensor(features).float()
f       features = features.unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(features)
            pred = torch.argmax(output, dim=1).item()

        label = "Real (Bonafide)" if pred == 0 else "Fake (Spoof)"
        st.success(f"Prediction: {label}")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
