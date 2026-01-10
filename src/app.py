import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

from model import CNNClassifier
from features import extract_logmel

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Voice Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
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
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = CNNClassifier()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cnn_asvspoof.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# SAFE AUDIO LOADER (CRITICAL)
# -----------------------------
def load_audio_safe(path, target_sr=16000):
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, sr, target_sr)
    except Exception:
        audio, sr = librosa.load(path, sr=target_sr)

    if audio is None or len(audio) < target_sr:
        raise ValueError("Audio too short or corrupted")

    return audio

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_audio(audio):
    features = extract_logmel(audio)

    if features is None or np.isnan(features).any():
        raise ValueError("Feature extraction failed")

    features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    label = "Bonafide (Real)" if np.argmax(probs) == 1 else "Spoof (Fake)"
    confidence = float(np.max(probs)) * 100

    return label, confidence, probs

# -----------------------------
# UI
# -----------------------------
st.title("🎙️ Voice Deepfake Detection")
st.caption("AI-powered system to detect whether a voice is real or AI-generated")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "aac", "m4a"]
)

# -----------------------------
# HANDLE UPLOAD
# -----------------------------
if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name

        audio = load_audio_safe(input_path)

        label, confidence, _ = predict_audio(audio)

        if "Bonafide" in label:
            st.markdown(
                f"<div class='result-box real'>✅ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box fake'>🚨 {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(int(confidence))

        os.remove(input_path)

    except Exception as e:
        st.error(f"❌ Error processing audio file:\n{e}")
