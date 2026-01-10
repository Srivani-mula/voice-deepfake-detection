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
body { background-color: #f5f7fb; }
.main { background-color: #f5f7fb; }
.result-box {
    padding: 1.5rem;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
.real { background-color: #e6f4ea; color: #1e7f43; }
.fake { background-color: #fdecea; color: #b71c1c; }
h1 { color: #0B1C2D !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONSTANTS (MATCH TRAINING)
# -----------------------------
TARGET_SR = 16000
FIXED_DURATION = 4.0   # seconds
FIXED_FRAMES = 400     # must match training

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = CNNClassifier()
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "cnn_asvspoof.pth")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# FEATURE POST-PROCESSING
# -----------------------------
def fix_length(feat, max_len=FIXED_FRAMES):
    if feat.shape[0] < max_len:
        pad = max_len - feat.shape[0]
        feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")
    else:
        feat = feat[:max_len, :]
    return feat

def normalize(feat):
    return (feat - feat.mean()) / (feat.std() + 1e-9)

# -----------------------------
# PREDICTION
# -----------------------------
def predict_audio(wav_path, max_duration=3.0):
    # Load audio
    y, sr = librosa.load(wav_path, sr=16000)

    # enforce mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # ensure length ~ max_duration seconds
    target_samples = int(sr * max_duration)

    if len(y) < target_samples:
        # pad
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        # trim
        y = y[:target_samples]

    # extract features
    features = extract_logmel(y)   # shape (T, F)

    # shape to (1,1,T,F)
    features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()

    # model forward
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
st.caption("Detect whether a voice is real or AI-generated")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "aac", "m4a"]
)

if uploaded_file:
    st.audio(uploaded_file)

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        audio, _ = librosa.load(
            temp_path,
            sr=TARGET_SR,
            duration=FIXED_DURATION
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
            wav_path = wav_tmp.name
            sf.write(wav_path, audio, TARGET_SR)

        label, confidence, probs = predict_audio(wav_path)

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

        os.remove(temp_path)
        os.remove(wav_path)

    except Exception as e:
        st.error(f"❌ Error processing audio:\n{e}")
