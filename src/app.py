import os
import tempfile
import numpy as np
import torch
import streamlit as st

from model import CNNClassifier
from features import extract_logmel


# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="Voice Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Voice Deepfake Detection")
st.write("Upload an audio file to check whether it is **Real (Bonafide)** or **Fake (Spoof)**.")


# =====================================================
# Load Model (cached)
# =====================================================
@st.cache_resource
def load_model():
    model = CNNClassifier()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "cnn_asvspoof.pth")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model



model = load_model()


# =====================================================
# Prediction Function
# =====================================================
def predict_audio(wav_path):
    features = extract_logmel(wav_path)   # (64, 96)

    features = torch.tensor(features)
    features = features.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 96)

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    label = "Bonafide (Real)" if np.argmax(probs) == 1 else "Spoof (Fake)"
    confidence = float(np.max(probs)) * 100

    return label, confidence



# =====================================================
# File Upload UI
# =====================================================
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "aac", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        # Save uploaded file to temp WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            wav_path = tmp.name

        # Predict
        label, confidence, probs = predict_audio(wav_path)

        # Display result
        if "Bonafide" in label:
            st.success(f"✅ {label}")
        else:
            st.error(f"🚨 {label}")

        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        st.write("### Class Probabilities")
        st.write({
            "Bonafide": float(probs[0]),
            "Spoof": float(probs[1])
        })

        # Cleanup
        os.remove(wav_path)

    except Exception as e:
        st.error(f"❌ Error processing audio:\n{e}")
