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
    st.audio(uploaded_file)
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name

        # Load audio safely
        audio, sr = librosa.load(input_path, sr=16000, mono=True)

        # Convert to WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
            wav_path = wav_tmp.name
            sf.write(wav_path, audio, sr)

        # Predict
        label, confidence = predict_audio(wav_path)

        # Display result
        if "Bonafide" in label:
            st.markdown(
                f"<div class='result-box real'>✅ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-box fake'>🚨 {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True,
            )

        st.progress(int(confidence))

        # Cleanup
        os.remove(input_path)
        os.remove(wav_path)
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
