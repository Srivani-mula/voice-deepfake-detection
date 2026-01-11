import streamlit as st
import torch
import numpy as np
import librosa
import tempfile
import os

from model import CNNClassifier
from features import extract_logmel

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Voice Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

# ===============================
# UI STYLING
# ===============================
st.markdown("""
<style>
h1 {
    color: #0B1C2D;
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
.uncertain {
    background-color: #fff4e5;
    color: #8a5a00;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = CNNClassifier()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cnn_asvspoof.pth")

    if not os.path.exists(model_path):
        st.error("Model file cnn_asvspoof.pth not found")
        st.stop()

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_audio(wav_path):
    # Extract features (T, F)
    features = extract_logmel(wav_path)

    # Convert to tensor → (1, 1, F, T)
    features = torch.tensor(features).unsqueeze(1).float()

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # Confidence-based decision
    if pred == 1 and confidence > 0.65:
        label = "Bonafide (Real)"
        css = "real"
    elif pred == 0 and confidence > 0.65:
        label = "Spoof (Fake)"
        css = "fake"
    else:
        label = "Uncertain"
        css = "uncertain"

    return label, confidence, probs, css

# ===============================
# UI
# ===============================
st.title("🎙️ Voice Deepfake Detection")
st.caption(
    "Upload an audio file to check whether it is Real (Bonafide) or Fake (Spoof)"
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "flac"]
)

# ===============================
# HANDLE AUDIO
# ===============================
if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load audio safely
        audio, sr = librosa.load(tmp_path, sr=16000)

        if audio is None or len(audio) == 0:
            raise ValueError("Empty or invalid audio")

        # Predict
        label, confidence, probs, css = predict_audio(tmp_path)

        # Display result
        st.markdown(
            f"<div class='result-box {css}'>"
            f"{label}<br>"
            f"Confidence: {confidence*100:.2f}%"
            f"</div>",
            unsafe_allow_html=True
        )

        # Probabilities (for evaluation)
        st.write("Bonafide probability:", round(probs[1] * 100, 2), "%")
        st.write("Spoof probability:", round(probs[0] * 100, 2), "%")

        st.progress(int(confidence * 100))

        os.remove(tmp_path)

    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
