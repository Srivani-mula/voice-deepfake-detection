import matplotlib.pyplot as plt
import librosa.display

from dataset import load_protocol
from features import extract_logmel

# Load protocol
df = load_protocol("train")
print("Protocol loaded")
print(df.head())

# Select one audio file
file_id = df.iloc[0]["file_id"]

audio_path = (
    r"C:\Users\SRIVANI\OneDrive\Desktop\DeepFake\data\ASVspoof2019\LA"
    r"\ASVspoof2019_LA_train\flac"
    fr"\{file_id}.flac"
)

print("\nTesting audio file:")
print(audio_path)

# Extract log-mel spectrogram
logmel = extract_logmel(audio_path)

print("\nFeature shape:", logmel.shape)

# ==============================
# DISPLAY SPECTROGRAM
# ==============================
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    logmel,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram (ASVspoof2019 LA)")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.show()
