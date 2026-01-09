import librosa
from dataset import load_protocol

df = load_protocol("train")
audio, sr = librosa.load(df.iloc[0]["path"], sr=16000)

print(sr, len(audio))
