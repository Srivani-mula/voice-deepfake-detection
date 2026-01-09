import librosa
import numpy as np

def load_audio(path, sr=16000, duration=4):
    audio, _ = librosa.load(path, sr=sr)

    max_len = sr * duration

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    return audio
