import librosa
import numpy as np

def extract_logmel(audio, sr, n_mels=64, max_len=100):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512
    )

    logmel = librosa.power_to_db(mel)

    # FIXED LENGTH (IMPORTANT)
    if logmel.shape[1] < max_len:
        pad_width = max_len - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad_width)), mode='constant')
    else:
        logmel = logmel[:, :max_len]

    return logmel
