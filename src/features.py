import librosa
import numpy as np

def extract_logmel(wav_path, sr=16000, n_mels=64, target_frames=96):
    y, sr = librosa.load(wav_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160
    )

    logmel = librosa.power_to_db(mel)

    # -------------------------------
    # FIX LENGTH (PAD / TRIM)
    # -------------------------------
    if logmel.shape[1] < target_frames:
        pad_width = target_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        logmel = logmel[:, :target_frames]

    return logmel.astype(np.float32)
