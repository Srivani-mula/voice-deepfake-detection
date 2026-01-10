import numpy as np
import librosa

def extract_logmel(audio, sr, n_mels=64, n_fft=1024, hop_length=256):
    """
    Extract log-mel spectrogram from audio
    Returns shape: (1, 64, 100)
    """

    # Ensure audio is numpy array
    audio = np.asarray(audio, dtype=np.float32)

    # Trim or pad audio to 3 seconds
    target_len = sr * 3
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, max(0, target_len - len(audio))))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Log scale
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Ensure fixed time dimension (100)
    if logmel.shape[1] > 100:
        logmel = logmel[:, :100]
    else:
        logmel = np.pad(logmel, ((0, 0), (0, 100 - logmel.shape[1])))

    # Shape → (1, 64, 100)
    return logmel[np.newaxis, :, :]
