# src/features.py

import numpy as np
import librosa


def extract_logmel(
    audio_path,
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
    max_len=400
):
    """
    Extract log-mel spectrogram from an audio file.

    Returns:
        np.ndarray of shape (1, n_mels, max_len)
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # If audio is too short, pad it
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Convert to log scale
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    # Pad / trim to fixed length
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :max_len]

    # Add batch dimension
    log_mel = np.expand_dims(log_mel, axis=0)

    return log_mel
