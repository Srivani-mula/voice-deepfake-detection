import librosa
import numpy as np

def extract_logmel(audio_path, sr=16000, n_mels=64, max_len=100):
    y, _ = librosa.load(audio_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad / truncate time axis
    if log_mel.shape[1] < max_len:
        pad = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :max_len]

    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    return log_mel.astype(np.float32)
