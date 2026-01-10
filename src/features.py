import librosa
import numpy as np


def extract_logmel(
    wav_path,
    sr=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256,
    max_frames=400
):
    """
    Extract log-mel spectrogram from a WAV file.
    Returns shape: (n_mels, time)
    """

    # Load audio
    audio, sr = librosa.load(wav_path, sr=sr)

    # Handle empty or silent audio
    if audio is None or len(audio) == 0:
        raise ValueError("Empty audio file")

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad / truncate to fixed length
    if log_mel.shape[1] < max_frames:
        pad_width = max_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_mel = log_mel[:, :max_frames]

    return log_mel.astype(np.float32)
