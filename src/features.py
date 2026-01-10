import librosa
import numpy as np

ddef extract_logmel(wav_path, target_len=4):
    audio, sr = librosa.load(wav_path, sr=16000)

    # FIX AUDIO LENGTH (same as training)
    max_len = target_len * sr
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=100,
        n_fft=1024,
        hop_length=512
    )

    logmel = librosa.power_to_db(mel)
    logmel = logmel.T  # (T, F)

    return logmel

