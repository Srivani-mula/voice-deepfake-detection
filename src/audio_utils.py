import subprocess
import tempfile
import os

def convert_to_wav(input_path, target_sr=16000):

    if input_path.lower().endswith(".wav"):
        return input_path

    output_path = tempfile.mktemp(suffix=".wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(target_sr),
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and add it to PATH."
        )

    return output_path
