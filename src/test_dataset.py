import os
from dataset import load_protocol

# Load protocol
df = load_protocol("train")

print(f"Total samples: {len(df)}")
print(df.head())

# Base audio directory (CHANGE ONLY IF YOUR PATH IS DIFFERENT)
BASE_AUDIO_DIR = (
    r"C:\Users\SRIVANI\OneDrive\Desktop\DeepFake\data"
    r"\ASVspoof2019\LA\ASVspoof2019_LA_train\flac"
)

# Construct full path from file_id
df["path"] = df["file_id"].apply(
    lambda x: os.path.join(BASE_AUDIO_DIR, f"{x}.flac")
)

# Check missing files
missing = df[~df["path"].apply(os.path.exists)]

print(f"\nMissing audio files: {len(missing)}")

if len(missing) > 0:
    print(missing.head())
else:
    print("✅ All audio files found!")
