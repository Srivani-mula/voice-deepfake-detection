import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from features import extract_logmel


# =====================================================
# LOAD PROTOCOL FILE
# =====================================================
def load_protocol(split):
    """
    Load ASVspoof 2019 LA protocol file.
    split: 'train', 'dev', or 'eval'
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    protocol_files = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    protocol_path = os.path.join(
        base_dir,
        "data",
        "ASVspoof2019",
        "LA",
        "ASVspoof2019_LA_cm_protocols",
        protocol_files[split],
    )

    df = pd.read_csv(protocol_path, sep=" ", header=None)
    df = df.iloc[:, :5]
    df.columns = ["speaker", "file_id", "env", "attack", "label"]

    return df


# =====================================================
# DATASET CLASS
# =====================================================
class ASVspoofDataset(Dataset):
    def __init__(self, split="train"):
        assert split in ["train", "dev", "eval"]

        self.split = split
        self.df = load_protocol(split)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        audio_dir_map = {
            "train": "ASVspoof2019_LA_train",
            "dev": "ASVspoof2019_LA_dev",
            "eval": "ASVspoof2019_LA_eval",
        }

        self.audio_dir = os.path.join(
            base_dir,
            "data",
            "ASVspoof2019",
            "LA",
            audio_dir_map[split],
            "flac",
        )

        # ✅ CORRECT LABEL MAPPING
        self.label_map = {
            "bonafide": 0,  # REAL
            "spoof": 1      # FAKE
        }

        # ✅ Print TRUE class distribution (ONCE)
        if split == "dev":
            print("\nValidation set label distribution:")
            print(self.df["label"].value_counts())
            print("-" * 40)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_id = row["file_id"]
        label_str = row["label"]

        label = self.label_map[label_str]

        audio_path = os.path.join(self.audio_dir, f"{file_id}.flac")

        if not os.path.exists(audio_path):
            return self.__getitem__((idx + 1) % len(self.df))

        features = extract_logmel(audio_path)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return features, label


# =====================================================
# SANITY CHECK
# =====================================================
if __name__ == "__main__":
    dataset = ASVspoofDataset(split="train")
    print("Total samples:", len(dataset))

    x, y = dataset[0]
    print("Feature shape:", x.shape)
    print("Label:", y.item(), "(0 = REAL, 1 = FAKE)")
