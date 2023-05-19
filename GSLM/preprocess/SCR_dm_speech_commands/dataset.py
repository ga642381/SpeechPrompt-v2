"""
from s3prl toolkit
"""

import re
import os
from random import randint
from pathlib import Path
import hashlib
import pandas as pd
from typing import List, Tuple, Union

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.sox_effects import apply_effects_file
from sklearn.model_selection import train_test_split

CLASSES = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19"
]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 19
        self.data = []

    def __getitem__(self, idx):
        class_name, audio_path = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0)

        return wav, self.class2index[class_name], audio_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, df, root_path, **kwargs):
        super().__init__()
        filename = [df[df.columns[0]][i].split("\t")[0] for i in range(len(df))]
        label = [df[df.columns[0]][i].split("\t")[1] for i in range(len(df))]

        omits = ["10", "13", "14", "15", "16", "19"]
        for class_name, audio_path in zip(label, filename):
            if class_name in omits:
                continue
            else:
                self.data.append((class_name, Path(audio_path)))
        
        class_counts = {class_name: 0 for class_name in CLASSES}
        for class_name, _ in self.data:
            class_counts[class_name] += 1

        self.root_path = root_path

    def __getitem__(self, idx):
        wav, label, audio_path = super().__getitem__(idx)
        return wav, label, audio_path


def get_dataloader(dataset, batch_size, num_workers, collate_fn, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def split_dataset(path, filename, dev_size):
    df = pd.read_csv(Path(path, filename))
    train, dev = train_test_split(df, test_size=dev_size, random_state=1)
    train.to_csv(Path(path, "train_split.csv"), index=False)
    dev.to_csv(Path(path, "dev_split.csv"), index=False)
