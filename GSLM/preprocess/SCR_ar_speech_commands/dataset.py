"""
from s3prl toolkit
"""

import re
import os
from random import randint
from pathlib import Path
import hashlib
from typing import List, Tuple, Union

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.sox_effects import apply_effects_file

CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 16
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
        data = [
            (class_name, Path(audio_path))
            for class_name, audio_path in zip(label, filename)
        ]
        
        class_counts = {class_name: 0 for class_name in CLASSES}
        for class_name, _ in data:
            class_counts[class_name] += 1

        sample_weights = [
            len(data) / class_counts[class_name] for class_name, _ in data
        ]
        
        self.data = data
        self.sample_weights = sample_weights
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
