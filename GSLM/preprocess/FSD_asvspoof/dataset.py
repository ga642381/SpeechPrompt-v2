"""
from s3prl toolkit
"""

import re
import os
import random
from pathlib import Path
import torchaudio

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

CLASSES = [
    "bonafide",
    "spoof"
]

class ASVSpoofDataset(Dataset):

    def __init__(self, meta_data, root_path):
        self.root_path = root_path
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 2
        with open(meta_data, "r") as f:
            lines = f.readlines()
            data = [
                (line.split(" ")[4].replace("\n", ""), Path(self.root_path, f'{line.split(" ")[1]}.flac'))
                for line in lines
            ]
        self.data = data

    def __getitem__(self, idx):
        wav_path = self.data[idx][1]
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)
        
        return wav.numpy(), self.class2index[self.data[idx][0]], wav_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


def get_dataloader(dataset, batch_size, num_workers, collate_fn, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )           

