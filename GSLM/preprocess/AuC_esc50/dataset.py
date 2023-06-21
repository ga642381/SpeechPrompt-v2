"""
from s3prl toolkit
"""

import re
import os
import random
from pathlib import Path
import pandas as pd
import torchaudio

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000
ORIGIN_SAMPLE_RATE = 44100

class ESC50Dataset(Dataset):

    def __init__(self, df, root_path):
        self.class_num = 50
        filename = df["filename"].values.tolist()
        label = df["target"].values.tolist()
        data = [
            (class_label, Path(root_path, audio_path))
            for class_label, audio_path in zip(label, filename)
        ]
        self.data = data
        self.resampler = Resample(ORIGIN_SAMPLE_RATE, SAMPLE_RATE)

    def __getitem__(self, idx):
        class_label, audio_path = self.data[idx]
        wav, _ = torchaudio.load(audio_path)
        wav = self.resampler(wav).squeeze(0)
        
        return wav.numpy(), class_label, audio_path

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

                

