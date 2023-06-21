"""
from s3prl toolkit
"""

import re
import os
import random
from pathlib import Path
import torchaudio
import librosa

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000

class FreesoundDataset(Dataset):

    def __init__(self, speech, background, root_path):
        self.class_num = 2
        self.data = speech + background
        self.root_path = root_path

    def __getitem__(self, idx):
        wav_path = Path(self.root_path, self.data[idx]["audio_filepath"])
        offset = self.data[idx]["offset"]
        duration = self.data[idx]["duration"]
        wav, sr = librosa.load(path=wav_path, offset=offset, duration=duration)
        
        return wav, self.data[idx]["label"], wav_path, offset, duration

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

                

