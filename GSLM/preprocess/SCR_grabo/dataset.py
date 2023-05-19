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

CLASSES = [f"recording{i}" for i in range(1, 37)]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 36
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

    def __init__(self, data_list, root_path, **kwargs):
        super().__init__()
        
        data = [
            (class_name, Path(audio_path))
            for class_name, audio_path in data_list
        ]

        
        self.data = data
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

def split_dataset(root_dir):
    train_list, valid_list, test_list = [], [], []
    for speaker in Path(root_dir, "speakers").iterdir():
        for entry in Path(speaker, "spchdatadir").iterdir():
            cnt = 0
            for audio_path in entry.glob("*.wav"):
                if cnt < 2:
                    train_list.append((entry.name, audio_path))
                elif cnt < 6:
                    valid_list.append((entry.name, audio_path))
                else:
                    test_list.append((entry.name, audio_path))
                cnt += 1
    return train_list, valid_list, test_list

