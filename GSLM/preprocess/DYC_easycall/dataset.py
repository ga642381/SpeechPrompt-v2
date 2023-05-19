import os
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import nlp2

MIN_LENGTH = 1000
TOM = {'f01': 1,
    'f02': 3,
    'f03': 1,
    'f05': 1,
    'f06': 1,
    'f07': 1,
    'f08': 1,
    'f09': 1,
    'f10': 5,
    'f11': 2,
    'm01': 3,
    'm02': 1,
    'm03': 3,
    'm04': 1,
    'm05': 4,
    'm06': 4,
    'm07': 4,
    'm08': 3,
    'm09': 1,
    'm10': 3,
    'm11': 5,
    'm12': 1,
    'm13': 1,
    'm14': 5,
    'm15': 1,
    'm16': 3,
    'm17': 1,
    'm18': 1,
    'm19': 3,
    'm20': 1,
}

class EasyCallDataset(Dataset):
    def __init__(self, base_path):
        self.files = list(nlp2.get_files_from_dir(base_path, match='wav'))
        self.base_path = base_path
        i = 0
        while i < len(self.files):
            wav_path = os.path.join(self.base_path, self.files[i])
            wav, sr = torchaudio.load(wav_path)
            wav = wav.squeeze(0)
            if wav.shape[0] < MIN_LENGTH:
                self.files.pop(i)
                i -= 1
            i += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, self.files[idx])
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)
        
        # label = 1

        label = TOM.get(os.path.relpath(wav_path,self.base_path).split('/')[0])
        if 'c' in os.path.relpath(wav_path,self.base_path).split('/')[0]:
            label = 0

        return wav.numpy(), label, Path(wav_path)

def collate_fn(samples):
    return zip(*samples)

def get_dataloader(dataset, batch_size, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )