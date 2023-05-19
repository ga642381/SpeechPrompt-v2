import os
import random
from pathlib import Path
import json

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import nlp2

MIN_LENGTH = 1000


class MustardDataset(Dataset):
    def __init__(self, base_path, data, fnames):
        self.base_path = base_path
        self.data = data
        self.fanmes = fnames

    def __len__(self):
        return len(self.fanmes)

    def __getitem__(self, idx):
        fname = self.fanmes[idx]
        fpath = os.path.join(self.base_path, f"{fname}.wav")
        wav, sr = torchaudio.load(fpath)
        wav = wav.squeeze(0)
        
        label = self.data[fname]['sarcasm']

        return wav.numpy(), label, Path(fpath)

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