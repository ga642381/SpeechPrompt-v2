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
    def __init__(self, base_path, data):
        self.files = list(nlp2.get_files_from_dir(base_path, match='wav'))
        self.base_path = base_path
        self.data = data
        # i = 0
        # while i < len(self.files):
        #     wav_path = os.path.join(self.base_path, self.files[i])
        #     wav, sr = torchaudio.load(wav_path)
        #     wav = wav.squeeze(0)
        #     if wav.shape[0] < MIN_LENGTH:
        #         self.files.pop(i)
        #         i -= 1
        #     i += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print(self.files[idx])
        _, filename_ext = os.path.split(self.files[idx])
        filename = filename_ext.split('.')[0]
        wav, sr = torchaudio.load(self.files[idx])
        wav = wav.squeeze(0)
        
        label = self.data.loc[filename, 'Sarcasm']

        return wav.numpy(), label, Path(self.files[idx])

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