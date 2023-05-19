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
    "de",
    "en",
    "es",
    "fr",
    "it",
    "ru"
]

class VoxforgeDataset(Dataset):

    def __init__(self, data_list):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 6
        self.data = data_list

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

def split_dataset(root_dir):
    train_list, valid_list, test_list = [], [], []
    for download in Path(root_dir, "downloads").iterdir():
        audios = []
        for speaker in Path(download, "Trunk", "Audio", "Main", "16kHz_16bit").iterdir():
            entry = Path(speaker, "wav")
            audios += list(entry.glob("*.wav"))
            if len(audios) > 1800:
                break
        
        random.seed(1)
        random.shuffle(audios)
        for audio_path in audios[:1200]:
            train_list.append((download.name, audio_path))
        for audio_path in audios[1200:1500]:
            valid_list.append((download.name, audio_path))
        for audio_path in audios[1500:1800]:
            test_list.append((download.name, audio_path))
    
    return train_list, valid_list, test_list
                

