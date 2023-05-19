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
from sklearn.model_selection import train_test_split

CLASSES = [
    "american",
    "australian",
    "bangla",
    "british",
    "indian",
    "malayalam",
    "odiya",
    "telugu",
    "welsh"
]

class AccentDBDataset(Dataset):

    def __init__(self, data_list, root_path):
        self.root_path = root_path
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 9
        
        data = [
            (class_name, Path(audio_path))
            for class_name, audio_path in data_list
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

def split_dataset(root_path):
    data_list = {}
    audios, labels = [], []
    for language in Path(root_path).iterdir():
        for speaker in Path(language).iterdir():
            audios += list(Path(speaker).glob("*.wav"))
            labels += [str(language).split("/")[-1]] * (len(audios) - len(labels))
    
    train_audios, test_audios, train_labels, test_labels = train_test_split(audios, labels, test_size=0.2, random_state=0)
    train_audios, valid_audios, train_labels, valid_labels = train_test_split(train_audios, train_labels, test_size=0.2, random_state=0)
    data_list["train"] = [
        (train_label, train_audio)
        for train_label, train_audio in zip(train_labels, train_audios)
    ]
    data_list["valid"] = [
        (valid_label, valid_audio)
        for valid_label, valid_audio in zip(valid_labels, valid_audios)
    ]
    data_list["test"] = [
        (test_label, test_audio)
        for test_label, test_audio in zip(test_labels, test_audios)
    ]
    
    return data_list