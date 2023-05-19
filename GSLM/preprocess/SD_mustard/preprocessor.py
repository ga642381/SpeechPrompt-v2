import argparse
import os
import pickle
from collections import Counter
from itertools import groupby
from pathlib import Path

import pandas as pd
from torch.utils.data import random_split
from tqdm import tqdm
import json

from preprocessor_base import PreprocessorBase
from SD_mustard.dataset import MustardDataset, get_dataloader


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, config):
        self.global_config = global_config
        self.config = config
        self.taskrc = self.config["taskrc"]
        self.datarc = self.config["datarc"]
        self.fairseqrc = self.global_config["fairseqrc"]
        with open(self.datarc["json_path"], "r") as read_file:
            self.data = json.load(read_file)
        self.dep = self.datarc["speaker_dependent"]

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )
        if self.dep:
            mustard_dir = Path(os.path.abspath(__file__)).parent
            split_file = os.path.join(mustard_dir, "split_indices.p")
            with open(split_file, mode='rb') as file:
                self.split = pickle.load(file, encoding="latin1")
            split_no = self.datarc["k_fold"]
            train_val_idx = self.split[split_no][0]
            test_idx = self.split[split_no][1]
            fnames = list(self.data.keys())
            train_val_fnames = [fnames[_id] for _id in train_val_idx]
            test_fnames = [fnames[_id] for _id in test_idx]
        else:
            train_val_fnames = [
                k for k, v in self.data.items() if ( v['show'] != 'FRIENDS')
            ]
            test_fnames = [
                k for k, v in self.data.items() if ( v['show'] == 'FRIENDS')
            ]
        
        train_val_set = MustardDataset(self.datarc["root_path"], self.data, train_val_fnames)
        test_set = MustardDataset(self.datarc["root_path"], self.data, test_fnames)
        val_len = int(len(train_val_set) / 10)
        train_len = len(train_val_set) - val_len
        train_set, val_set = random_split(train_val_set, [train_len, val_len])
        for split, subset in zip(["train", "valid", "test"], (train_set, val_set, test_set)):
            dataloader = get_dataloader(
                dataset=subset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
            )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def get_class(self, file_name):
        key = file_name.split(".")[0]
        label = int(self.data[key]["sarcasm"])
        return label
