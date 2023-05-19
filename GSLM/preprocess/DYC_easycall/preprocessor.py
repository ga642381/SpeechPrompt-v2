import argparse
import os
import re
import subprocess
from collections import Counter
from itertools import groupby
from pathlib import Path

import pandas as pd
from torch.utils.data import random_split
from tqdm import tqdm

from DYC_easycall.dataset import EasyCallDataset, get_dataloader, TOM
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )
        dataset = EasyCallDataset(self.datarc["root_path"])
        val_test_len = int(len(dataset) / 10)
        train_len = len(dataset) - 2 * val_test_len
        dataset_split = random_split(dataset, [train_len, val_test_len, val_test_len])
        for split, subset in zip(["train", "valid", "test"], dataset_split):
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

    def class2index(self, file_name):
        label = TOM.get(file_name.split("/")[0])
        if "c" in file_name.split("/")[0]:
            label = 0
        assert label is not None
        return label
