import argparse
import os
import re
import subprocess
from collections import Counter
from itertools import groupby
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm

from preprocessor_base import PreprocessorBase
from IC_fluent_commands.dataset import FluentCommandsDataset, get_dataloader


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)
        self.get_dataset()

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        for split in ["train", "valid", "test"]:
            dataset = FluentCommandsDataset(self.df[split], self.datarc["root_path"], self.Sy_intent)
            dataloader = get_dataloader(
                dataset=dataset,
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

    def get_dataset(self):
        self.df = {}
        train_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "train_data.csv"))
        valid_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "valid_data.csv"))
        test_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "test_data.csv"))

        Sy_intent = {"action": {}, "object": {}, "location": {}}

        values_per_slot = []
        count = 0
        for slot in ["action", "object", "location"]:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                if self.taskrc["no_overlap"]:
                    index = index + count
                Sy_intent[slot][value] = index
                Sy_intent[slot][index] = value
            count += len(slot_values)
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        self.df["train"] = train_df
        self.df["valid"] = valid_df
        self.df["test"] = test_df

    def get_class(self, file_name):
        for key, df in self.df.items():
            if file_name in list(df["path"].values):
                act = df[df["path"] == file_name]["action"].values[0]
                obj = df[df["path"] == file_name]["object"].values[0]
                loc = df[df["path"] == file_name]["location"].values[0]

                return f"{act} {obj} {loc}"

    # deprecated
    def class2index(self, file_name):
        for key, df in self.df.items():
            if file_name in list(df["path"].values):
                act_idx = self.Sy_intent["action"][df[df["path"] == file_name]["action"].values[0]]
                obj_idx = self.Sy_intent["object"][df[df["path"] == file_name]["object"].values[0]]
                loc_idx = self.Sy_intent["location"][df[df["path"] == file_name]["location"].values[0]]

                return f"{act_idx} {obj_idx} {loc_idx}"
