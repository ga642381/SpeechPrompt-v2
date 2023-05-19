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
import pandas as pd

from SD_mustardpp.dataset import MustardDataset, get_dataloader


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.taskrc = self.config["taskrc"]
        self.datarc = self.config["datarc"]
        self.fairseqrc = self.config["fairseqrc"]
        self.data = pd.read_csv(self.datarc["csv_path"], index_col="KEY")

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )
        dataset = MustardDataset(self.datarc["root_path"], self.data)
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
        key = file_name.split(".")[0]
        label = int(self.data.loc[key, "Sarcasm"])
        return label

    def postprocess(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "preprocessed"),
            exist_ok=True,
        )

        for split in ["train", "valid", "test"]:
            quantized_file_path = Path(self.datarc["output_path"], "quantized", f"{split}")
            output_path = Path(self.datarc["output_path"], "preprocessed", f"{split}")

            with open(quantized_file_path, "r") as f:
                with open(output_path, "w") as f_output:
                    for line in tqdm(f.readlines(), desc=split):
                        file_name, tokens = line.rstrip("\n").split("|")

                        if self.taskrc["merge"]:
                            token_list = tokens.split()
                            merged_tokens_list = [x[0] for x in groupby(token_list)]
                            tokens = " ".join(merged_tokens_list)

                        preprocessed_line = f"{file_name}|{tokens}|{self.class2index(file_name)}\n"

                        f_output.write(preprocessed_line)

    def quantized(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "quantized"),
            exist_ok=True,
        )

        python_file = Path(
            self.fairseqrc["root_path"],
            "examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py",
        )

        for split in ["train", "valid", "test"]:
            manifest_path = Path(self.datarc["output_path"], "manifest", f"{split}.manifest")
            output_path = Path(self.datarc["output_path"], "quantized", f"{split}")

            subprocess.call(
                [
                    "python",
                    python_file,
                    "--feature_type",
                    self.fairseqrc["feature_type"],
                    "--kmeans_model_path",
                    self.fairseqrc["km_model_path"],
                    "--acoustic_model_path",
                    self.fairseqrc["ssl_model_path"],
                    "--layer",
                    str(self.fairseqrc["layer"]),
                    "--manifest_path",
                    manifest_path,
                    "--out_quantized_file_path",
                    output_path,
                    "--extension",
                    ".wav",
                    "--full_file_name",
                ]
            )
