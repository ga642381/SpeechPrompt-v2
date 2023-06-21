import os
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import groupby
from pathlib import Path

import torchaudio
from tqdm import tqdm

from AuC_esc50.dataset import ESC50Dataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        meta_data = pd.read_csv(Path(self.datarc["root_path"], "meta", "esc50.csv"))
        split_df = {}
        split_df["test"] = meta_data[meta_data["fold"] == self.datarc["test_fold"]]
        train_val_df = meta_data[meta_data["fold"] != self.datarc["test_fold"]]
        split_df["train"], split_df["valid"] = train_test_split(
            train_val_df, test_size=self.datarc["valid_ratio"], random_state=1
        )

        for split in ["train", "valid", "test"]:
            dataset = ESC50Dataset(df=split_df[split], root_path=Path(self.datarc["root_path"], "audio"))
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
                collate_fn=dataset.collate_fn,
            )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                        if not audio_path.exists():
                            torchaudio.save(audio_path, wav.unsqueeze(0), 16000)

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def get_class(self, file_name):
        class_name = file_name.split("/")[-1].split(".")[0].split("-")[-1]
        return class_name
