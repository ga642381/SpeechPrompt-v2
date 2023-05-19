import os
import pandas as pd
from pathlib import Path

import torchaudio
from tqdm import tqdm

from SCR_dm_speech_commands.dataset import SpeechCommandsDataset, get_dataloader, split_dataset
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        split_dataset(self.datarc["data_path"], "train_full.csv", self.datarc["dev_size"])
        mapping = dict(zip(["train", "valid", "test"], ["train_split.csv", "dev_split.csv", "test_full.csv"]))
        for split in ["train", "valid", "test"]:
            df = pd.read_csv(Path(self.datarc["data_path"], mapping[split]))
            dataset = SpeechCommandsDataset(df=df, root_path=self.datarc["root_path"])
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
        CLASSES = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "close",
            "up",
            "down",
            "previous",
            "next",
            "in",
            "out",
            "left",
            "right",
            "home",
        ]
        mapping = {i + 1: CLASSES[i] for i in range(len(CLASSES))}
        return mapping[int(file_name.split("/")[1])]
