import os
import pandas as pd
from pathlib import Path

import torchaudio
from tqdm import tqdm

from SCR_ar_speech_commands.dataset import SpeechCommandsDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        mapping = dict(zip(["train", "valid", "test"], ["train_full.csv", "dev_full.csv", "test_full.csv"]))
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
        mapping = {
            "A": "add",
            "B": "back",
            "C": "cancel",
            "D": "delete",
            "E": "confirm",
            "F": "continue",
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        return mapping[file_name.split("/")[-1].split("_")[0]]
