import os
import pandas as pd
from pathlib import Path

import torchaudio
from tqdm import tqdm

from SCR_lt_speech_commands.dataset import SpeechCommandsDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        mapping = dict(zip(["train", "valid", "test"], ["train_limit20.csv", "dev_full.csv", "test_full.csv"]))
        df_noise = pd.read_csv(Path(self.datarc["data_path"], "noise_full.csv"))
        for split in ["train", "valid", "test"]:
            df = pd.read_csv(Path(self.datarc["data_path"], mapping[split]))
            # concat noise (silence) into train, valid, test dfs
            # ref: https://github.com/dodohow1011/SpeechAdvReprogram/blob/main/LT-SCR/main.py
            if split == "test":
                df = pd.concat([df, df_noise[:10]], ignore_index=True)
            elif split == "valid":
                df = pd.concat([df, df_noise[10:20]], ignore_index=True)
            elif split == "train":
                df = pd.concat([df, df_noise[20:]], ignore_index=True)

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
            "ne",
            "ačiū",
            "stop",
            "įjunk",
            "išjunk",
            "į_viršų",
            "į_apačią",
            "į_dešinę",
            "į_kairę",
            "startas",
            "pauzė",
            "labas",
            "iki",
            "unknown",
            "silence",
        ]
        class_name = file_name.split("/")[1]
        if class_name == "_background_noise_":
            class_name = "silence"

        if class_name not in CLASSES:
            class_name = "unknown"

        return class_name
