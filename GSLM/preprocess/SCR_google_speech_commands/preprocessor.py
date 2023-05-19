import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from SCR_google_speech_commands.dataset import (
    SpeechCommandsDataset,
    SpeechCommandsTestingDataset,
    get_dataloader,
    split_dataset,
)
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["root_path"], "temp", "_background_noise_"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        split_data_list = {}
        split_data_list["train"], split_data_list["valid"] = split_dataset(
            Path(self.datarc["root_path"], "speech_commands_v0.01")
        )

        for split in ["train", "valid", "test"]:
            if split == "test":
                dataset = SpeechCommandsTestingDataset(root_path=self.datarc["root_path"])
                dataloader = get_dataloader(
                    dataset=dataset,
                    batch_size=self.datarc["batch_size"],
                    num_workers=self.datarc["num_workers"],
                    balanced=False,
                )

            else:
                dataset = SpeechCommandsDataset(data_list=split_data_list[split], root_path=self.datarc["root_path"])
                dataloader = get_dataloader(
                    dataset=dataset,
                    batch_size=self.datarc["batch_size"],
                    num_workers=self.datarc["num_workers"],
                    balanced=True,
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

    @staticmethod
    def get_class(file_name):
        CLASSES = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "_unknown_",
            "_silence_",
        ]
        class_name = file_name.split("/")[1]
        if class_name == "_background_noise_":
            class_name = "_silence_"
        elif class_name not in CLASSES:
            class_name = "_unknown_"
        return class_name
