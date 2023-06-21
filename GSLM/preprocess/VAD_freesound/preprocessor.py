import os
import json
from pathlib import Path

import torchaudio
from tqdm import tqdm

from VAD_freesound.dataset import FreesoundDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        mapping = {"train": "training", "valid": "validation", "test": "testing"}
        for split in ["train", "valid", "test"]:
            # Read manifest file generated from NeMo preprocessing script
            speech_data, background_data = [], []
            with open(Path(self.datarc["manifest_path"], f"balanced_speech_{mapping[split]}_manifest.json"), "r") as f:
                for line in f:
                    speech_data.append(json.loads(line))
            with open(
                Path(self.datarc["manifest_path"], f"balanced_background_{mapping[split]}_manifest.json"), "r"
            ) as f:
                for line in f:
                    background_data.append(json.loads(line))

            dataset = FreesoundDataset(
                speech=speech_data, background=background_data, root_path=Path(self.datarc["root_path"])
            )
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
                collate_fn=dataset.collate_fn,
            )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_paths, offsets, durs in tqdm(dataloader, desc=split):
                    for wav, label, audio_path, offset, dur in zip(wavs, labels, audio_paths, offsets, durs):
                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\t{offset}\t{dur}\n")

    def get_class(self, file_name):
        if file_name.split("/")[-3] == "freesound":
            class_name = "background"
        else:
            class_name = "speech"
        return class_name
