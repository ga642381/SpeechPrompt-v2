import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from FSD_asvspoof.dataset import ASVSpoofDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)
        self.get_label()

    def get_label(self):
        self.label_dict = {}
        for split in ["train", "dev", "eval"]:
            if split == "train":
                meta_data = Path(
                    self.datarc["root_path"], "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.trn.txt"
                )
            else:
                meta_data = Path(
                    self.datarc["root_path"], "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.trl.txt"
                )
            with open(meta_data, "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.label_dict[f'{line.split(" ")[1]}.flac'] = line.split(" ")[4].replace("\n", "")

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        mapping = {"train": "train", "valid": "dev", "test": "eval"}
        for split in ["train", "valid", "test"]:
            if split == "train":
                meta_data = Path(
                    self.datarc["root_path"],
                    "ASVspoof2019_LA_cm_protocols",
                    f"ASVspoof2019.LA.cm.{mapping[split]}.trn.txt",
                )
            else:
                meta_data = Path(
                    self.datarc["root_path"],
                    "ASVspoof2019_LA_cm_protocols",
                    f"ASVspoof2019.LA.cm.{mapping[split]}.trl.txt",
                )
            dataset = ASVSpoofDataset(
                meta_data=meta_data,
                root_path=Path(self.datarc["root_path"], f"ASVspoof2019_LA_{mapping[split]}", "flac"),
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
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                        if not audio_path.exists():
                            torchaudio.save(audio_path, wav.unsqueeze(0), 16000)

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def get_class(self, file_name):
        class_name = self.label_dict[file_name.split("/")[-1]]
        return class_name
