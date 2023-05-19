import os
from pathlib import Path

import numpy as np
import json
from tqdm import tqdm
from ER_iemocap.dataset import IEMOCAPDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)
 
    def random_split(self):
        meta_path = os.path.join(self.datarc["root_path"], "meta_data", self.datarc["test_fold"], "train_meta_data.json")
        with open(meta_path, 'r') as f:
            data = json.load(f)
            meta_data = data["meta_data"]
            np.random.seed(0)
            perm = np.random.permutation(len(meta_data))
            valid_size = int(len(meta_data) * self.datarc["valid_ratio"])
            train_data = {"labels": data["labels"], "meta_data": [meta_data[i] for i in perm[valid_size:]]}
            valid_data = {"labels": data["labels"], "meta_data": [meta_data[i] for i in perm[:valid_size]]}
        
        train_path = os.path.join(self.datarc["root_path"], "meta_data", self.datarc["test_fold"], "train_meta_data.json")
        valid_path = os.path.join(self.datarc["root_path"], "meta_data", self.datarc["test_fold"], "valid_meta_data.json")
        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(valid_path, 'w') as f:
            json.dump(valid_data, f)

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], self.datarc["test_fold"], "manifest"),
            exist_ok=True,
        )
        # split training data to training and validation
        self.random_split()

        for split in ["train", "valid", "test"]:
            meta_path = os.path.join(self.datarc["root_path"], "meta_data", self.datarc["test_fold"], f"{split}_meta_data.json")
            dataset = IEMOCAPDataset(self.datarc["root_path"], meta_path)
            
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
            )
            with open(Path(self.datarc["output_path"], self.datarc["test_fold"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                        f.write(f"{audio_path}\t{str(len(wav))}\n")

    def get_label(self):
        self.label = {}
        for split in ["train", "valid", "test"]:
            meta_path = os.path.join(self.datarc["root_path"], "meta_data", self.datarc["test_fold"], f"{split}_meta_data.json")
            dataset = IEMOCAPDataset(self.datarc["root_path"], meta_path)
            
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
            )

            for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                    self.label[str(audio_path)] = label
    
    def get_class(self, file_name):
        class_name = self.label[file_name]
        return class_name
        