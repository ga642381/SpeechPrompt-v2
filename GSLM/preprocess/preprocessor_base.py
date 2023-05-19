import os
import subprocess
from itertools import groupby
from pathlib import Path

from tqdm import tqdm


class PreprocessorBase:
    def __init__(self, global_config, dataset_config):
        self.global_config = global_config
        self.dataset_config = dataset_config
        self.fairseqrc = self.global_config["fairseqrc"]
        self.taskrc = self.dataset_config["taskrc"]
        self.datarc = self.dataset_config["datarc"] | self.global_config["datarc"]

        if not "output_path" in self.dataset_config:
            self.datarc["output_path"] = os.path.join(self.datarc["output_dir"], self.dataset_config["name"])

    def generate_manifest(self):
        raise NotImplementedError("Please provide a function to retrieve manifest files!")

    def get_class(self):
        raise NotImplementedError("Please provide a mapping between file names and labels!")

    def quantize(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "quantized"),
            exist_ok=True,
        )
        prep_dir = Path(os.path.abspath(__file__)).parent
        python_file = os.path.join(prep_dir, "speech2unit/quantize_with_kmeans.py")

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

    def reduce_quantized(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "reduced_units"),
            exist_ok=True,
        )

        for split in ["train", "valid", "test"]:
            quantized_file_path = Path(self.datarc["output_path"], "quantized", f"{split}")
            output_path = Path(self.datarc["output_path"], "reduced_units", f"{split}")

            with open(quantized_file_path, "r") as f:
                with open(output_path, "w") as f_output:
                    for line in tqdm(f.readlines(), desc=split):
                        file_name, tokens = line.rstrip("\n").split("|")

                        if self.taskrc["merge"]:
                            token_list = tokens.split()
                            merged_tokens_list = [x[0] for x in groupby(token_list)]
                            tokens = " ".join(merged_tokens_list)

                        reduced_units_line = f"{file_name}|{tokens}|{self.get_class(file_name)}\n"

                        f_output.write(reduced_units_line)

    def create_lm_dataset(self):
        assert self.fairseqrc["LM_datatype"] in ["src-only", "src-tgt"]

        # TODO : add merge the redundant parts
        if self.fairseqrc["LM_datatype"] == "src-only":
            for split in ["train", "valid", "test"]:
                ###########
                #  input  #
                ###########
                data_dir = Path(self.datarc["output_path"], "reduced_units")
                in_path = data_dir / split
                if not in_path.is_file():
                    print(f"[INFO] File not found: {in_path}")
                    continue
                with open(in_path, "r") as f:
                    data = f.readlines()
                ###########
                #  output #
                ###########
                # == write src == #
                dest_dir = Path(self.datarc["output_path"], "lm_dataset")
                dest_dir.mkdir(parents=True, exist_ok=True)
                out_path = dest_dir / (split + ".txt")
                data = [d.split("|") for d in data]
                out_data = [d[1] + " " + "<s>" + " " + d[2] for d in tqdm(data, desc=split)]

                with open(out_path, "w") as f:
                    f.writelines(out_data)

        elif self.fairseqrc["LM_datatype"] == "src-tgt":
            for split in ["train", "valid", "test"]:
                ###########
                #  input  #
                ###########
                data_dir = Path(self.datarc["output_path"], "reduced_units")
                in_path = data_dir / split
                if not in_path.is_file():
                    print(f"[INFO] File not found: {in_path}")
                    continue
                with open(in_path, "r") as f:
                    data = f.readlines()

                data = [d.split("|") for d in data]
                """
                please refer to an example to know how to format the data:
                https://fairseq.readthedocs.io/en/latest/tutorial_classifying_names.html
                """

                ###########
                #  output #
                ###########
                dest_dir = Path(self.datarc["output_path"], "lm_dataset")
                dest_dir.mkdir(parents=True, exist_ok=True)

                # == write src == #
                src_path = dest_dir / (split + ".en")
                src_data = [d[1] + "\n" for d in tqdm(data, desc=split + "-src")]
                with open(src_path, "w") as f:
                    f.writelines(src_data)

                # == write tgt == #
                tgt_path = dest_dir / (split + ".es")
                tgt_data = [d[2] for d in tqdm(data, desc=split + "-tgt")]
                with open(tgt_path, "w") as f:
                    f.writelines(tgt_data)
        else:
            raise NotImplementedError

    def all(self):
        self.generate_manifest()
        self.quantize()
        self.reduce_quantized()
        self.create_lm_dataset()
