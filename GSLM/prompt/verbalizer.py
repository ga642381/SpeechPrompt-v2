import json
import random
from pathlib import Path
import argparse
import utils

SPLITS = ["train", "valid", "test"]


def count_token(data):
    count_dict = {}
    for d in data:
        tokens = d.split()
        for t in tokens:
            if t in count_dict:
                count_dict[t] += 1
            else:
                count_dict[t] = 1
    return count_dict


def sort_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}


def mapping_table(file_path, method):
    src_count_dict = {}
    label_count_dict = {}
    with open(file_path) as f:
        data = f.readlines()

    srcs = [d.split("<s>")[0].strip() for d in data]
    labels = [d.split("<s>")[1].strip() for d in data]

    src_count_dict = count_token(srcs)
    label_count_dict = count_token(labels)

    src_sorted = sorted(src_count_dict, key=src_count_dict.get, reverse=True)
    label_sorted = sorted(label_count_dict, key=label_count_dict.get, reverse=True)

    print(f"[INFO] Number of source tokens: {len(src_sorted)}")
    print(f"[INFO] Number of label tokens: {len(label_sorted)}")

    if method == "freq":
        print(f"[INFO] Using frequency-based mapping")
        print(f"\n[INFO] source counting statistics: {sort_dict(src_count_dict)}")
        print(f"\n[INFO] label counting statistics: {sort_dict(label_count_dict)}")

    # If the number of task labels is larger than the number of source tokens, we should not use rule-based verbalizer.
    # Instead, we use learnable verbalzier.
    assert len(src_sorted) > len(label_sorted)

    # === mapping === #
    mapping_table = {}
    if method == "random":
        for i, t in enumerate(label_sorted):
            src_random = random.sample(src_sorted, len(src_sorted))
            mapping_table[t] = src_random[i]

    elif method == "freq":
        for i, t in enumerate(label_sorted):
            mapping_table[t] = src_sorted[i]

    elif method == "identity":
        for i, t in enumerate(label_sorted):
            mapping_table[t] = t

    elif method == "learnable":
        for i, t in enumerate(label_sorted):
            mapping_table[t] = str(i)

    else:
        raise NotImplementedError

    return mapping_table


def map_with_verbalizer(data, verbalizer, sep="<s>"):
    data = data.strip().split(sep)
    assert len(data) == 2  # source and labels
    source = data[0].strip()
    labels = data[1].strip()
    verbalized_labels = [verbalizer[l] for l in labels.split(" ")]
    target = " ".join(verbalized_labels)
    return source + " " + sep + " " + target


class Verbalizer:
    def __init__(self, args) -> None:
        # === directories === #
        self.root_dir = utils.get_storage_dir()
        # self.root_dir = Path(args.root_dir).resolve()
        self.downstream_dir = self.root_dir / "s2u_data" / args.downstream
        self.lm_dir = self.downstream_dir / "lm_dataset"
        self.prompt_lm_dir = self.downstream_dir / f"verbalized_{args.method}_data"

        # === verbalizer === #
        self.method = args.method
        self.verbalizer_path = self.prompt_lm_dir / f"verbalizer.json"

    def generate_verbalizer(self):
        # === define verbalizer === #
        verbalizer = mapping_table(self.lm_dir / "train.txt", method=self.method)
        print(f"\n[INFO] {self.method} Verbalzier: {verbalizer}")

        # === save verbalizer === #
        self.verbalizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.verbalizer_path, "w") as f:
            json.dump(verbalizer, f, indent=4, ensure_ascii=False)
        print(f"\n[INFO] Save verbalizer to {self.verbalizer_path}")

    def verbalize(self):
        # === load verbalizer === #
        with open(self.verbalizer_path) as f:
            verbalizer = json.load(f)
        print(f"\n[INFO] Load verbalizer from {self.verbalizer_path}")

        # === verbalize for each splits === #
        for split in SPLITS:
            data_path = self.lm_dir / f"{split}.txt"
            print(f"\n[INFO] Verbalizing {data_path}")
            out_path = self.prompt_lm_dir / f"{split}.txt"
            with open(data_path, "r") as f:
                data = f.readlines()
            verbalized_data = [map_with_verbalizer(d, verbalizer) for d in data]
            with open(out_path, "w") as f:
                f.writelines([line + "\n" for line in verbalized_data])
            print(f"[INFO] Save verbalized data to {out_path}")


def main(args):
    V = Verbalizer(args)
    if args.action == "generate_verbalizer":
        V.generate_verbalizer()
    elif args.action == "verbalize":
        V.verbalize()
    elif args.action == "all":
        V.generate_verbalizer()
        V.verbalize()


if __name__ == "__main__":
    """
    e.g.
    python verbalizer.py --downstream SCR_google_speech_commands --action all --method freq

    # process in separate steps:
    python verbalizer.py --downstream SCR_google_speech_commands --action generate_verbalizer --method freq
    python verbalizer.py --downstream SCR_google_speech_commands --action verbalize --method freq
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downstream", help="the downstream task to process", required=True, default="SCR_google_speech_commands"
    )
    parser.add_argument("--method", required=True, choices=["random", "freq", "learnable"])
    parser.add_argument("--action", help="generate_verbalizer or verbalize", default="all")

    args = parser.parse_args()
    main(args)
