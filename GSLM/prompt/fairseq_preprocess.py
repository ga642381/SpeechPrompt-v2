import json
import os
import argparse
from pathlib import Path
import utils


def fairseq_preprocess(args):
    root_dir = utils.get_storage_dir()
    verbalized_dir = (Path(root_dir) / "s2u_data" / args.downstream / f"verbalized_{args.vb_method}_data").resolve()
    dict_path = Path(root_dir) / "pretrained_models" / "GSLM" / "dict.txt"
    verbalize_method = args.vb_method

    dest_dir = verbalized_dir.parent / f"{verbalize_method}_data_bin"
    train_data = verbalized_dir / "train.txt"
    valid_data = verbalized_dir / "valid.txt"
    test_data = verbalized_dir / "test.txt"
    if os.path.exists(dest_dir):
        print(f"[INFO] {dest_dir} exists! SKIP!")
        return
    print(f"[INFO] Preprocessing {verbalized_dir}")
    cmd = f"fairseq-preprocess --only-source --srcdict {dict_path}\
    --trainpref {train_data} --validpref {valid_data} --testpref {test_data}\
    --destdir {dest_dir} --workers 16"
    os.system(cmd)

    print(f"[INFO] Saved fairseq data to {dest_dir}")


def main(args):
    print("[INFO] Starting Fairseq Preprocess... (Converting data into binary format)")
    fairseq_preprocess(args)


if __name__ == "__main__":
    """
    python fairseq_preprocess.py --downstream SCR_google_speech_commands --vb_method freq
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downstream", help="the downstream task to process", required=True, default="SCR_google_speech_commands"
    )
    parser.add_argument("--vb_method", choices=["random", "freq", "learnable"])

    args = parser.parse_args()
    main(args)
