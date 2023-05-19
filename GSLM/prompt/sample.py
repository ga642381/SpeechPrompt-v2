from fairseq import options
import os
import argparse
import utils

from fairseq_cli.sample import cli_main

storage_dir = utils.get_storage_dir()


def convert_to_fairseq_args(args):
    fairseq_parser = options.get_interactive_generation_parser()
    fairseq_parser.add_argument("--input_src", type=str, default=None, required=True)
    fairseq_parser.add_argument("--output", type=str, default=None)
    fairseq_parser.add_argument("--prompt_path", type=str, default=None)
    fairseq_parser.add_argument("--raw_file", type=str, default=None)

    # not useful
    fairseq_parser.add_argument("--samples_per_prompt", type=int, default=1)
    fairseq_parser.add_argument("--debug", action="store_true")
    fairseq_parser.add_argument("--slice_sampling", action="store_true")

    """
    --input_src: path to the test split
    --output: path to the output file
    --prompt_path: path to the prompt model
    --raw_file: path to the reduced units file, which provides file names for each utterance (users don't need to pass this arg)
    """
    input_src = storage_dir / "s2u_data" / args.downstream / f"verbalized_{args.vb_method}_data" / "test.txt"
    output = storage_dir / "exp_results" / args.downstream / args.exp_name / "samples" / "samples.json"
    prompt_path = storage_dir / "exp_results" / args.downstream / args.exp_name / "checkpoints" / "checkpoint_best.pt"
    raw_file = storage_dir / "s2u_data" / args.downstream / "reduced_units" / "test"

    speech_prompt_args = [
        f"--input_src={input_src}",
        f"--output={output}",
        f"--prompt_path={prompt_path}",
        f"--raw_file={raw_file}",
    ]

    bin_data_dir = storage_dir / "s2u_data" / args.downstream / f"{args.vb_method}_data_bin"
    model_path = storage_dir / "exp_results" / args.downstream / args.exp_name / "checkpoints" / "base_prompt_model.pt"

    path_args = [
        str(bin_data_dir),
        f"--path={model_path}",
    ]

    sampling_args = [
        "--user-dir=./prompt_GSLM",
        "--task=language_modeling",
        "--sampling",
        "--sampling-topk=1",
        "--seed=1",
        "--max-len-a=0",
        "--max-len-b=150",
        "--num-workers=12",
        "--prefix-size=-1",
        "--batch-size=32",
        "--fp16",
    ]

    fairseq_args = path_args + sampling_args + speech_prompt_args
    print(fairseq_args)
    return fairseq_parser, fairseq_args


def get_input_args():
    # ========== #
    #   Parser   #
    # ========== #
    parser = options.get_interactive_generation_parser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="SCR_lt_speech_commands_plen.5")
    parser.add_argument("--downstream", type=str, default="SCR_lt_speech_commands")
    parser.add_argument("--vb_method", type=str, default="freq")

    args = parser.parse_args()
    fairseq_parser, fairseq_args = convert_to_fairseq_args(args)

    return fairseq_parser, fairseq_args


def main():
    """
    e.g.
    python sample.py \
        --exp_name SCR_lt_speech_commands_plen.5 \
        --downstream SCR_lt_speech_commands \
        --vb_method freq
    """
    fairseq_parser, fairseq_args = get_input_args()
    cli_main(fairseq_parser, fairseq_args)


if __name__ == "__main__":
    main()
