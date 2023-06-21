# This script is used to train the prompt model. Orignally, we should run fairseq-train with the input_args.
# We use this script for the convenience of developing and debugging.

# TODO: Let user only pass args to SpeechPrompt parser, and let SpeechPrompt parser pass the args to fairseq-train parser.
# Don't let user directly pass args to fairseq-train parser.
from fairseq import options
import argparse
import yaml
from fairseq_cli.train import cli_main
import utils

storage_dir = utils.get_storage_dir()


def get_task_config(config_path, downstream):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    num_classes = config["task_num_classes"][downstream]
    batch_size = config["task_batch_size"][downstream]
    return num_classes, batch_size


def convert_to_fairseq_args(args):
    fairseq_parser = options.get_training_parser()
    fairseq_parser.add_argument("--prompt_length", type=int)
    fairseq_parser.add_argument("--num_classes", type=int)
    fairseq_parser.add_argument("--deep_prompt", action="store_true")
    fairseq_parser.add_argument("--linear_verbalizer", action="store_true")

    num_classes, batch_size = get_task_config("config.yaml", args.downstream)

    speech_prompt_args = [
        f"--prompt_length={args.prompt_length}",
        f"--num_classes={num_classes}",  # only meaningful when using --linear_verbalizer
    ]
    speech_prompt_args += ["--deep_prompt"] if args.deep_prompt else []
    speech_prompt_args += ["--linear_verbalizer"] if args.linear_verbalizer else []

    # ====================== #
    #   Fairseq-train args   #
    # ====================== #
    # For Fairseq-train args, please refer to the official documentation.
    # https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train

    bin_data_dir = storage_dir / "s2u_data" / args.downstream / f"{args.vb_method}_data_bin"
    restore_file = storage_dir / "pretrained_models" / "GSLM" / "checkpoint_best.pt"
    tensorboard_logdir = storage_dir / "exp_results" / args.downstream / args.exp_name / "logs"
    save_dir = storage_dir / "exp_results" / args.downstream / args.exp_name / "checkpoints"
    path_args = [
        str(bin_data_dir),
        f"--restore-file={restore_file}",
        f"--tensorboard-logdir={tensorboard_logdir}",
        f"--save-dir={save_dir}",
    ]

    task_args = [
        "--user-dir=./fairseq_usr",
        "--task=prompt_language_modeling",
        "--arch=GSLM_SpeechPrompt_v1",
        "--criterion=cross_entropy_prompt",
        "--share-decoder-input-output-embed",
        "--sample-break-mode=eos",
    ]

    training_args = [
        "--reset-optimizer",
        f"--batch-size={batch_size}",
        "--optimizer=adam",
        "--adam-betas=(0.9, 0.98)",
        "--clip-norm=1.0",
        "--update-freq=1",
        "--max-tokens=4096",
        "--num-workers=10",
        "--skip-invalid-size-inputs-valid-test",
        "--patience=1",
        "--max-epoch=300",
        "--log-interval=10",
        "--seed=100501",
        "--fp16",
        "--dropout=0",
        "--attention-dropout=0",
        "--save-interval=1",
        "--lr=5e-3",
    ]

    fairseq_args = path_args + task_args + training_args + speech_prompt_args
    print(fairseq_args)
    return fairseq_parser, fairseq_args


def get_input_args():
    # ========== #
    #   Parser   #
    # ========== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream", type=str, default="")
    parser.add_argument("--vb_method", choices=["random", "freq", "learnable"])
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--prompt_length", type=int, default=5)
    parser.add_argument("--deep_prompt", action="store_true", default=True)
    parser.add_argument("--linear_verbalizer", action="store_true")
    args = parser.parse_args()

    fairseq_parser, fairseq_args = convert_to_fairseq_args(args)

    return fairseq_parser, fairseq_args


def main():
    """
    e.g.
    python train.py \
        --downstream SCR_lt_speech_commands \
        --vb_method freq \
        --exp_name SCR_lt_speech_commands_plen.5 \
        --prompt_length 5 \
        --deep_prompt
    """
    fairseq_parser, fairseq_args = get_input_args()
    cli_main(fairseq_parser, fairseq_args)


if __name__ == "__main__":
    main()
