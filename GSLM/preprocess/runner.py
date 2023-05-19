import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path
import git
from pprint import pprint
import IPython
import yaml
from model_downloader import ModelDownloader, GSLM_MODELS


def main(args):
    ##########
    # Config #
    ##########
    with open(f"./config.yaml", "r") as f:
        try:
            global_config = yaml.safe_load(f)[args.model]
        except yaml.YAMLError as exc:
            print(exc)

    with open(f"{args.downstream}/config.yaml", "r") as f:
        try:
            dataset_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    ######################
    # Pre-trained Models #
    ######################
    assert (
        global_config["fairseqrc"]["download_models"] is False and "ssl_model_path" not in global_config["fairseqrc"],
        "[ERROR] You need to download the pre-trained models first, or specify the path to the pre-trained models in the config file.",
    )

    # model_dir = get_model_dir(args.model)
    model_dir = Path(global_config["fairseqrc"]["download_model_dir"])
    if global_config["fairseqrc"]["download_models"]:
        if args.model == "GSLM":
            downloader = ModelDownloader(GSLM_MODELS)
            downloader.download_models(model_dir)
            global_config["fairseqrc"]["ssl_model_path"] = (
                model_dir / "HuBERT" / Path(GSLM_MODELS["HuBERT_MODEL"]).name
            )
            global_config["fairseqrc"]["km_model_path"] = (
                model_dir / "HuBERT" / Path(GSLM_MODELS["HuBERT_KMEANS"]).name
            )
        else:
            raise NotImplementedError(f"[ERROR] The model {args.model} is not supported.")

    ##############
    # Preprocess #
    ##############
    print(f"[CONFIG] Model: {args.model}")
    print("[CONFIG] Global Config:")
    pprint(global_config)
    print("[CONFIG] Dataset Config:")
    pprint(dataset_config)
    preprocessor_module = SourceFileLoader("preprocessor", f"{args.downstream}/preprocessor.py").load_module()
    preprocessor = preprocessor_module.Preprocessor(global_config, dataset_config)
    eval(f"preprocessor.{args.action}")()


if __name__ == "__main__":
    # There are 4 stages to preprocess the data: (1)generate_manifest, (2)quantize, (3)reduce_quantized, (4)create_lm_dataset.
    # You can run through these 4 stages sequentially by the following command:
    """example usage
    python runner.py --model GSLM --downstream SCR_google_speech_commands --action generate_manifest
    python runner.py --model GSLM --downstream SCR_google_speech_commands --action quantize
    python runner.py --model GSLM --downstream SCR_google_speech_commands --action reduce_quantized
    python runner.py --model GSLM --downstream SCR_google_speech_commands --action create_lm_dataset
    """

    # Or you can run --action all to run through all the 4 stages.:
    """
    python runner.py --model GSLM --downstream SCR_google_speech_commands --action
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="which backbone spoken language model", default="GSLM", choices=["GSLM"])
    parser.add_argument(
        "--downstream", help="which downstream task to preprocess", default="SCR_google_speech_commands"
    )
    parser.add_argument("--action", help="which action to run", default="all")

    args = parser.parse_args()
    main(args)
