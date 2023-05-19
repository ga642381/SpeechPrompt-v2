import yaml
from pathlib import Path


def get_storage_dir():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return Path(config["storage_dir"]).resolve()
