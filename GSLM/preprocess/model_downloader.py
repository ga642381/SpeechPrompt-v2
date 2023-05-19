import requests
from tqdm import tqdm
from pathlib import Path
import os
import git
import tarfile

GSLM_MODELS = {
    "GSLM_MODEL": "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz",
    "HuBERT_MODEL": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
    "HuBERT_KMEANS": "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin",
}


def extract_GSLM(tar_file, dest_dir):
    tar = tarfile.open(tar_file)
    for member in tar.getmembers():
        if member.isreg():
            member.name = os.path.basename(member.name)  # remove the path by reset it
            tar.extract(member, dest_dir)


def get_model_dir(model):
    repo = git.Repo(search_parent_directories=True)
    repo_dir = Path(repo.working_dir)  # /.../SpeechPrompt-dev
    # download_dir = repo_dir / model / "prompt" / "storage" / "pretrained_models"
    download_dir = repo_dir / "storage" / model / "pretrained_models"
    return download_dir


class ModelDownloader:
    def __init__(self, models):
        self.models = models

    def download_models(self, download_dir):
        for model_name, model_url in self.models.items():
            model_path = Path(download_dir) / model_name.split("_")[0] / Path(model_url).name
            if not model_path.exists():
                print(f"[INFO] Downloading {model_name}...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.download_file(model_url, model_path)
                if model_name == "GSLM_MODEL":
                    print(f"[INFO] Extracting {model_name}...")
                    extract_GSLM(model_path, dest_dir=model_path.parent)
            else:
                print(f"[INFO] {model_name} already exists, skipping...")

    def download_file(self, url, file_path):
        with requests.get(url, stream=True) as r:
            total_length = int(r.headers.get("content-length"))
            with open(file_path, "wb") as f:
                for chunk in self.get_chunks(r, total_length):
                    f.write(chunk)

    def get_chunks(self, response, total_length, chunk_size=1024):
        progress_bar = tqdm(total=total_length, unit="B", unit_scale=True)
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
                progress_bar.update(len(chunk))
        progress_bar.close()


if __name__ == "__main__":
    download_dir = get_model_dir("GSLM")
    downloader = ModelDownloader(GSLM_MODELS)
    downloader.download_models(download_dir)
