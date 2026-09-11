"""
Reproducible dataset setup for the Food Recommendation Engine.

Datasets live outside git (see ../.gitignore). Run this once after cloning to
rebuild the datasets/ folder. Idempotent — existing files are skipped.

    python datasets/download_datasets.py

Auto-downloaded:
  - RIRD            (GitHub:  D3Mlab/rir_data)          -> datasets/rir_data-main/
  - M-ABSA (Thai)   (HuggingFace: Multilingual-NLP/M-ABSA) -> datasets/m-absa/
  - Wongnai         (HuggingFace: Highgroundbkk/wongnai_reviews) -> datasets/wongnai/

Manual (needs a Kaggle account — see README "Dataset Setup"):
  - Thailand Foods  (Kaggle: ponthakornsodchun/foods-in-thailand) -> datasets/thailand_foods.csv
"""
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # datasets/


def log(msg):
    print(f"[setup] {msg}")


def download_rird():
    dest = HERE / "rir_data-main"
    if (dest / "PMD.csv").exists():
        log("RIRD already present — skip")
        return
    url = "https://github.com/D3Mlab/rir_data/archive/refs/heads/main.zip"
    log(f"downloading RIRD from {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(HERE)
    log(f"RIRD -> {dest}")


def download_mabsa():
    dest = HERE / "m-absa"
    if (dest / "restaurant" / "th" / "train.txt").exists():
        log("M-ABSA (Thai) already present — skip")
        return
    from huggingface_hub import HfApi, hf_hub_download
    repo = "Multilingual-NLP/M-ABSA"
    files = [s.rfilename for s in HfApi().dataset_info(repo).siblings]
    want = [f for f in files if f.count("/") == 2 and f.split("/")[1] == "th"]
    want += ["README.md"]
    log(f"downloading {len(want)} M-ABSA Thai files")
    for f in want:
        p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset")
        out = dest / f
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, out)
    log(f"M-ABSA -> {dest}")


def download_wongnai():
    dest = HERE / "wongnai"
    if (dest / "train.parquet").exists():
        log("Wongnai already present — skip")
        return
    from datasets import load_dataset
    dest.mkdir(parents=True, exist_ok=True)
    log("downloading Wongnai from HuggingFace (Highgroundbkk/wongnai_reviews)")
    ds = load_dataset("Highgroundbkk/wongnai_reviews")
    for split in ds:
        ds[split].to_pandas().to_parquet(dest / f"{split}.parquet", index=False)
    (dest / "SOURCE.txt").write_text(
        "Wongnai reviews\n"
        "Source: HuggingFace 'Highgroundbkk/wongnai_reviews'\n"
        "Columns: review_body (Thai text), star_rating (0-4)\n"
        "Splits: train=40000, test=6203\n",
        encoding="utf-8",
    )
    log(f"Wongnai -> {dest}")


def check_thailand_foods():
    p = HERE / "thailand_foods.csv"
    if p.exists():
        log("Thailand Foods present")
    else:
        log("MISSING thailand_foods.csv — download manually from Kaggle "
            "(ponthakornsodchun/foods-in-thailand) and place it at datasets/thailand_foods.csv")


def main():
    download_rird()
    download_mabsa()
    download_wongnai()
    check_thailand_foods()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
