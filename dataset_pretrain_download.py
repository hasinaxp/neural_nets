import os

from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm


DATASET_FOLDER = "dataset/raw"
REPO_TYPE = "dataset"

DEFAULT_DATA_FILE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_COSMOPEDIA_FILE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

REPOS = {
    "fineweb": {
        "id": "HuggingFaceFW/fineweb",
        "indices": DEFAULT_DATA_FILE_INDICES,
    },
    "fineweb-edu": {
        "id": "HuggingFaceFW/fineweb-edu",
        "indices": DEFAULT_DATA_FILE_INDICES,
    },
    "cosmopedia": {
        "id": "HuggingFaceTB/cosmopedia",
        "indices": DEFAULT_COSMOPEDIA_FILE_INDICES,
    },
}


def prepare_folders():
    for key in REPOS:
        os.makedirs(f"{DATASET_FOLDER}/{key}", exist_ok=True)


def download_dataset(key):
    repo = REPOS[key]
    print(f"Listing {key} parquet files...")
    all_files = [
        f
        for f in list_repo_files(repo["id"], repo_type=REPO_TYPE)
        if f.endswith(".parquet")
    ]

    print(f"Found {len(all_files)} parquet files total.\n")

    files_to_download = [
        all_files[i]
        for i in repo["indices"]
        if i < len(all_files)
    ]

    for filepath in tqdm(files_to_download, desc=f"Downloading {key}"):
        downloaded_path = hf_hub_download(
            repo_id=repo["id"],
            filename=filepath,
            repo_type=REPO_TYPE,
        )

        dest_path = (
            f"{DATASET_FOLDER}/{key}/{os.path.basename(filepath)}"
        )

        with open(downloaded_path, "rb") as src, open(dest_path, "wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)


def main():
    prepare_folders()

    for key in REPOS:
        print(f"\n--- Downloading dataset: {key} ---")
        download_dataset(key)


if __name__ == "__main__":
    main()
