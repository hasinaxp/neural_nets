import os
import requests
from huggingface_hub import list_repo_files, hf_hub_url
import pandas as pd
from tqdm import tqdm

REPO_ID = "HuggingFaceFW/fineweb"
REPO_TYPE = "dataset"
OUTPUT_DIR = "dataset/fineweb/raw"
TXT_OUTPUT_DIR = "dataset/fineweb/txt"
TARGET_TOTAL_BYTES = 2 * 1024 ** 10
DATA_PER_FILE   = 200 * 1024 ** 2
MINIMUM_CHUNK_LEN = 1024
DOWNLOAD_CHUNK_SIZE = 16 * 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

def download_file(url: str, dest_path: str):
    """Stream-download a file in 8 KB blocks."""
    print(f"Downloading -> {dest_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for block in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(block)

def download_and_process_fineweb():
    print("Listing FineWeb parquet files …")
    all_files = [
        f for f in list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        if f.endswith(".parquet")
    ]
    print(f"Found {len(all_files)} parquet files total.\n")

    downloaded_bytes = 0
    data_buffer = ""
    file_count = 0

    for remote_path in all_files:
        if downloaded_bytes >= TARGET_TOTAL_BYTES:
            print(f"\nReached 4 GB target ({downloaded_bytes / 1024**3:.2f} GB). Done.")
            break

        filename  = os.path.basename(remote_path)
        dest_path = f"{OUTPUT_DIR}/{filename}"

        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            print(f"Skipping (already exists, {size / 1024**2:.1f} MB): {filename}")
            downloaded_bytes += size
            continue

        url = hf_hub_url(REPO_ID, filename=remote_path, repo_type=REPO_TYPE)

        try:
            download_file(url, dest_path)
            size = os.path.getsize(dest_path)
            downloaded_bytes += size
            print(f"  {filename}  ({size / 1024**2:.1f} MB)  |  "
                  f"Total so far: {downloaded_bytes / 1024**3:.2f} GB")

            print(f"pre-processing file: {filename}")
            df = pd.read_parquet(dest_path, columns=['text', 'language'])
            df = df[df['language'] == 'en']
            for _, row in tqdm(df.iterrows()):
                text = row['text'].encode('ascii', errors="ignore").decode('ascii')
                if len(text) < MINIMUM_CHUNK_LEN:
                    continue
                data_buffer += text
                data_buffer += "\n\n\n\n"
                if len(data_buffer) > DATA_PER_FILE:
                    with open(f"{TXT_OUTPUT_DIR}/corpus_{file_count}.txt", 'w') as f:
                        f.write(data_buffer)
                    file_count += 1
                    data_buffer = ""
            
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_and_process_fineweb()
