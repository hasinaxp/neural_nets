import gzip
import math
import os
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

RAW_DATASET_FOLDER = "dataset/raw"
OMIT_LANGUAGE_CHECK_FILEPATTERNS = ["cosmopedia"]

MINIMUM_CHUNK_SIZE = 1024
MAXIMUM_CHUNK_SIZE = 3 * 1024
MAX_CORPUS_SIZE = 200 * 1024 ** 2


_pretrain_dataset_folders = sorted(os.listdir(RAW_DATASET_FOLDER))
_pret_data_cached_file = None
_pret_data_cached_df = None

_all_files = []
for folder in _pretrain_dataset_folders:
    folder_path = os.path.join(RAW_DATASET_FOLDER, folder)
    if os.path.isdir(folder_path): _all_files.extend(sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")))

def load_processed_chunks(
        index,
        n=10,
        min_chunk_size=MINIMUM_CHUNK_SIZE,
        max_chunk_size=MAXIMUM_CHUNK_SIZE, dataset_folder=None):
    global _pret_data_cached_file, _pret_data_cached_df
    files = _all_files if dataset_folder is None else sorted(os.path.join(RAW_DATASET_FOLDER, dataset_folder, f) for f in os.listdir(os.path.join(RAW_DATASET_FOLDER, dataset_folder)) if f.endswith(".parquet"))
    start, end = index * n, index * n + n

    for filepath in files:
        omit_language_check = any(pattern in filepath for pattern in OMIT_LANGUAGE_CHECK_FILEPATTERNS)

        if _pret_data_cached_file == filepath:
            df = _pret_data_cached_df
        else:
            cols_to_read = ["text"] if omit_language_check else ["text", "language"]
            df = pd.read_parquet(filepath, columns=cols_to_read)
            _pret_data_cached_file, _pret_data_cached_df = filepath, df

        if start >= (df_len := len(df)):
            start -= df_len
            end -= df_len
            continue

        rows = df.iloc[start:end]

        if omit_language_check:
            texts = rows["text"].to_numpy()
        else:
            texts = rows.loc[rows["language"] == "en", "text"].to_numpy()

        chunks = []

        for text in texts:
            if not isinstance(text, str): continue
            text = text.encode("ascii", errors="ignore").decode("ascii")
            if len(text) <= min_chunk_size: continue
            if len(text) <= max_chunk_size: chunks.append(text); continue

            while len(text) > max_chunk_size:
                split_at = text.rfind("\n\n", 0, max_chunk_size + 1)
                if split_at <= 0:
                    split_at = text.rfind(".\n", 0, max_chunk_size + 1)
                    if split_at > 0: split_at += 1
                if split_at <= 0: split_at = max_chunk_size

                left, text = text[:split_at].strip(), text[split_at:].strip()
                if len(left) > min_chunk_size: chunks.append(left)

            if len(text) > min_chunk_size: chunks.append(text)

        return chunks
    return []


def load_prcessed_wikipedia_chunks(chunk_size=MINIMUM_CHUNK_SIZE):
    path = "dataset/wikipedia.txt.gz"
    if not os.path.exists(path):
        path = "dataset/wikipedia.zip"

    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("wikipedia.txt") as f:
                text = f.read().decode("utf-8")

    chunks, buf = [], ""

    for part in text.split("\n\n"):
        for s in part.split("\n"):
            for s in s.split("."):
                s = s.strip()
                if not s:
                    continue

                while len(s) > 4 * 1024:
                    chunks.append(s[:4 * 1024])
                    s = s[4 * 1024:]

                if len(buf) + len(s) + 1 > 4 * 1024:
                    chunks.append(buf)
                    buf = ""

                buf += (" " if buf else "") + s

                if len(buf) >= chunk_size:
                    chunks.append(buf)
                    buf = ""

    if buf:
        chunks.append(buf)

    return chunks


class PretrainTextDataset(Dataset):
    """Combines the parquet-backed pretrain corpus and the wikipedia corpus,
    yielding fixed-size batches of raw text chunks."""

    def __init__(self, batch_size=10, min_chunk_size=MINIMUM_CHUNK_SIZE,
                 max_chunk_size=MAXIMUM_CHUNK_SIZE, dataset_folder=None,
                 include_wikipedia=True, wikipedia_chunk_size=MINIMUM_CHUNK_SIZE):
        self.batch_size = batch_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.dataset_folder = dataset_folder

        files = _all_files if dataset_folder is None else sorted(
            os.path.join(RAW_DATASET_FOLDER, dataset_folder, f)
            for f in os.listdir(os.path.join(RAW_DATASET_FOLDER, dataset_folder))
            if f.endswith(".parquet")
        )
        self._num_pretrain_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        self._num_pretrain_batches = math.ceil(self._num_pretrain_rows / batch_size)

        self.wiki_chunks = load_prcessed_wikipedia_chunks(wikipedia_chunk_size) if include_wikipedia else []
        self._num_wiki_batches = math.ceil(len(self.wiki_chunks) / batch_size) if self.wiki_chunks else 0

    def __len__(self):
        return self._num_pretrain_batches + self._num_wiki_batches

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if idx < self._num_pretrain_batches:
            return load_processed_chunks(
                idx, n=self.batch_size,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size,
                dataset_folder=self.dataset_folder,
            )

        wiki_idx = idx - self._num_pretrain_batches
        start = wiki_idx * self.batch_size
        return self.wiki_chunks[start:start + self.batch_size]


if __name__ == "__main__":
    for c in load_processed_chunks(0, 2): print(c, "\n")
    print("\n======\nchunks2\n======\n")
    for c in load_processed_chunks(10000, 2): print(c, "\n")

    wiki_chunks = load_prcessed_wikipedia_chunks()
    print(len(wiki_chunks))

    ds = PretrainTextDataset(batch_size=8)
    print(len(ds), "batches")
    print(ds[0][0])