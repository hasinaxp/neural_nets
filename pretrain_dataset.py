import os
import zipfile
import pandas as pd

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
    with zipfile.ZipFile("dataset/wikipedia.zip", "r") as zf:
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



if __name__ == "__main__":
    for c in load_processed_chunks(0, 2): print(c, "\n")
    print("\n======\nchunks2\n======\n")
    for c in load_processed_chunks(10000, 2): print(c, "\n")

    wiki_chunks = load_prcessed_wikipedia_chunks()
    print(len(wiki_chunks))