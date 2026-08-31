import gzip
import math
import os
import bisect  # Added for O(log N) file lookup
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

RAW_DATASET_FOLDER = "dataset/raw"
OMIT_LANGUAGE_CHECK_FILEPATTERNS = ["cosmopedia"]

MINIMUM_CHUNK_SIZE = 1024
MAXIMUM_CHUNK_SIZE = 3 * 1024
MAX_CORPUS_SIZE = 200 * 1024 ** 2

_pretrain_dataset_folders = sorted(os.listdir(RAW_DATASET_FOLDER))

# Small LRU of parquet dataframes. The stream now interleaves the sources
# (cosmopedia / fineweb / fineweb-edu), so at any moment one file per source is
# hot instead of one file total. Each source is still walked in row order, so
# this only needs to hold the ~3 currently-active files.
_DF_CACHE_MAX = int(os.environ.get("PRETRAIN_DF_CACHE", "4"))
_df_cache = {}


def _get_df(filepath, cols):
    df = _df_cache.pop(filepath, None)
    if df is None:
        df = pd.read_parquet(filepath, columns=cols)
    _df_cache[filepath] = df                      # move/insert as most-recent
    while len(_df_cache) > _DF_CACHE_MAX:
        _df_cache.pop(next(iter(_df_cache)))      # evict least-recently-used
    return df


_all_files = []
for folder in _pretrain_dataset_folders:
    folder_path = os.path.join(RAW_DATASET_FOLDER, folder)
    if os.path.isdir(folder_path): 
        _all_files.extend(sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")))

# ==============================================================================
# OPTIMIZATION: Precompute cumulative row counts for O(log N) random access
# ==============================================================================
_file_metadata = []
_cumulative_rows = []
_current_cumulative = 0

for filepath in _all_files:
    try:
        rows = pq.ParquetFile(filepath).metadata.num_rows
    except Exception:
        rows = 0
    _file_metadata.append((filepath, rows))
    _current_cumulative += rows
    _cumulative_rows.append(_current_cumulative)
# ==============================================================================

def load_processed_chunks(
        index,
        n=10,
        min_chunk_size=MINIMUM_CHUNK_SIZE,
        max_chunk_size=MAXIMUM_CHUNK_SIZE, dataset_folder=None):

    # Fallback to original logic if a specific dataset_folder is requested
    if dataset_folder is not None:
        files = sorted(os.path.join(RAW_DATASET_FOLDER, dataset_folder, f) for f in os.listdir(os.path.join(RAW_DATASET_FOLDER, dataset_folder)) if f.endswith(".parquet"))
        start, end = index * n, index * n + n
        for filepath in files:
            omit_language_check = any(pattern in filepath for pattern in OMIT_LANGUAGE_CHECK_FILEPATTERNS)
            cols_to_read = ["text"] if omit_language_check else ["text", "language"]
            df = _get_df(filepath, cols_to_read)

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

    # === OPTIMIZED PATH: O(log F) file lookup instead of O(F) sequential scan ===
    target_global_start = index * n
    target_global_end = target_global_start + n
    
    # Find the exact file containing the target_global_start using binary search
    file_idx = bisect.bisect_right(_cumulative_rows, target_global_start)
    
    if file_idx >= len(_file_metadata):
        return []
        
    filepath, file_rows = _file_metadata[file_idx]
    prev_cumulative = _cumulative_rows[file_idx - 1] if file_idx > 0 else 0
    
    local_start = target_global_start - prev_cumulative
    local_end = local_start + n
    
    omit_language_check = any(pattern in filepath for pattern in OMIT_LANGUAGE_CHECK_FILEPATTERNS)
    cols_to_read = ["text"] if omit_language_check else ["text", "language"]

    df = _get_df(filepath, cols_to_read)

    if local_start >= len(df):
        return []
        
    rows = df.iloc[local_start:local_end]

    if omit_language_check:
        texts = rows["text"].to_numpy()
    else:
        texts = rows.loc[rows["language"] == "en", "text"].to_numpy()

    chunks = []
    for text in texts:
        if not isinstance(text, str): continue
        text = text.encode("ascii", errors="ignore").decode("ascii")
        if len(text) <= min_chunk_size: continue
        if len(text) <= max_chunk_size: 
            chunks.append(text)
            continue

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


def _folder_row_bounds():
    """[(folder, cumulative_row_end), ...] over _all_files, in file order."""
    bounds, cum, cur = [], 0, None
    for filepath, rows in _file_metadata:
        folder = os.path.basename(os.path.dirname(filepath))
        if cur is not None and folder != cur:
            bounds.append((cur, cum))
        cur, cum = folder, cum + rows
    if cur is not None:
        bounds.append((cur, cum))
    return bounds


def _interleave_order(n_pretrain_batches, n_wiki_batches, batch_size):
    """Permutation of range(n_pretrain_batches + n_wiki_batches) that rotates
    through the sources (cosmopedia / fineweb / fineweb-edu / wikipedia) instead
    of running each one to exhaustion before the next -- which is what made the
    loss collapse partway through pretraining when the stream crossed from one
    source into the next.

    Blocks stay in ascending order *within* each source, so every source is
    still read sequentially from its parquet files (no random-access cost), and
    the schedule is proportional to size so all sources finish together.
    """
    groups, prev = [], 0
    for _folder, row_end in _folder_row_bounds():
        end = min(n_pretrain_batches, math.ceil(row_end / batch_size))
        if end > prev:
            groups.append(list(range(prev, end)))
        prev = end
    if n_wiki_batches:
        groups.append(list(range(n_pretrain_batches,
                                 n_pretrain_batches + n_wiki_batches)))

    if len(groups) <= 1:
        return [i for g in groups for i in g]

    weights = [len(g) for g in groups]
    total = sum(weights)
    cursors = [0] * len(groups)
    order = []
    for _ in range(total):
        # virtual-time / stride scheduling: serve whichever source has taken the
        # smallest fraction of its blocks so far. Keeps every source's density
        # constant across the whole permutation, so they all finish together.
        j = min((k for k in range(len(groups)) if cursors[k] < len(groups[k])),
                key=lambda k: (cursors[k] + 0.5) / weights[k])
        order.append(groups[j][cursors[j]])
        cursors[j] += 1
    return order


class PretrainTextDataset(Dataset):
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

        # Rotate through the sources instead of serving them in blocks.
        self._order = _interleave_order(
            self._num_pretrain_batches, self._num_wiki_batches, batch_size)

    def __len__(self):
        return self._num_pretrain_batches + self._num_wiki_batches

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        idx = self._order[idx]

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
