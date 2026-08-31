"""Raw text sources for pretraining.

Everything here yields plain ``str`` documents. Tokenising happens once, in
``prepare.py``; nothing in this module is on the training hot path.
"""

from __future__ import annotations

import gzip
import math
import os
from typing import Iterator, Optional

RAW_DATASET_FOLDER = "dataset/raw"
OMIT_LANGUAGE_CHECK_FILEPATTERNS = ["cosmopedia"]

MINIMUM_CHUNK_SIZE = 1024
MAXIMUM_CHUNK_SIZE = 3 * 1024

__all__ = [
    "list_parquet_files", "iter_parquet_documents", "iter_wikipedia_documents",
    "iter_all_documents", "split_long_text", "source_of",
]


def list_parquet_files(raw_dir: str = RAW_DATASET_FOLDER) -> list[str]:
    """All parquet files under ``raw_dir``, grouped by source folder, sorted."""
    files: list[str] = []
    if not os.path.isdir(raw_dir):
        return files
    for folder in sorted(os.listdir(raw_dir)):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files.extend(sorted(
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith(".parquet")
        ))
    return files


def source_of(filepath: str) -> str:
    return os.path.basename(os.path.dirname(filepath))


def split_long_text(text: str,
                    min_chunk_size: int = MINIMUM_CHUNK_SIZE,
                    max_chunk_size: int = MAXIMUM_CHUNK_SIZE) -> list[str]:
    """Split on paragraph boundaries where possible, sentence ends otherwise.

    Documents shorter than ``min_chunk_size`` are dropped: they are mostly
    boilerplate and navigation chrome in web crawl data.
    """
    text = text.encode("ascii", errors="ignore").decode("ascii")
    if len(text) <= min_chunk_size:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    while len(text) > max_chunk_size:
        split_at = text.rfind("\n\n", 0, max_chunk_size + 1)
        if split_at <= 0:
            split_at = text.rfind(".\n", 0, max_chunk_size + 1)
            if split_at > 0:
                split_at += 1
        if split_at <= 0:
            split_at = max_chunk_size
        left, text = text[:split_at].strip(), text[split_at:].strip()
        if len(left) > min_chunk_size:
            chunks.append(left)
    if len(text) > min_chunk_size:
        chunks.append(text)
    return chunks


def iter_parquet_documents(
    filepath: str,
    min_chunk_size: int = MINIMUM_CHUNK_SIZE,
    max_chunk_size: int = MAXIMUM_CHUNK_SIZE,
    row_group_batch: int = 4096,
) -> Iterator[str]:
    """Stream one parquet file in row-group batches.

    Reading in batches rather than ``pd.read_parquet`` on the whole file keeps
    peak RSS flat regardless of file size -- these are 2GB+ files.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(filepath)
    available = set(pf.schema_arrow.names)
    check_language = (
        "language" in available
        and not any(p in filepath for p in OMIT_LANGUAGE_CHECK_FILEPATTERNS)
    )
    columns = ["text", "language"] if check_language else ["text"]

    for batch in pf.iter_batches(batch_size=row_group_batch, columns=columns):
        texts = batch.column("text").to_pylist()
        if check_language:
            langs = batch.column("language").to_pylist()
            texts = [t for t, lang in zip(texts, langs) if lang == "en"]
        for text in texts:
            if not isinstance(text, str):
                continue
            yield from split_long_text(text, min_chunk_size, max_chunk_size)


def iter_wikipedia_documents(
    path: str = "dataset/wikipedia.txt.gz",
    chunk_size: int = MINIMUM_CHUNK_SIZE,
    max_chunk_size: int = 4 * 1024,
) -> Iterator[str]:
    """Stream the bundled wikipedia dump line by line."""
    if not os.path.exists(path):
        alt = path.replace(".txt.gz", ".zip")
        if not os.path.exists(alt):
            return
        import zipfile
        with zipfile.ZipFile(alt) as zf:
            with zf.open("wikipedia.txt") as f:
                lines = f.read().decode("utf-8").splitlines(keepends=True)
        stream: Iterator[str] = iter(lines)
    else:
        stream = gzip.open(path, "rt", encoding="utf-8", errors="ignore")

    buf = ""
    try:
        for line in stream:
            for sentence in line.split("."):
                s = sentence.strip()
                if not s:
                    continue
                while len(s) > max_chunk_size:
                    yield s[:max_chunk_size]
                    s = s[max_chunk_size:]
                if len(buf) + len(s) + 1 > max_chunk_size:
                    yield buf
                    buf = ""
                buf += (" " if buf else "") + s
                if len(buf) >= chunk_size:
                    yield buf
                    buf = ""
    finally:
        if hasattr(stream, "close"):
            stream.close()
    if buf:
        yield buf


def iter_all_documents(
    raw_dir: str = RAW_DATASET_FOLDER,
    include_wikipedia: bool = True,
    wikipedia_path: str = "dataset/wikipedia.txt.gz",
    min_chunk_size: int = MINIMUM_CHUNK_SIZE,
    max_chunk_size: int = MAXIMUM_CHUNK_SIZE,
    interleave: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield ``(source, document)`` across every corpus.

    With ``interleave``, sources are round-robined by size rather than run to
    exhaustion one after another. Reading them in blocks makes the loss step
    every time the stream crosses a corpus boundary -- the model sees a sudden
    distribution shift with no examples of the old distribution left to anchor
    it. Interleaving keeps each source at constant density throughout.
    """
    streams: list[tuple[str, Iterator[str], float]] = []
    for filepath in list_parquet_files(raw_dir):
        size = os.path.getsize(filepath)
        streams.append((
            source_of(filepath),
            iter_parquet_documents(filepath, min_chunk_size, max_chunk_size),
            float(size),
        ))
    if include_wikipedia and os.path.exists(wikipedia_path):
        streams.append((
            "wikipedia",
            iter_wikipedia_documents(wikipedia_path, min_chunk_size),
            float(os.path.getsize(wikipedia_path)) * 4,   # gz expands ~4x
        ))

    if not streams:
        return

    if not interleave:
        for name, stream, _ in streams:
            for doc in stream:
                yield name, doc
        return

    # Virtual-time scheduling: always serve whichever stream has emitted the
    # smallest fraction of its expected share. Streams that run dry drop out.
    total = sum(w for _, _, w in streams) or 1.0
    served = [0.0] * len(streams)
    live = list(range(len(streams)))
    while live:
        i = min(live, key=lambda k: served[k] / (streams[k][2] / total))
        name, stream, _ = streams[i]
        doc = next(stream, None)
        if doc is None:
            live.remove(i)
            continue
        served[i] += len(doc)
        yield name, doc
