"""Tokenise the raw corpus into binary shards, once.

This is the step that makes the training loop cheap. The BPE implementation in
``nanollm.tokenizer`` is pure Python; running it inside the training loop caps
throughput well below what the GPU can consume. Here it runs once across a
process pool, and training reads uint16 memmaps forever after.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Iterator, Optional

import numpy as np

from ..tokenizer import Tokenizer
from .shards import ShardWriter, free_bytes
from .sources import iter_all_documents

__all__ = ["prepare_shards", "encode_documents"]

# Set once per worker process; a Tokenizer is a few MB of merge tables and
# re-loading it per document would dominate the run.
_WORKER_TOKENIZER: Optional[Tokenizer] = None
_WORKER_EOS: int = 0


def _init_worker(tokenizer_file: str, vocab_size: int, eos_id: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOS
    tok = Tokenizer(vocab_size=vocab_size)
    tok.load(tokenizer_file)
    _WORKER_TOKENIZER = tok
    _WORKER_EOS = eos_id


def _encode_one(doc: str) -> np.ndarray:
    ids = _WORKER_TOKENIZER.encode(doc)
    ids.append(_WORKER_EOS)
    return np.asarray(ids, dtype=np.int32)


def encode_documents(
    documents: Iterator[tuple[str, str]],
    tokenizer_file: str,
    vocab_size: int,
    eos_id: int,
    num_workers: int = 0,
    chunksize: int = 32,
) -> Iterator[np.ndarray]:
    """Yield one token array per document, in stream order."""
    if num_workers <= 0:
        _init_worker(tokenizer_file, vocab_size, eos_id)
        for _source, doc in documents:
            yield _encode_one(doc)
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_file, vocab_size, eos_id),
    ) as pool:
        # imap keeps ordering and bounds memory; the generator is consumed
        # lazily so the whole corpus never sits in RAM.
        texts = (doc for _source, doc in documents)
        yield from pool.imap(_encode_one, texts, chunksize=chunksize)


def prepare_shards(
    raw_dir: str,
    out_dir: str,
    tokenizer_file: str,
    vocab_size: int,
    eos_id: int,
    shard_tokens: int = 100_000_000,
    num_workers: int = 0,
    max_tokens: Optional[int] = None,
    include_wikipedia: bool = True,
    progress: bool = True,
    min_free_gb: float = 5.0,
    log=print,
) -> "ShardIndex":
    """Full pipeline: raw text -> interleaved documents -> shards + index.

    Stops cleanly when free space drops below ``min_free_gb`` rather than
    letting the filesystem fill. A run that stops this way still writes a
    valid index and the shards it produced are usable -- the alternative is
    an ENOSPC deep inside a buffered write, which leaves the last shard
    headerless and the index missing.
    """
    documents = iter_all_documents(
        raw_dir=raw_dir, include_wikipedia=include_wikipedia)

    writer = ShardWriter(out_dir, "tokens", vocab_size, shard_tokens=shard_tokens)
    started = time.time()
    n_docs = 0

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=max_tokens, desc="tokenising", unit="tok",
                       unit_scale=True)
        except ImportError:
            bar = None

    min_free_bytes = int(min_free_gb * 1024 ** 3)
    stopped_early = None

    try:
        for tokens in encode_documents(
            documents, tokenizer_file, vocab_size, eos_id, num_workers
        ):
            writer.add(tokens)
            n_docs += 1

            # Checked every ~4096 documents: a statvfs per document would
            # cost more than the tokenising.
            if n_docs % 4096 == 0 and free_bytes(out_dir) < min_free_bytes:
                stopped_early = (
                    f"only {free_bytes(out_dir)/1024**3:.1f}GB free "
                    f"(floor is {min_free_gb}GB)")
                log(f"  stopping: {stopped_early}")
                break
            if bar is not None:
                bar.update(int(tokens.size))
            elif progress and n_docs % 20000 == 0:
                rate = writer.total_tokens / max(1e-6, time.time() - started)
                log(f"  {n_docs:,} docs | {writer.total_tokens/1e6:.1f}M tokens "
                    f"| {rate/1e3:.0f}k tok/s")
            if max_tokens is not None and writer.total_tokens >= max_tokens:
                log(f"  reached max_tokens={max_tokens:,}, stopping")
                break
    finally:
        if bar is not None:
            bar.close()
        index = writer.close()

    index.meta = {
        "documents": n_docs,
        "stopped_early": stopped_early,
        "raw_dir": raw_dir,
        "tokenizer_file": tokenizer_file,
        "eos_id": eos_id,
        "seconds": round(time.time() - started, 1),
    }
    index.save()

    elapsed = time.time() - started
    log(f"  wrote {len(index.shards)} shards | {index.total_tokens:,} tokens "
        f"from {n_docs:,} documents in {elapsed/60:.1f} min "
        f"({index.total_tokens/max(1e-6, elapsed)/1e3:.0f}k tok/s)")
    if stopped_early:
        log(f"  NOTE: stopped early ({stopped_early}). The shards written are "
            f"valid and usable; free space and re-run to extend the corpus.")
    return index
