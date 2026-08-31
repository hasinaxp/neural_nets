"""Binary token shards: write once, memory-map forever.

Format (little-endian), one file per shard::

    offset  0  : magic   uint32  = 0x6E4C4C4D  ("nLLM")
    offset  4  : version uint32  = 1
    offset  8  : dtype   uint32  = 1 (uint16) | 2 (uint32)
    offset 12  : ntokens uint64
    offset 20  : reserved, zero-filled to HEADER_BYTES
    offset 1024: ntokens tokens, contiguous

A 1024-byte header keeps the token array page-aligned, so ``np.memmap`` hands
back a view the OS can fault in directly.

uint16 covers vocabularies up to 65536, which every config here is under. That
halves the bytes read per token versus int32 -- at 6B tokens the difference is
12GB vs 24GB of page cache.
"""

from __future__ import annotations

import errno
import json
import os
from typing import Iterable, Iterator, Optional

import numpy as np

MAGIC = 0x6E4C4C4D
VERSION = 1
HEADER_BYTES = 1024
DTYPE_CODES = {1: np.uint16, 2: np.uint32}
CODE_FOR_DTYPE = {np.dtype(np.uint16): 1, np.dtype(np.uint32): 2}

__all__ = ["ShardWriter", "read_shard", "shard_token_count", "ShardIndex",
           "dtype_for_vocab", "HEADER_BYTES", "OutOfSpace", "free_bytes"]


class OutOfSpace(OSError):
    """The output filesystem filled up mid-write.

    Raised in place of a bare ``OSError: [Errno 28]`` from deep inside a
    buffered write, where the message gives no hint about how much was
    written or what to do about it.
    """


def free_bytes(path: str) -> int:
    """Bytes available on the filesystem holding ``path``."""
    import shutil
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe or ".").free


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


def _write_header(f, dtype: np.dtype, ntokens: int) -> None:
    header = np.zeros(HEADER_BYTES // 4, dtype=np.uint32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = CODE_FOR_DTYPE[dtype]
    header[3:5] = np.frombuffer(np.uint64(ntokens).tobytes(), dtype=np.uint32)
    f.seek(0)
    f.write(header.tobytes())


def _read_header(path: str) -> tuple[np.dtype, int]:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: truncated header")
    header = np.frombuffer(raw, dtype=np.uint32)
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: bad magic {header[0]:#x}, not a nanollm shard")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path}: shard version {header[1]}, expected {VERSION}")
    dtype = np.dtype(DTYPE_CODES[int(header[2])])
    ntokens = int(np.frombuffer(header[3:5].tobytes(), dtype=np.uint64)[0])
    return dtype, ntokens


class ShardWriter:
    """Append tokens; roll to a new file every ``shard_tokens``.

    The header is rewritten on close with the true count, so a shard that is
    still being written is detectably incomplete (ntokens == 0) rather than
    silently short.
    """

    def __init__(self, out_dir: str, prefix: str, vocab_size: int,
                 shard_tokens: int = 100_000_000, buffer_tokens: int = 1 << 20):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix
        self.dtype = dtype_for_vocab(vocab_size)
        self.vocab_size = vocab_size
        self.shard_tokens = shard_tokens
        self.buffer_tokens = buffer_tokens

        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._file = None
        self._shard_index = 0
        self._shard_written = 0
        self.paths: list[str] = []
        self.total_tokens = 0

    # -- file lifecycle -----------------------------------------------------

    def _open_shard(self) -> None:
        path = os.path.join(
            self.out_dir, f"{self.prefix}_{self._shard_index:05d}.bin")
        self._file = open(path, "wb")
        _write_header(self._file, self.dtype, 0)
        self._file.seek(HEADER_BYTES)
        self._shard_written = 0
        self.paths.append(path)

    def _close_shard(self) -> None:
        if self._file is None:
            return
        _write_header(self._file, self.dtype, self._shard_written)
        self._file.close()
        self._file = None
        self._shard_index += 1

    # -- writing ------------------------------------------------------------

    def add(self, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        if tokens.max(initial=0) >= self.vocab_size or tokens.min(initial=0) < 0:
            raise ValueError(
                f"token id outside [0, {self.vocab_size - 1}] in this batch")
        self._buf.append(tokens.astype(self.dtype, copy=False))
        self._buf_len += tokens.size
        if self._buf_len >= self.buffer_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        block = np.concatenate(self._buf)
        self._buf.clear()
        self._buf_len = 0

        pos = 0
        while pos < block.size:
            if self._file is None:
                self._open_shard()
            room = self.shard_tokens - self._shard_written
            take = min(room, block.size - pos)
            try:
                self._file.write(block[pos:pos + take].tobytes())
            except OSError as e:
                if e.errno != errno.ENOSPC:
                    raise
                raise OutOfSpace(
                    errno.ENOSPC,
                    f"disk full after {self.total_tokens:,} tokens "
                    f"({self.total_tokens * self.dtype.itemsize / 1024**3:.1f}GB "
                    f"in {len(self.paths)} shards) in {self.out_dir!r}. "
                    f"Free space, then re-run with --max-tokens to cap the "
                    f"output (1B tokens is ~2GB at uint16)."
                ) from e
            self._shard_written += take
            self.total_tokens += take
            pos += take
            if self._shard_written >= self.shard_tokens:
                self._close_shard()

    def close(self) -> "ShardIndex":
        self._flush()
        self._close_shard()
        index = ShardIndex(
            dir=self.out_dir,
            shards=[os.path.basename(p) for p in self.paths],
            total_tokens=self.total_tokens,
            dtype=str(self.dtype),
            vocab_size=self.vocab_size,
        )
        index.save()
        return index

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_shard(path: str) -> np.ndarray:
    """Memory-map a shard's tokens. No data is read until it is indexed."""
    dtype, ntokens = _read_header(path)
    if ntokens == 0:
        raise ValueError(f"{path}: shard has 0 tokens (writer did not close cleanly)")
    return np.memmap(path, dtype=dtype, mode="r",
                     offset=HEADER_BYTES, shape=(ntokens,))


def shard_token_count(path: str) -> int:
    return _read_header(path)[1]


class ShardIndex:
    """The manifest written alongside a shard set."""

    FILENAME = "index.json"

    def __init__(self, dir: str, shards: list[str], total_tokens: int,
                 dtype: str, vocab_size: int, meta: Optional[dict] = None):
        self.dir = dir
        self.shards = shards
        self.total_tokens = total_tokens
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.meta = meta or {}

    @property
    def paths(self) -> list[str]:
        return [os.path.join(self.dir, s) for s in self.shards]

    def save(self) -> None:
        path = os.path.join(self.dir, self.FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "shards": self.shards,
                "total_tokens": self.total_tokens,
                "dtype": self.dtype,
                "vocab_size": self.vocab_size,
                "meta": self.meta,
            }, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, directory: str) -> "ShardIndex":
        path = os.path.join(directory, cls.FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no {cls.FILENAME} in {directory!r}. Run "
                f"`python scripts/prepare_data.py` first.")
        with open(path) as f:
            blob = json.load(f)
        return cls(
            dir=directory,
            shards=blob["shards"],
            total_tokens=blob["total_tokens"],
            dtype=blob["dtype"],
            vocab_size=blob["vocab_size"],
            meta=blob.get("meta", {}),
        )
