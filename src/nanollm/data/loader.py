"""Batch construction from token shards.

The whole corpus is a single logical token array spread over memory-mapped
shards. A batch is ``micro_batch`` random windows of ``seq_len + 1`` tokens
from that array; x is the first seq_len, y is the shifted copy.

Two properties this buys that an on-the-fly tokenising pipeline does not have:

* **Exact resume.** Batch contents are a pure function of (step, rank), so
  resuming at step N reproduces the same stream without replaying the pipeline.
* **No CPU bottleneck.** Sampling is a memmap read; there is no BPE in the
  training loop, so a single process can saturate an A100.
"""

from __future__ import annotations

import bisect
import queue
import threading
from typing import Iterator, Optional

import numpy as np
import torch

from .shards import ShardIndex, read_shard

__all__ = ["TokenCorpus", "BatchStream", "CudaPrefetcher"]


class TokenCorpus:
    """Read-only view over a set of shards as one concatenated token array."""

    def __init__(self, paths: list[str], seq_len: int):
        if not paths:
            raise ValueError("TokenCorpus needs at least one shard")
        self.paths = list(paths)
        self.seq_len = seq_len
        self._maps: Optional[list[np.ndarray]] = None

        # Header-only pass: sizes are known without faulting in any token data.
        from .shards import shard_token_count
        self.lengths = [shard_token_count(p) for p in self.paths]
        self.total_tokens = sum(self.lengths)

        self.offsets = []
        running = 0
        for n in self.lengths:
            running += n
            self.offsets.append(running)

        # A window must fit entirely inside one shard; sampling across a shard
        # boundary would splice unrelated documents mid-sequence.
        self.window = seq_len + 1
        self.usable = [max(0, n - self.window) for n in self.lengths]
        self.total_windows = sum(self.usable)
        if self.total_windows <= 0:
            raise ValueError(
                f"no shard is longer than seq_len+1={self.window}; "
                f"shard sizes {self.lengths}")

        self._win_offsets = []
        running = 0
        for n in self.usable:
            running += n
            self._win_offsets.append(running)

    @classmethod
    def from_index(cls, index: ShardIndex, seq_len: int,
                   subset: Optional[list[str]] = None) -> "TokenCorpus":
        paths = [p for p in index.paths
                 if subset is None or p.split("/")[-1] in subset]
        return cls(paths, seq_len)

    def _ensure_maps(self) -> list[np.ndarray]:
        # Lazily opened so the object stays picklable across worker processes.
        if self._maps is None:
            self._maps = [read_shard(p) for p in self.paths]
        return self._maps

    def gather(self, window_ids: np.ndarray) -> np.ndarray:
        """Materialise ``len(window_ids)`` windows of ``seq_len + 1`` tokens."""
        maps = self._ensure_maps()
        out = np.empty((len(window_ids), self.window), dtype=np.int64)
        for row, wid in enumerate(window_ids):
            shard = bisect.bisect_right(self._win_offsets, wid)
            prev = self._win_offsets[shard - 1] if shard else 0
            start = int(wid - prev)
            out[row] = maps[shard][start:start + self.window]
        return out


class BatchStream:
    """Deterministic (step, rank) -> batch mapping.

    Each rank draws its own windows from a generator seeded with
    ``seed * 1_000_003 + step``, offset by rank, so ranks never collide and any
    step is reproducible in isolation.
    """

    def __init__(self, corpus: TokenCorpus, micro_batch_size: int,
                 seed: int = 1337, rank: int = 0, world_size: int = 1,
                 start_step: int = 0, pin_memory: bool = True):
        self.corpus = corpus
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.step = start_step
        self.pin_memory = pin_memory

    def batch_at(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + step) * self.world_size + self.rank)
        ids = rng.integers(0, self.corpus.total_windows,
                           size=self.micro_batch_size, dtype=np.int64)
        block = torch.from_numpy(self.corpus.gather(ids))
        xs = block[:, :-1].contiguous()
        ys = block[:, 1:].contiguous()
        if self.pin_memory:
            xs, ys = xs.pin_memory(), ys.pin_memory()
        return xs, ys

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.batch_at(self.step)
            self.step += 1

    def take(self, n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """A fixed list of batches -- used to freeze a validation set."""
        return [self.batch_at(i) for i in range(n)]


class CudaPrefetcher:
    """Overlaps host->device copies and batch assembly with compute.

    The producer thread builds batches and issues each H2D copy on a side CUDA
    stream; the consumer waits on the copy's event, so neither the memmap reads
    nor the transfer sit in the critical path.
    """

    def __init__(self, stream, device: torch.device, depth: int = 4,
                 enabled: bool = True):
        self.stream = stream
        self.device = device
        self.depth = depth
        self.enabled = enabled
        self.copy_stream = (torch.cuda.Stream(device=device)
                            if device.type == "cuda" else None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def _issue(self, xs, ys):
        if self.copy_stream is None:
            return xs.to(self.device), ys.to(self.device), None
        with torch.cuda.stream(self.copy_stream):
            xs = xs.to(self.device, non_blocking=True)
            ys = ys.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return xs, ys, event

    @staticmethod
    def _consume(item):
        xs, ys, event = item
        if event is not None:
            compute = torch.cuda.current_stream()
            compute.wait_event(event)
            # Without record_stream the allocator can hand these blocks out
            # again while the compute stream is still reading them.
            xs.record_stream(compute)
            ys.record_stream(compute)
        return xs, ys

    def __iter__(self):
        if not self.enabled:
            for xs, ys in self.stream:
                yield self._consume(self._issue(xs, ys))
            return

        q: queue.Queue = queue.Queue(maxsize=self.depth)
        sentinel = object()
        self._stop.clear()

        def put(item) -> bool:
            while not self._stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for xs, ys in self.stream:
                    if not put(self._issue(xs, ys)):
                        return
            except Exception as e:            # surface it, never hang
                put(e)
                return
            put(sentinel)

        self._thread = threading.Thread(
            target=producer, daemon=True, name="batch-prefetch")
        self._thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield self._consume(item)
