import math
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset

MINIMUM_CHUNK_SIZE = 1024
MAXIMUM_CHUNK_SIZE = 3 * 1024
MAX_CORPUS_SIZE = 200 * 1024 ** 2


def chunk_text(text, min_chunk_size=MINIMUM_CHUNK_SIZE, max_chunk_size=MAXIMUM_CHUNK_SIZE):
    """Split a raw text string into chunks respecting paragraph/sentence boundaries."""
    if not isinstance(text, str):
        return []

    # Strip non-ASCII to match original behaviour
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


class PretrainTextDataset(IterableDataset):
    """Streams a random mix of FineWeb, FineWeb-Edu, Cosmopedia and Wikipedia
    from HuggingFace, yielding fixed-size batches of raw text chunks.
    """

    # Updated configs: 
    # - 'wikimedia/wikipedia' replaces the deprecated 'wikipedia' (fixes the .py script error)
    # - 'sample-10BT' for fineweb/fineweb-edu (fast streaming, avoids 15TB download)
    # - 'web_samples_v1' for cosmopedia (it has no 'default' config)
    _DATASET_SPECS = [
        # (hf_repo, config, split, approx_rows_for_weighting, needs_lang_filter)
        ("HuggingFaceFW/fineweb", "sample-10BT", "train", 15_000_000, False),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 15_000_000, False),
        ("HuggingFaceTB/cosmopedia", "web_samples_v1", "train", 5_000_000, False),
        ("wikimedia/wikipedia", "20231101.en", "train", 6_000_000, False),
    ]

    def __init__(
        self,
        batch_size=10,
        min_chunk_size=MINIMUM_CHUNK_SIZE,
        max_chunk_size=MAXIMUM_CHUNK_SIZE,
        dataset_folder=None,           # ignored – kept for API compat
        include_wikipedia=True,
        wikipedia_chunk_size=MINIMUM_CHUNK_SIZE,
        seed=42,
        buffer_target=512,             # keep this many chunks in memory
    ):
        super().__init__()
        self.batch_size = batch_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_target = buffer_target

        specs = list(self._DATASET_SPECS) if include_wikipedia else self._DATASET_SPECS[:-1]

        streams = []
        weights = []
        
        print("[PretrainTextDataset] Connecting to HuggingFace datasets (streaming mode)...")
        for repo, config, split, approx_rows, _ in specs:
            print(f"   - Loading {repo} ({config})...")
            ds = load_dataset(
                repo,
                config,
                split=split,
                streaming=True,
            )
            # Keep only the text column to save bandwidth and memory
            ds = ds.select_columns(["text"])
            streams.append(ds)
            weights.append(approx_rows)

        # Normalise weights into probabilities for balanced interleaving
        total = sum(weights)
        probs = [w / total for w in weights]

        print("[PretrainTextDataset] Interleaving datasets...")
        self._mixed = interleave_datasets(
            streams,
            probabilities=probs,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

        # Approximate total batches (for logging only)
        self._approx_rows = sum(s[3] for s in specs)
        self._approx_batches = max(1, math.ceil(self._approx_rows / batch_size))

    def __len__(self):
        return self._approx_batches

    def __iter__(self):
        buffer = []
        for item in self._mixed:
            text = item.get("text")
            if not text:
                continue

            for chunk in chunk_text(
                text,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size,
            ):
                buffer.append(chunk)

                # Yield batches as soon as we have enough
                if len(buffer) >= self.batch_size:
                    yield buffer[: self.batch_size]
                    buffer = buffer[self.batch_size :]

                # Safety cap so we don't blow up memory on huge documents
                if len(buffer) > self.buffer_target:
                    yield buffer[: self.batch_size]
                    buffer = buffer[self.batch_size :]


if __name__ == "__main__":
    ds = PretrainTextDataset(batch_size=4)
    print(f"~{len(ds)} batches per epoch (approx)")

    for i, batch in enumerate(ds):
        print(f"\n--- batch {i} ({len(batch)} chunks) ---")
        for j, c in enumerate(batch):
            print(f"  [{j}] len={len(c):,}  preview: {c[:120]!r}...")
        if i >= 2:
            break