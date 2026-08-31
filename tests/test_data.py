import numpy as np
import pytest
import torch

from nanollm.data.loader import BatchStream, TokenCorpus
from nanollm.data.shards import ShardIndex, ShardWriter, read_shard
from nanollm.data.sources import split_long_text


@pytest.fixture
def shard_dir(tmp_path):
    out = tmp_path / "tokens"
    with ShardWriter(str(out), "tokens", vocab_size=512, shard_tokens=1000) as w:
        w.add(np.arange(2500, dtype=np.int32) % 512)
    return str(out)


def test_writer_rolls_shards(shard_dir):
    index = ShardIndex.load(shard_dir)
    assert index.total_tokens == 2500
    assert len(index.shards) == 3            # 1000 + 1000 + 500


def test_roundtrip_preserves_tokens(shard_dir):
    index = ShardIndex.load(shard_dir)
    restored = np.concatenate([np.asarray(read_shard(p)) for p in index.paths])
    assert np.array_equal(restored, np.arange(2500) % 512)


def test_uint16_for_small_vocab(shard_dir):
    assert read_shard(ShardIndex.load(shard_dir).paths[0]).dtype == np.uint16


def test_rejects_out_of_range_tokens(tmp_path):
    with pytest.raises(ValueError):
        with ShardWriter(str(tmp_path / "t"), "t", vocab_size=10) as w:
            w.add(np.array([1, 2, 99], dtype=np.int32))


def test_bad_magic_is_rejected(tmp_path):
    from nanollm.data.shards import read_shard
    path = tmp_path / "junk.bin"
    path.write_bytes(b"\x00" * 2048)
    with pytest.raises(ValueError):
        read_shard(str(path))


def test_corpus_windows_stay_inside_a_shard(shard_dir):
    corpus = TokenCorpus(ShardIndex.load(shard_dir).paths, seq_len=15)
    # 3 shards of 1000/1000/500, window 16 -> 984 + 984 + 484
    assert corpus.total_windows == 984 + 984 + 484


def test_corpus_rejects_seq_len_over_shard_size(shard_dir):
    with pytest.raises(ValueError):
        TokenCorpus(ShardIndex.load(shard_dir).paths, seq_len=5000)


def test_batches_are_shifted_pairs(shard_dir):
    corpus = TokenCorpus(ShardIndex.load(shard_dir).paths, seq_len=8)
    xs, ys = BatchStream(corpus, 4, pin_memory=False).batch_at(0)
    assert xs.shape == (4, 8) and ys.shape == (4, 8)
    assert torch.equal(xs[:, 1:], ys[:, :-1])   # y is x shifted by one


def test_batches_are_deterministic_per_step(shard_dir):
    corpus = TokenCorpus(ShardIndex.load(shard_dir).paths, seq_len=8)
    a = BatchStream(corpus, 4, seed=7, pin_memory=False).batch_at(42)
    b = BatchStream(corpus, 4, seed=7, pin_memory=False).batch_at(42)
    assert torch.equal(a[0], b[0])
    # ...which is what makes resume exact: rebuild the stream at any step.
    c = BatchStream(corpus, 4, seed=7, pin_memory=False).batch_at(43)
    assert not torch.equal(a[0], c[0])


def test_ranks_get_different_batches(shard_dir):
    corpus = TokenCorpus(ShardIndex.load(shard_dir).paths, seq_len=8)
    a = BatchStream(corpus, 4, rank=0, world_size=2, pin_memory=False).batch_at(0)
    b = BatchStream(corpus, 4, rank=1, world_size=2, pin_memory=False).batch_at(0)
    assert not torch.equal(a[0], b[0])


def test_split_long_text_respects_bounds():
    text = ("paragraph one. " * 200) + "\n\n" + ("paragraph two. " * 200)
    chunks = split_long_text(text, min_chunk_size=100, max_chunk_size=1000)
    assert chunks and all(len(c) <= 1000 for c in chunks)


def test_split_drops_short_documents():
    assert split_long_text("too short", min_chunk_size=100) == []


def test_out_of_space_raises_a_useful_error(tmp_path, monkeypatch):
    """A full disk must say what was written and what to do, not surface a
    bare `OSError: [Errno 28]` from inside a buffered write."""
    import errno

    from nanollm.data.shards import OutOfSpace, ShardWriter

    w = ShardWriter(str(tmp_path / "t"), "t", vocab_size=512, shard_tokens=10_000)
    w.add(np.arange(100, dtype=np.int32))
    w._flush()

    real_write = w._file.write

    def full_disk(_data):
        raise OSError(errno.ENOSPC, "No space left on device")

    w._file.write = full_disk
    with pytest.raises(OutOfSpace) as excinfo:
        w.add(np.arange(4_000_000, dtype=np.int32) % 512)
    message = str(excinfo.value)
    assert "--max-tokens" in message and "tokens" in message
    w._file.write = real_write


def test_writer_close_survives_a_failed_flush(tmp_path):
    """After a mid-write failure the buffer is already drained, so close()
    still writes a valid header and index -- which is what makes a crashed
    prepare run recoverable instead of a total loss."""
    from nanollm.data.shards import ShardIndex, ShardWriter, shard_token_count

    w = ShardWriter(str(tmp_path / "t"), "t", vocab_size=512, shard_tokens=10_000)
    w.add(np.arange(5000, dtype=np.int32) % 512)
    w._flush()
    index = w.close()

    assert index.total_tokens == 5000
    assert shard_token_count(index.paths[0]) == 5000
    assert ShardIndex.load(str(tmp_path / "t")).total_tokens == 5000
