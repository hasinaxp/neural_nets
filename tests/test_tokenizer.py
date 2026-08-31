import pytest

from nanollm.tokenizer import Tokenizer

SAMPLE = ("the quick brown fox jumps over the lazy dog. " * 60
          + "the quick brown cat sleeps under the warm sun. " * 60)


@pytest.fixture(scope="module")
def trained():
    tok = Tokenizer(vocab_size=400)
    tok.train_from_text(SAMPLE)
    return tok


def test_roundtrip_is_lossless(trained):
    text = "the quick brown fox jumps over the lazy dog."
    assert trained.decode(trained.encode(text)) == text


def test_merges_compress(trained):
    text = "the quick brown fox " * 10
    assert len(trained.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens_survive_roundtrip(trained):
    text = "<|BOS|><|USER|>hello<|ASSISTANT|>hi<|EOS|>"
    ids = trained.encode(text)
    for name in ("<|BOS|>", "<|USER|>", "<|ASSISTANT|>", "<|EOS|>"):
        assert trained.special_tokens[name] in ids
    assert trained.decode(ids) == text


def test_special_token_ids_are_stable():
    # Shard files and checkpoints hardcode these ids; a reordering silently
    # corrupts every corpus tokenised before the change.
    tok = Tokenizer(vocab_size=1000)
    assert tok.special_tokens["<|BOS|>"] == 256
    assert tok.special_tokens["<|EOS|>"] == 257
    assert tok.merge_id_offset == 256 + len(Tokenizer.SPECIAL_TOKENS)


def test_save_load_roundtrip(trained, tmp_path):
    path = tmp_path / "tok.txt"
    trained.save(str(path))
    other = Tokenizer(vocab_size=400)
    other.load(str(path))
    text = "the quick brown fox jumps."
    assert other.encode(text) == trained.encode(text)
    assert other.decode(other.encode(text)) == text


def test_all_ids_are_within_vocab(trained):
    ids = trained.encode(SAMPLE[:2000])
    assert ids and max(ids) < max(trained.vocab) + 1


def test_empty_string_encodes_to_nothing(trained):
    assert trained.encode("") == []
