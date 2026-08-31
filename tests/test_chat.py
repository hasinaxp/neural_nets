import pytest
import torch

from nanollm.chat import ChatSession, SamplingParams, stream_reply
from nanollm.model import Transformer
from nanollm.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer(vocab_size=400)
    t.train_from_text("the quick brown fox jumps over the lazy dog. " * 80)
    return t


@pytest.fixture
def session(tok):
    return ChatSession(tokenizer=tok, n_seq=256)


def ids_to_markers(tok, ids):
    """Keep only the special-token markers, so template order is checkable."""
    inv = {v: k for k, v in tok.special_tokens.items()}
    return [inv[i] for i in ids if i in inv]


def test_prompt_matches_the_sft_template(session, tok):
    session.add_user("hello")
    ids, _ = session.build_prompt(reserve=16)
    # Must be exactly what render_conversation produces, up to <|ASSISTANT|>.
    assert ids_to_markers(tok, ids) == ["<|BOS|>", "<|USER|>", "<|ASSISTANT|>"]
    assert ids[0] == tok.special_tokens["<|BOS|>"]
    assert ids[-1] == tok.special_tokens["<|ASSISTANT|>"]


def test_multi_turn_closes_assistant_turns_with_eos(session, tok):
    session.add_user("one")
    session.add_assistant("two")
    session.add_user("three")
    ids, _ = session.build_prompt(reserve=16)
    assert ids_to_markers(tok, ids) == [
        "<|BOS|>", "<|USER|>", "<|ASSISTANT|>", "<|EOS|>",
        "<|USER|>", "<|ASSISTANT|>",
    ]


def test_history_is_preserved_across_turns(session):
    session.add_user("a")
    session.add_assistant("b")
    session.add_user("c")
    assert [m["role"] for m in session.messages] == ["user", "assistant", "user"]


def test_undo_drops_one_full_exchange(session):
    session.add_user("a")
    session.add_assistant("b")
    session.add_user("c")
    session.add_assistant("d")
    session.undo()
    assert [m["content"] for m in session.messages] == ["a", "b"]


def test_undo_then_re_add_reasks_the_same_question(session):
    """This is what /retry does: it must resample the *latest* question."""
    session.add_user("first")
    session.add_assistant("answer one")
    session.add_user("second")
    session.add_assistant("answer two")

    last_user = [m for m in session.messages if m["role"] == "user"][-1]
    session.undo()
    session.add_user(last_user["content"])

    assert [m["content"] for m in session.messages] == [
        "first", "answer one", "second"]


def test_reset_clears_everything(session):
    session.add_user("a")
    session.add_assistant("b")
    session.reset()
    assert session.messages == []
    ids, _ = session.build_prompt(reserve=8)
    assert len(ids) == 2       # just <|BOS|> <|ASSISTANT|>


def test_oldest_turns_are_dropped_when_context_overflows(tok):
    session = ChatSession(tokenizer=tok, n_seq=128)
    for i in range(12):
        session.add_user(f"question number {i} the quick brown fox jumps over")
        session.add_assistant(f"answer number {i} the lazy dog sleeps quietly")
    session.add_user("the final question that must survive")

    ids, dropped = session.build_prompt(reserve=32)
    assert dropped > 0
    assert len(ids) <= 128 - 32
    # The newest question is the one that has to be intact.
    tail = tok.decode([i for i in ids if i not in tok.special_tokens.values()])
    assert "final question" in tail


def test_prompt_always_fits_the_reply_budget(tok):
    session = ChatSession(tokenizer=tok, n_seq=64)
    session.add_user("the quick brown fox jumps over the lazy dog " * 20)
    ids, _ = session.build_prompt(reserve=16)
    assert len(ids) <= 64 - 16


def test_stream_reply_stops_at_eos(tok):
    """EOS terminates the reply and is never emitted as text.

    The sampler is stubbed rather than the weights: logit_proj is tied to the
    embedding, so forcing a token by editing that matrix also changes the
    input representation, which does not isolate what is under test.
    """
    torch.manual_seed(0)
    model = Transformer(vocab_size=400, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=128, loss_chunk_size=32).eval()
    eos = tok.special_tokens["<|EOS|>"]

    emitted = []

    def fake_sample(logits, *_args, **_kwargs):
        # two real tokens, then stop
        token = [7, 8, eos][min(len(emitted), 2)]
        emitted.append(token)
        return torch.tensor([[token]])

    model._sample = fake_sample
    params = SamplingParams(temperature=1.0, max_new_tokens=10,
                            repetition_penalty=1.0)
    out = "".join(stream_reply(model, tok, [tok.special_tokens["<|BOS|>"], 5],
                               params, torch.device("cpu"), torch.float32))
    assert emitted == [7, 8, eos]
    assert out == tok.decode([7, 8])
    assert "<|EOS|>" not in out


def test_stream_reply_respects_max_new_tokens(tok):
    torch.manual_seed(0)
    model = Transformer(vocab_size=400, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=128, loss_chunk_size=32).eval()
    params = SamplingParams(temperature=1.0, max_new_tokens=5,
                            repetition_penalty=1.0)
    prompt = [tok.special_tokens["<|BOS|>"], tok.special_tokens["<|USER|>"], 7]
    deltas = list(stream_reply(model, tok, prompt, params,
                               torch.device("cpu"), torch.float32))
    assert 0 < len(deltas) <= 5


def test_stream_reply_never_exceeds_the_context(tok):
    """max_new_tokens larger than the remaining window must be clamped, not
    raise out of make_kv_cache."""
    torch.manual_seed(0)
    model = Transformer(vocab_size=400, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=32, loss_chunk_size=16).eval()
    params = SamplingParams(temperature=1.0, max_new_tokens=1000,
                            repetition_penalty=1.0)
    prompt = [tok.special_tokens["<|BOS|>"]] + [7] * 20
    deltas = list(stream_reply(model, tok, prompt, params,
                               torch.device("cpu"), torch.float32))
    assert len(deltas) <= 32 - 21
