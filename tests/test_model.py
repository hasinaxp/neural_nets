import pytest
import torch

from nanollm.model import IGNORE_INDEX, Transformer, build_document_mask


def test_forward_shapes(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, aux = tiny_model(idx)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert aux is None


def test_loss_is_near_uniform_at_init(tiny_model, tiny_config):
    import math
    idx = torch.randint(0, tiny_config.vocab_size, (4, 33))
    loss = tiny_model.calculate_loss(idx[:, :-1], idx[:, 1:])
    # A freshly initialised model should be near ln(vocab); far off means the
    # init or the logit scale is wrong.
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 1.0


def test_chunked_loss_matches_unchunked(tiny_config):
    torch.manual_seed(0)
    model = Transformer.from_config(tiny_config).eval()
    idx = torch.randint(0, tiny_config.vocab_size, (2, 32))

    model.loss_chunk_size = 8
    chunked = model.calculate_loss(idx[:, :-1], idx[:, 1:])
    model.loss_chunk_size = 0            # 0 -> one chunk covering the sequence
    whole = model.calculate_loss(idx[:, :-1], idx[:, 1:])
    assert torch.allclose(chunked, whole, atol=1e-5)


def test_ignore_index_excluded_from_loss(tiny_config):
    torch.manual_seed(0)
    model = Transformer.from_config(tiny_config).eval()
    idx = torch.randint(0, tiny_config.vocab_size, (2, 17))
    xs, ys = idx[:, :-1], idx[:, 1:].clone()

    full = model.calculate_loss(xs, ys)
    ys_masked = ys.clone()
    ys_masked[:, ::2] = IGNORE_INDEX
    masked = model.calculate_loss(xs, ys_masked)
    # Masking half the targets must change the mean, not crash or return nan.
    assert torch.isfinite(masked) and not torch.allclose(full, masked)


def test_attention_is_causal(tiny_config):
    """Changing a later token must not change an earlier position's output."""
    torch.manual_seed(0)
    model = Transformer.from_config(tiny_config).eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, 16))

    with torch.no_grad():
        a = model.forward_hidden(idx)
        altered = idx.clone()
        altered[0, -1] = (altered[0, -1] + 1) % tiny_config.vocab_size
        b = model.forward_hidden(altered)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_kv_cache_matches_full_forward(tiny_config):
    """Cached incremental decoding must equal a single full forward pass.

    This is the check that catches a non-causal prefill: with the cache path
    bidirectional over the prompt, these two diverge.
    """
    torch.manual_seed(0)
    model = Transformer.from_config(tiny_config).eval()
    idx = torch.randint(0, tiny_config.vocab_size, (2, 12))

    with torch.no_grad():
        full = model.forward_hidden(idx)

        cache = model.make_kv_cache(2, 16)
        prefill = model.forward_hidden(idx[:, :8], start_pos=0, kv_cache=cache)
        assert torch.allclose(full[:, :8], prefill, atol=1e-4)

        stepwise = []
        for t in range(8, 12):
            out = model.forward_hidden(idx[:, t:t + 1], start_pos=t, kv_cache=cache)
            stepwise.append(out)
        stepwise = torch.cat(stepwise, dim=1)
    assert torch.allclose(full[:, 8:], stepwise, atol=1e-4)


def test_generate_respects_length_and_eos(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (2, 4))
    out = tiny_model.generate(idx, max_count=6, temperature=0.0)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], idx)


def test_generate_masks_padded_vocab(tiny_model, tiny_config):
    idx = torch.randint(0, 100, (1, 3))
    out = tiny_model.generate(idx, max_count=20, valid_vocab_size=100, top_k=0)
    assert int(out.max()) < 100


def test_embeddings_are_tied_by_default(tiny_model):
    assert tiny_model.logit_proj.weight is tiny_model.l_embeddings.weight


def test_untied_embeddings_add_params(tiny_config):
    import dataclasses
    untied = dataclasses.replace(tiny_config, tie_embeddings=False)
    a = Transformer.from_config(tiny_config).get_param_count()
    b = Transformer.from_config(untied).get_param_count()
    assert b - a == tiny_config.vocab_size * tiny_config.n_dim


def test_residual_init_applied_once(tiny_config):
    """wo/d projections carry std/sqrt(2L), not std/(2L).

    Applying the GPT-2 residual scaling in both the model and the training
    script squares the factor and starts the run far too small.
    """
    import math
    torch.manual_seed(0)
    model = Transformer.from_config(tiny_config)
    expected = tiny_config.init_std / math.sqrt(2 * tiny_config.n_layer)
    for name, p in model.named_parameters():
        if name.endswith(("wo.weight", "d.weight")):
            assert 0.5 * expected < p.std().item() < 2.0 * expected, name


def test_document_mask_blocks_across_eos():
    idx = torch.tensor([[1, 2, 9, 3, 4]])
    mask = build_document_mask(idx, eos_id=9)
    assert mask.shape == (1, 1, 5, 5)
    assert not mask[0, 0, 4, 0]      # last token cannot see the first document
    assert mask[0, 0, 4, 3]          # but can see its own
    assert not mask[0, 0, 0, 1]      # still causal


def test_backward_produces_finite_grads(tiny_config):
    model = Transformer.from_config(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 33))
    model.calculate_loss(idx[:, :-1], idx[:, 1:]).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rejects_sequence_longer_than_n_seq(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.n_seq + 1))
    with pytest.raises(ValueError):
        tiny_model.forward_hidden(idx)
