import math

import pytest
import torch

from nanollm.model import IGNORE_INDEX, Transformer
from nanollm.train.common import (LossWeighting, build_optimizer, clip_and_step,
                                  pad_batch)
from nanollm.train.dpo import build_pair_batch, dpo_loss, sequence_logprobs
from nanollm.utils.schedules import make_lr_fn


# -- schedules --------------------------------------------------------------

def test_cosine_warms_up_and_decays():
    lr = make_lr_fn(1e-3, 1000, 100, min_lr_ratio=0.1, schedule="cosine")
    assert lr(0) == pytest.approx(1e-5)
    assert lr(99) == pytest.approx(1e-3)
    assert lr(1000) == pytest.approx(1e-4, rel=1e-3)
    assert lr(500) > lr(900)


def test_wsd_holds_then_decays():
    lr = make_lr_fn(1e-3, 1000, 50, min_lr_ratio=0.1, schedule="wsd",
                    decay_frac=0.2)
    assert lr(400) == pytest.approx(1e-3)     # stable phase
    assert lr(800) == pytest.approx(1e-3)     # decay starts at 800
    assert lr(1000) == pytest.approx(1e-4, rel=1e-3)


def test_lr_never_goes_negative_or_past_max():
    lr = make_lr_fn(1e-3, 100, 10)
    assert all(0 < lr(s) <= 1e-3 + 1e-12 for s in range(0, 200))


# -- padding / masking ------------------------------------------------------

def test_pad_batch_masks_prompt_tokens():
    xs, ys = pad_batch([([1, 2, 3, 4], [0, 0, 1, 1])], pad_id=0)
    assert xs.tolist() == [[1, 2, 3]]
    # position 0 predicts token 2 (mask 0 -> ignored); position 1 predicts 3.
    assert ys.tolist() == [[IGNORE_INDEX, 3, 4]]


def test_pad_batch_pads_to_longest_row():
    xs, ys = pad_batch([([1, 2, 3, 4], [0, 1, 1, 1]), ([5, 6], [0, 1])], pad_id=7)
    assert xs.shape == (2, 3)
    assert xs[1].tolist() == [5, 6, 7]           # padded with pad_id
    assert ys[1].tolist() == [6, IGNORE_INDEX, IGNORE_INDEX]


def test_pad_batch_rejects_empty():
    with pytest.raises(ValueError):
        pad_batch([], pad_id=0)


# -- gradient clipping ------------------------------------------------------

def test_clip_scales_large_grads_to_the_threshold():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])      # norm 5
    opt = torch.optim.SGD([p], lr=0.0)
    norm, finite = clip_and_step([p], opt, grad_clip=1.0)
    assert norm.item() == pytest.approx(5.0)
    assert finite.item() == 1.0


def test_clip_zeroes_non_finite_grads():
    """A NaN gradient must produce a no-op update, not poison the weights."""
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([float("nan"), 1.0])
    opt = torch.optim.SGD([p], lr=1.0)
    _norm, finite = clip_and_step([p], opt, grad_clip=1.0)
    assert finite.item() == 0.0
    assert torch.equal(p.detach(), torch.ones(2))


def test_optimizer_excludes_norms_from_decay():
    cfg_model = Transformer(vocab_size=64, n_dim=32, n_layer=1, n_head=2,
                            n_kv_head=1, n_seq=16)
    opt = build_optimizer(cfg_model, 1e-3, 0.1, (0.9, 0.95), 1e-8,
                          torch.device("cpu"))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


# -- DPO --------------------------------------------------------------------

def test_dpo_loss_rewards_a_better_chosen_reply():
    ref = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    good = torch.tensor([-0.5, -2.0, -0.5, -2.0])     # chosen clearly ahead
    bad = torch.tensor([-2.0, -0.5, -2.0, -0.5])
    loss_good, stats_good = dpo_loss(good, ref, beta=0.1)
    loss_bad, stats_bad = dpo_loss(bad, ref, beta=0.1)
    assert loss_good < loss_bad
    assert stats_good["accuracy"] == 1.0 and stats_bad["accuracy"] == 0.0


def test_dpo_loss_is_ln2_when_policy_equals_reference():
    lp = torch.tensor([-1.0, -2.0])
    loss, _ = dpo_loss(lp, lp.clone(), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_label_smoothing_bounds_the_loss():
    ref = torch.tensor([-1.0, -1.0])
    lp = torch.tensor([-0.01, -50.0])        # a huge, overconfident margin
    plain, _ = dpo_loss(lp, ref, beta=1.0)
    smoothed, _ = dpo_loss(lp, ref, beta=1.0, label_smoothing=0.1)
    assert smoothed > plain


def test_pair_batch_shares_the_prompt_between_branches():
    xs, ys, n = build_pair_batch([([1, 2, 3], [4, 5], [6])], pad_id=0)
    assert n == 1 and xs.shape[0] == 2
    # Both branches must carry identical prompt tokens, or the implicit reward
    # is comparing two different contexts.
    assert xs[0, :3].tolist() == xs[1, :3].tolist() == [1, 2, 3]
    # Prompt positions contribute no loss.
    assert ys[0, :2].tolist() == [IGNORE_INDEX, IGNORE_INDEX]


def test_sequence_logprobs_are_negative_and_per_sequence():
    torch.manual_seed(0)
    model = Transformer(vocab_size=64, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=32, loss_chunk_size=4).eval()
    xs = torch.randint(0, 64, (2, 8))
    ys = xs.clone()
    ys[:, :3] = IGNORE_INDEX
    with torch.no_grad():
        lp = sequence_logprobs(model, xs, ys)
    assert lp.shape == (2,) and (lp < 0).all()


def test_sequence_logprobs_ignore_masked_positions():
    torch.manual_seed(0)
    model = Transformer(vocab_size=64, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=32, loss_chunk_size=4).eval()
    xs = torch.randint(0, 64, (1, 8))
    all_masked = torch.full_like(xs, IGNORE_INDEX)
    with torch.no_grad():
        assert sequence_logprobs(model, xs, all_masked).item() == 0.0


def test_sequence_logprobs_length_normalization():
    torch.manual_seed(0)
    model = Transformer(vocab_size=64, n_dim=32, n_layer=1, n_head=2,
                        n_kv_head=1, n_seq=32, loss_chunk_size=4).eval()
    xs = torch.randint(0, 64, (2, 8))
    ys = xs.clone()
    ys[:, :4] = IGNORE_INDEX                       # 4 scored tokens per row
    with torch.no_grad():
        summed = sequence_logprobs(model, xs, ys, length_normalize=False)
        mean = sequence_logprobs(model, xs, ys, length_normalize=True)
    assert torch.allclose(mean, summed / 4)
    # length-normalised is the default
    with torch.no_grad():
        assert torch.allclose(sequence_logprobs(model, xs, ys), mean)


# -- SFT loss weighting -----------------------------------------------------

def test_eos_weight_lifts_long_replies_and_leaves_short_ones_alone():
    w = LossWeighting(eos_id=9, eos_share=0.02, eos_cap=12.0)
    # A short reply already stops reliably; its EOS share is above the floor,
    # so the weight must stay at 1 rather than being pulled *down*.
    short = w.row_weights([1, 2, 3, 9], [0, 1, 1, 1])
    assert short == [0.0, 1.0, 1.0, 1.0]
    # A long reply: 2% of the mass on EOS means w/((L-1)+w) == 0.02.
    ids = list(range(200)) + [9]
    mask = [0] + [1] * 200
    long = w.row_weights(ids, mask)
    supervised = sum(mask)
    assert long[-1] == pytest.approx(0.02 * (supervised - 1) / 0.98)
    assert long[-1] / (supervised - 1 + long[-1]) == pytest.approx(0.02)


def test_eos_weight_respects_the_cap():
    w = LossWeighting(eos_id=9, eos_share=0.5, eos_cap=4.0)
    ids = list(range(500)) + [9]
    assert w.row_weights(ids, [0] + [1] * 500)[-1] == 4.0


def test_eos_weight_ignores_rows_that_do_not_end_in_eos():
    w = LossWeighting(eos_id=9, eos_share=0.02)
    ids = list(range(200)) + [5]            # truncated: no closing EOS
    assert w.row_weights(ids, [0] + [1] * 200)[-1] == 1.0


def test_per_example_normalisation_gives_every_example_equal_mass():
    w = LossWeighting(per_example=True)
    short = w.row_weights([1, 2, 9], [0, 1, 1])
    long = w.row_weights(list(range(100)) + [9], [0] + [1] * 100)
    assert sum(short) == pytest.approx(1.0)
    assert sum(long) == pytest.approx(1.0)


def test_pad_batch_weights_align_with_targets():
    w = LossWeighting(eos_id=9, eos_share=0.0)     # inactive -> 2-tuple
    assert len(pad_batch([([1, 2, 3, 9], [0, 1, 1, 1])], pad_id=0,
                         weighting=w)) == 2
    w = LossWeighting(per_example=True)
    xs, ys, ws = pad_batch([([1, 2, 3, 9], [0, 1, 1, 1])], pad_id=0, weighting=w)
    assert ws.shape == ys.shape
    # Weight is non-zero exactly where the target is supervised.
    assert torch.equal(ws[0] > 0, ys[0] != IGNORE_INDEX)


def test_pad_batch_zero_weights_padding():
    w = LossWeighting(per_example=True)
    _, ys, ws = pad_batch([([1, 2, 3, 9], [0, 1, 1, 1]), ([4, 9], [0, 1])],
                          pad_id=0, weighting=w)
    assert ws[1, 1:].sum() == 0.0            # padded tail carries no loss
    assert torch.equal(ws > 0, ys != IGNORE_INDEX)


def test_uniform_weights_reproduce_the_token_mean(tiny_model):
    idx = torch.randint(0, tiny_model.vocab_size, (2, 12))
    xs, ys = idx[:, :-1], idx[:, 1:].clone()
    ys[:, :3] = IGNORE_INDEX
    plain = tiny_model.calculate_loss(xs, ys)
    ones = tiny_model.calculate_loss(xs, ys, weights=torch.ones_like(ys,
                                                                    dtype=torch.float))
    scaled = tiny_model.calculate_loss(xs, ys, weights=torch.full_like(
        ys, 7.0, dtype=torch.float))
    assert torch.allclose(plain, ones, atol=1e-6)
    # A uniform rescale cancels in a weighted mean.
    assert torch.allclose(plain, scaled, atol=1e-5)


def test_weighting_matches_a_hand_computed_weighted_mean(tiny_model):
    """The weighted loss is exactly sum(w*ce)/sum(w) over supervised tokens.

    Checked against the per-token losses directly rather than against a
    "is it closer to X" inequality: on an untrained model every token has
    essentially the same loss, so any such comparison is measuring noise.
    """
    torch.manual_seed(0)
    tiny_model.z_loss_weight = 0.0            # isolate the CE term
    idx = torch.randint(0, tiny_model.vocab_size, (2, 12))
    xs, ys = idx[:, :-1], idx[:, 1:].clone()
    ys[:, :2] = IGNORE_INDEX
    w = torch.rand_like(ys, dtype=torch.float) + 0.5
    w[0, -1] = 12.0                           # an EOS-style upweight

    got = tiny_model.calculate_loss(xs, ys, weights=w)

    logits = tiny_model(xs)[0].float()
    per = torch.nn.functional.cross_entropy(
        logits.reshape(-1, tiny_model.vocab_size), ys.reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX).reshape(ys.shape)
    valid = (ys != IGNORE_INDEX).float()
    want = (per * w * valid).sum() / (w * valid).sum()
    assert torch.allclose(got, want, atol=1e-5)
