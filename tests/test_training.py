import math

import pytest
import torch

from nanollm.model import IGNORE_INDEX, Transformer
from nanollm.train.common import build_optimizer, clip_and_step, pad_batch
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
