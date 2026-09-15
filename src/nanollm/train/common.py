"""Pieces shared by the pretrain / SFT / DPO loops."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import TrainConfig
from ..model import IGNORE_INDEX, Transformer
from ..tokenizer import Tokenizer

__all__ = [
    "configure_backends", "load_tokenizer", "build_optimizer", "clip_and_step",
    "load_pretrained", "pad_batch", "peak_flops", "resolve_amp_dtype",
    "LossWeighting",
]

# Dense bf16 peak, FLOP/s, for the MFU denominator. Unknown cards report 0 and
# MFU is simply not shown rather than being quietly wrong.
PEAK_TFLOPS = {
    "a100": 312e12, "h100": 989e12, "h200": 989e12, "l40": 181e12,
    "a10": 125e12, "v100": 125e12, "rtx 4090": 165e12, "rtx 3090": 71e12,
    # RTX PRO 6000 Blackwell: 24064 shaders at ~2.6GHz is ~125 TFLOP/s FP32,
    # and the Blackwell tensor cores do bf16 with fp32 accumulate at 2x the
    # shader rate (same ratio as Ada, where the 4090's 82.6 -> 165 above comes
    # from). Dense, no sparsity.
    "rtx pro 6000": 250e12,
}


def peak_flops(device: torch.device, world_size: int = 1) -> float:
    if device.type != "cuda":
        return 0.0
    name = torch.cuda.get_device_name(device).lower()
    for key, value in PEAK_TFLOPS.items():
        if key in name:
            return value * world_size
    return 0.0


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    want = {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]
    if want is torch.bfloat16 and device.type == "cuda" \
            and not torch.cuda.is_bf16_supported():
        return torch.float16
    return want


def configure_backends(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(cfg.runtime.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    return resolve_amp_dtype(cfg.runtime.dtype, device)


def load_tokenizer(cfg: TrainConfig, log) -> tuple[Tokenizer, int, int]:
    """Returns (tokenizer, padded model vocab, eos id).

    The model vocab is rounded up to a multiple of 64 so the logit matmul lands
    on tensor-core-friendly shapes; the padded tail is masked when sampling.
    """
    tok = Tokenizer(vocab_size=cfg.model.vocab_size)
    tok.load(cfg.data.tokenizer_file)
    actual = max(tok.vocab) + 1 if tok.vocab else tok.vocab_size
    padded = int(math.ceil(actual / 64) * 64)
    log.info(f"    tokenizer vocab {actual} -> model vocab {padded} (padded to /64)")
    eos = tok.special_tokens.get("<|EOS|>") or tok.special_tokens.get("<|BOS|>")
    if eos is None:
        raise RuntimeError("tokenizer needs an <|EOS|> or <|BOS|> special token")
    return tok, padded, eos


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                    betas: tuple[float, float], eps: float,
                    device: torch.device, log=None) -> torch.optim.AdamW:
    """Decay matrices, not gains and biases -- the standard split."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    if log is not None:
        log.info(f"    decay tensors: {len(decay)} "
                 f"({sum(p.numel() for p in decay):,} params) | "
                 f"no-decay: {len(no_decay)} ({sum(p.numel() for p in no_decay):,})")
    fused_ok = (device.type == "cuda"
                and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps,
        **({"fused": True} if fused_ok else {}),
    )


def clip_and_step(params, optimizer, grad_clip: float, scaler=None):
    """Clip, step, and report (grad_norm, finite) without a host sync.

    The clip coefficient carries the finite check: a non-finite global norm
    gives scale 0, so the CPU never waits on a device->host bool to decide
    whether to skip the step.

    Scaling alone is not enough to neutralise a bad step, though -- NaN * 0 is
    NaN, not 0, so a single NaN gradient would survive into ``p`` and, worse,
    into AdamW's ``exp_avg``, which poisons every subsequent step even after
    the gradients recover. The nan_to_num pass is what actually makes a
    non-finite step a no-op. It is one extra elementwise pass over the
    gradients (well under 1% of step time) and it runs unconditionally,
    because branching on the result would reintroduce the sync.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)

    grads = [p.grad for p in params if p.grad is not None]
    grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    finite = torch.isfinite(grad_norm)
    scale = torch.where(finite,
                        (grad_clip / (grad_norm + 1e-6)).clamp(max=1.0),
                        grad_norm.new_zeros(()))
    torch._foreach_mul_(grads, scale)
    for g in grads:
        torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm.detach().float(), finite.float()


def load_pretrained(model: Transformer, path: str, device, log,
                    strict: bool = True) -> dict:
    """Initialise from a previous stage's checkpoint or bare state dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no weights at {path}. Run the previous stage first, or point "
            f"init_from at a checkpoint you have.")
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

    # Strip DDP / compile prefixes so a checkpoint saved either way loads.
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state dict mismatch loading {path}: "
            f"{len(missing)} missing (e.g. {missing[:2]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:2]})")
    log.info(f"    initialised from {path}")
    return blob if isinstance(blob, dict) else {}


@dataclass
class LossWeighting:
    """Per-token weights for the SFT cross entropy.

    Two independent corrections, both of which exist because the plain token
    mean spends the gradient somewhere other than where the mixture says.

    ``eos_share`` -- the end-of-reply token is one token out of however many the
    reply has, so a 6-token QA answer trains stopping ~25x harder than a
    155-token draft. Measured on a finished run, P(EOS) at the correct stopping
    position fell from 0.84 on replies of 1-16 tokens to 0.55 past 257, and the
    tasks that ramble (writing 0.39, instruct 0.43, chat 0.61) are exactly the
    long-reply ones. This pins EOS at a floor share of each example's loss mass:
    for a reply of L supervised tokens the weight solving w/((L-1)+w) = share,
    clamped so short replies -- already at P(EOS) ~0.99 -- are never touched and
    a very long one cannot hand a single position an unbounded weight.

    ``per_example`` -- the token mean weights an example by its reply length, so
    TASK_WEIGHTS describes the share of *examples* while the gradient follows
    the share of *tokens*. At the configured mixture that gives chat 78% of the
    gradient against its nominal 24%, and extractive_qa 0.4% against 12%.
    Normalising each example to a fixed mass makes the delivered gradient match
    the mixture that was actually designed.
    """

    eos_id: Optional[int] = None
    eos_share: float = 0.0
    eos_cap: float = 12.0
    per_example: bool = False

    @property
    def active(self) -> bool:
        return (self.eos_share > 0 and self.eos_id is not None) or self.per_example

    def row_weights(self, ids: list[int], mask: list[int]) -> list[float]:
        """Weight per position, aligned with ``ids`` (so it shifts like targets)."""
        w = [1.0 if m else 0.0 for m in mask]
        supervised = sum(mask)
        if (self.eos_share > 0 and self.eos_id is not None and supervised > 1
                and ids[-1] == self.eos_id and mask[-1]):
            target = self.eos_share * (supervised - 1) / (1.0 - self.eos_share)
            # Never below 1.0: short replies already stop reliably and pushing
            # them further only trades answer length for a stop token.
            w[-1] = min(self.eos_cap, max(1.0, target))
        if self.per_example:
            total = sum(w)
            if total > 0:
                w = [x / total for x in w]
        return w


def pad_batch(rows: list[tuple[list[int], list[int]]], pad_id: int,
              device=None, weighting: Optional[LossWeighting] = None):
    """Pad (ids, loss_mask) rows to the batch's longest sequence.

    Returns (xs, ys) already shifted for next-token prediction, with
    non-target positions set to IGNORE_INDEX so they contribute no loss.
    Padding to the longest row in the batch rather than to seq_len keeps the
    step cost proportional to the actual content.

    With an active ``weighting`` the return is (xs, ys, ws), where ws is the
    per-token multiplier to hand to ``Transformer.calculate_loss``. Without one
    the return stays a 2-tuple and nothing about the loss changes.
    """
    if not rows:
        raise ValueError("empty batch")
    width = max(len(ids) for ids, _ in rows)
    n = len(rows)
    weighted = weighting is not None and weighting.active

    tokens = torch.full((n, width), pad_id, dtype=torch.long)
    targets = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
    weights = torch.zeros((n, width), dtype=torch.float) if weighted else None
    for i, (ids, mask) in enumerate(rows):
        length = len(ids)
        tokens[i, :length] = torch.tensor(ids, dtype=torch.long)
        keep = torch.tensor(mask, dtype=torch.bool)
        row = torch.tensor(ids, dtype=torch.long)
        targets[i, :length] = torch.where(
            keep, row, torch.full_like(row, IGNORE_INDEX))
        if weighted:
            weights[i, :length] = torch.tensor(
                weighting.row_weights(ids, mask), dtype=torch.float)

    xs = tokens[:, :-1].contiguous()
    ys = targets[:, 1:].contiguous()      # predict position t+1 from t
    if device is not None:
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
    if not weighted:
        return xs, ys
    ws = weights[:, 1:].contiguous()
    if device is not None:
        ws = ws.to(device, non_blocking=True)
    return xs, ys, ws
