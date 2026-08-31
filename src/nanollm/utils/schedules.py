"""Learning-rate schedules."""

from __future__ import annotations

import math

__all__ = ["lr_lambda", "make_lr_fn"]


def make_lr_fn(peak_lr: float, max_steps: int, warmup_steps: int,
               min_lr_ratio: float = 0.1, schedule: str = "cosine",
               decay_frac: float = 0.1):
    """Returns ``step -> lr``.

    * ``cosine``  -- linear warmup, cosine decay to ``peak_lr * min_lr_ratio``.
    * ``wsd``     -- warmup, hold at peak, then a short 1-sqrt decay. Lets you
                     stop or extend a run at any point during the stable phase
                     without having chosen ``max_steps`` up front.
    * ``constant``-- warmup then flat, for debugging.
    """
    min_lr = peak_lr * min_lr_ratio
    warmup_steps = max(1, warmup_steps)

    def cosine(step: int) -> float:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def wsd(step: int) -> float:
        decay_steps = max(1, int(decay_frac * max_steps))
        stable_end = max_steps - decay_steps
        if step < stable_end:
            return peak_lr
        progress = (step - stable_end) / decay_steps
        progress = min(1.0, max(0.0, progress))
        return min_lr + (peak_lr - min_lr) * (1.0 - math.sqrt(progress))

    def constant(step: int) -> float:
        return peak_lr

    body = {"cosine": cosine, "wsd": wsd, "constant": constant}[schedule]

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        return body(step)

    return lr_at


# Backwards-compatible alias.
lr_lambda = make_lr_fn
