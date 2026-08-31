"""Checkpoint save/load: atomic writes, architecture-aware resume."""

from __future__ import annotations

import os
import shutil
from typing import Any, Optional

import torch

__all__ = ["atomic_save", "save_checkpoint", "load_checkpoint",
           "unwrap_model", "ArchitectureMismatch"]


class ArchitectureMismatch(Exception):
    """Not an error: the checkpoint describes a different model, so it is set
    aside and the run starts fresh."""


def atomic_save(obj: Any, path: str) -> None:
    """Write to a temp file then rename. A crash mid-write cannot corrupt the
    previous checkpoint, which is otherwise the classic way to lose a run."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def unwrap_model(model):
    """Strip DDP / torch.compile wrappers to reach the real module.

    State dicts must be saved unwrapped -- ``module.`` and ``_orig_mod.``
    prefixes otherwise leak into the keys and break every other loader.
    """
    seen = set()
    while True:
        inner = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if inner is None or id(inner) in seen:
            return model
        seen.add(id(inner))
        model = inner


def save_checkpoint(path: str, *, model, optimizer, step: int, config: dict,
                    history: Optional[dict] = None, scaler=None,
                    extra: Optional[dict] = None) -> None:
    payload = {
        "global_step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history or {},
        "format_version": 2,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    atomic_save(payload, path)


def load_checkpoint(path: str, *, model, optimizer=None, scaler=None,
                    map_location="cpu", expect_arch: Optional[dict] = None,
                    log=print) -> dict:
    """Restore in place. Raises ArchitectureMismatch if shapes disagree."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    target = unwrap_model(model)

    if expect_arch:
        old = ck.get("config") or {}
        old_model = old.get("model", old)      # v2 nests under "model"
        diff = {k: (old_model[k], v) for k, v in expect_arch.items()
                if k in old_model and old_model[k] != v}
        if diff:
            raise ArchitectureMismatch(f"config differs (old -> new): {diff}")

    # Shape check before load_state_dict, which would otherwise half-load.
    want = target.state_dict()
    have = ck["model_state_dict"]
    bad = [k for k in want
           if k not in have or tuple(have[k].shape) != tuple(want[k].shape)]
    if bad:
        raise ArchitectureMismatch(
            f"{len(bad)} tensors differ in shape or are missing, e.g. {bad[0]}")

    target.load_state_dict(have)

    if optimizer is not None and "optimizer_state_dict" in ck:
        try:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        except Exception as e:
            log(f"optimizer state did not load ({e}); Adam moments restart at zero")
    if scaler is not None and "scaler_state_dict" in ck:
        try:
            scaler.load_state_dict(ck["scaler_state_dict"])
        except Exception as e:
            log(f"scaler state did not load ({e}); continuing")

    return ck


def set_aside(path: str, suffix: str, log=print) -> str:
    """Move a checkpoint out of the way instead of overwriting it."""
    stale = f"{path}.{suffix}.bak"
    shutil.move(path, stale)
    log(f"previous checkpoint kept at {stale}")
    return stale
