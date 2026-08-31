"""Run logging: one file per run, rank-0 only, optional TB/W&B mirrors."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

__all__ = ["setup_logging", "MetricLogger", "run_id"]

_NOISY = ("datasets", "pyarrow", "huggingface_hub", "urllib3", "fsspec",
          "tokenizers", "transformers", "matplotlib", "PIL")


def run_id(name: str = "") -> str:
    return name or datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(log_dir: str, run: str, prefix: str = "train",
                  is_main: bool = True, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("nanollm")
    logger.handlers.clear()
    logger.propagate = False

    # Non-main ranks log warnings and above only; otherwise every message is
    # printed world_size times.
    logger.setLevel(logging.INFO if is_main else logging.WARNING)

    fmt = logging.Formatter(
        f"%(asctime)s [%(levelname)s]{'' if is_main else f' [rank{rank}]'} %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if is_main:
        path = os.path.join(log_dir, f"{prefix}_{run}.log")
        fileh = logging.FileHandler(path)
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
        logger.info(f"logging to {path}")

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


class MetricLogger:
    """Accumulates run history and mirrors it to whatever sinks are enabled.

    TensorBoard and W&B are both optional; a missing package downgrades to a
    warning rather than taking the run down at step 0.
    """

    KEYS = ("steps", "train_loss", "lr", "grad_norm", "tokens_per_sec", "mfu",
            "val_steps", "val_loss")

    def __init__(self, log_dir: str, run: str, enabled: bool = True,
                 tensorboard: bool = False, wandb: bool = False,
                 wandb_project: str = "nanollm", config: Optional[dict] = None,
                 logger=None):
        self.enabled = enabled
        self.log_dir = log_dir
        self.run = run
        self.path = os.path.join(log_dir, "metrics.json")
        self.history = {k: [] for k in self.KEYS}
        self.history["run_id"] = run
        self._log = logger or logging.getLogger("nanollm")

        self.tb = None
        self.wandb = None
        if not enabled:
            return

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, "tb", run))
            except Exception as e:
                self._log.warning(f"tensorboard disabled: {e}")
        if wandb:
            try:
                import wandb as _wandb
                _wandb.init(project=wandb_project, name=run, config=config or {})
                self.wandb = _wandb
            except Exception as e:
                self._log.warning(f"wandb disabled: {e}")

    def log(self, step: int, **values) -> None:
        if not self.enabled:
            return
        for key, value in values.items():
            if key in self.history:
                self.history[key].append(value)
            if self.tb is not None:
                self.tb.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def log_train(self, step: int, loss: float, lr: float, grad_norm: float,
                  tokens_per_sec: float, mfu: float = 0.0) -> None:
        if not self.enabled:
            return
        self.history["steps"].append(step)
        self.history["train_loss"].append(loss)
        self.history["lr"].append(lr)
        self.history["grad_norm"].append(grad_norm)
        self.history["tokens_per_sec"].append(tokens_per_sec)
        self.history["mfu"].append(mfu)
        self._mirror(step, {"train/loss": loss, "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "perf/tokens_per_sec": tokens_per_sec,
                            "perf/mfu": mfu})

    def log_val(self, step: int, loss: float) -> None:
        if not self.enabled:
            return
        self.history["val_steps"].append(step)
        self.history["val_loss"].append(loss)
        self._mirror(step, {"val/loss": loss})

    def _mirror(self, step: int, values: dict) -> None:
        if self.tb is not None:
            for k, v in values.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def save(self) -> None:
        if not self.enabled:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.history, f)
        os.replace(tmp, self.path)

    def load(self, history: dict) -> None:
        for key in self.KEYS:
            if isinstance(history.get(key), list):
                self.history[key] = list(history[key])

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()

    def plot(self) -> None:
        """Loss / LR / grad-norm / throughput panel next to the log."""
        if not self.enabled or not self.history["steps"]:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        h = self.history
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 11))
        ax_loss, ax_lr, ax_gn, ax_tp = axes

        ax_loss.plot(h["steps"], h["train_loss"], label="train", lw=1)
        if h["val_steps"]:
            ax_loss.plot(h["val_steps"], h["val_loss"], marker="o", ms=3,
                         label="val", color="tab:red")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title(f"pretraining -- {self.run}")
        ax_loss.legend()

        ax_lr.plot(h["steps"], h["lr"], color="tab:green")
        ax_lr.set_ylabel("lr")

        ax_gn.plot(h["steps"], h["grad_norm"], color="tab:orange", lw=1)
        ax_gn.set_ylabel("grad norm")
        ax_gn.set_yscale("log")

        ax_tp.plot(h["steps"], h["tokens_per_sec"], color="tab:purple", lw=1)
        ax_tp.set_ylabel("tokens/sec")
        ax_tp.set_xlabel("optimizer step")

        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "loss_history.png"), dpi=110)
        plt.close(fig)
