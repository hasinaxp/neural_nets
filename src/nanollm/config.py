"""Typed configuration for the nanollm stack.

Every knob the training scripts read lives in one of the dataclasses below.
A run is fully described by `TrainConfig`, which serialises to YAML and is
embedded in every checkpoint, so a checkpoint always knows how it was made.

Precedence, lowest to highest:

    dataclass defaults  <  --config file.yaml  <  --set key=value  <  env vars

The legacy ``CONFIG`` dict is still exported (see ``config.py`` at the repo
root) so older call sites keep working.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

__all__ = [
    "ModelConfig",
    "DataConfig",
    "OptimConfig",
    "RuntimeConfig",
    "SFTConfig",
    "DPOConfig",
    "TrainConfig",
    "load_config",
    "apply_overrides",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture. Defaults describe the ~186M reference model.

    n_dim / n_head is chosen so head_dim == 64: the flash-attention kernels are
    tuned for head_dim in (32, 64, 128) and fall back to a much slower math
    kernel otherwise. 896/14 is the same head geometry as Qwen2-0.5B.
    """

    vocab_size: int = 32768        # padded to a multiple of 64 at build time
    n_dim: int = 896
    n_layer: int = 18
    n_head: int = 14               # head_dim = 896 / 14 = 64
    n_kv_head: int = 2             # GQA: 7 query heads share each KV head
    n_seq: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    z_loss_weight: float = 1e-4
    loss_chunk_size: int = 512
    activation_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_dim // self.n_head

    def validate(self) -> None:
        if self.n_dim % self.n_head:
            raise ValueError(f"n_dim={self.n_dim} not divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head:
            raise ValueError(
                f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

    def estimate_params(self) -> int:
        """Parameter count without building the model. Matches
        ``Transformer.get_param_count()`` exactly for the default topology."""
        d, L, hd = self.n_dim, self.n_layer, self.head_dim
        hidden = swiglu_hidden_dim(d)
        attn = 2 * d * d + 2 * d * self.n_kv_head * hd
        ffn = 3 * d * hidden
        norms = 2 * d + 2 * hd
        embed = self.vocab_size * d
        total = embed + L * (attn + ffn + norms) + d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total


def swiglu_hidden_dim(n_dim: int, multiple_of: int = 256) -> int:
    """8/3 * n_dim keeps a SwiGLU FFN param-matched to a 4x GELU FFN."""
    hidden = int(8 * n_dim / 3)
    return ((hidden + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Where tokens come from.

    The default path reads pre-tokenized uint16 shards written by
    ``scripts/prepare_data.py``. That keeps the BPE cost out of the training
    loop entirely -- tokenising is done once, not once per epoch.
    """

    data_dir: str = "dataset/tokens"
    tokenizer_file: str = "artifacts/tokenizer-32768.txt"
    raw_dir: str = "dataset/raw"

    # Fraction of shards held out for validation. Held-out shards are never
    # touched by the training sampler.
    val_shards: int = 1
    val_batches: int = 40

    num_workers: int = 4
    prefetch_factor: int = 4
    shuffle_buffer: int = 8192
    seed: int = 1337


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

@dataclass
class OptimConfig:
    # Global batch is micro_batch * grad_accum * world_size, in sequences.
    micro_batch_size: int = 16
    grad_accum_steps: int = 8
    max_steps: int = 24000

    peak_lr: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    schedule: str = "cosine"       # cosine | wsd | constant

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # WSD only: fraction of total steps spent decaying at the end.
    decay_frac: float = 0.1

    def tokens_per_step(self, n_seq: int, world_size: int = 1) -> int:
        return self.micro_batch_size * self.grad_accum_steps * world_size * n_seq


# ---------------------------------------------------------------------------
# Post-training
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Supervised fine-tuning on instruction / chat data.

    The LR is two orders of magnitude below pretraining. At 3e-5 for three
    epochs the weights walk far enough off the pretrained solution that general
    language modelling degrades -- the model gets fluent at the SFT formats and
    worse at everything else.
    """

    init_from: str = "artifacts/pretrain_model.pt"
    seq_len: int = 1024

    micro_batch_size: int = 16
    grad_accum_steps: int = 2
    epochs: int = 3
    max_steps: int = 0             # 0 -> derive from epochs

    peak_lr: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.05      # a longer ramp: the first steps out of a
                                   # converged checkpoint do the damage
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Pretraining replay (rehearsal). SFT's objective is narrow -- loss on
    # assistant tokens over a handful of task formats -- and nothing in it asks
    # the model to keep modelling ordinary text, so it drifts. Mixing plain
    # next-token batches back in keeps the original objective pulling on the
    # same weights.
    replay_frac: float = 0.25      # share of micro-batches drawn from pretraining
    replay_seq_len: int = 512      # a regularizer, not a second pretraining run
    replay_loss_weight: float = 1.0

    val_every: int = 250
    val_batches: int = 40


@dataclass
class DPOConfig:
    """Direct preference optimisation against a frozen reference policy."""

    init_from: str = "artifacts/sft_model.pt"
    seq_len: int = 1024

    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 1
    max_steps: int = 0

    beta: float = 0.1              # KL strength; higher = stays nearer the ref
    label_smoothing: float = 0.0   # cDPO: assume this share of labels are noise
    sft_loss_weight: float = 0.1   # NLL on the chosen reply. Pure DPO can push
                                   # both log-probs down as long as the margin
                                   # grows; this anchors the chosen branch.

    peak_lr: float = 5e-7
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    val_every: int = 100
    val_batches: int = 30


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    out_dir: str = "artifacts"
    log_dir: str = "logs"
    run_name: str = ""             # blank -> timestamp

    compile: bool = True
    compile_mode: str = "default"
    dtype: str = "bfloat16"        # bfloat16 | float16 | float32
    matmul_precision: str = "high"

    log_every: int = 50
    eval_every: int = 500
    sample_every: int = 1000
    ckpt_every: int = 500
    snapshot_every: int = 5000
    plot_every: int = 200

    # Progress/telemetry sinks. tensorboard and wandb are both optional
    # imports -- a missing package downgrades to a warning, never a crash.
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "nanollm"

    seed: int = 1337
    resume: str = "auto"           # auto | never | <path to checkpoint>


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def validate(self) -> None:
        self.model.validate()
        if self.optim.micro_batch_size < 1 or self.optim.grad_accum_steps < 1:
            raise ValueError("micro_batch_size and grad_accum_steps must be >= 1")
        if self.runtime.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unknown dtype {self.runtime.dtype!r}")
        if self.optim.schedule not in ("cosine", "wsd", "constant"):
            raise ValueError(f"unknown schedule {self.optim.schedule!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        try:
            import yaml
            return yaml.safe_dump(self.to_dict(), sort_keys=False)
        except ImportError:
            return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        cfg = cls()
        _merge_into(cfg, d)
        return cfg

    def summary(self, world_size: int = 1) -> str:
        p = self.model.estimate_params()
        tps = self.optim.tokens_per_step(self.model.n_seq, world_size)
        budget = tps * self.optim.max_steps
        return (
            f"params ~{p/1e6:.1f}M (non-embedding "
            f"{(p - self.model.vocab_size*self.model.n_dim)/1e6:.1f}M) | "
            f"head_dim {self.model.head_dim} | "
            f"tokens/step {tps:,} | budget {budget/1e9:.2f}B "
            f"({budget/max(1, p):.1f} tok/param, Chinchilla ~20)"
        )


# ---------------------------------------------------------------------------
# Loading / overriding
# ---------------------------------------------------------------------------

def _merge_into(obj: Any, updates: dict) -> None:
    """Recursively apply a nested dict onto a dataclass instance, in place."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in updates.items():
        if key not in valid:
            raise KeyError(
                f"unknown config key {key!r} for {type(obj).__name__}; "
                f"valid keys: {', '.join(sorted(valid))}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value)
        else:
            setattr(obj, key, _coerce(value, valid[key].type))


def _coerce(value: Any, target: Any) -> Any:
    """YAML gives strings for env-style overrides; make them match the field."""
    name = getattr(target, "__name__", str(target))
    if value is None or "Optional" in str(target):
        return value
    try:
        if name == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if name == "int":
            return int(value)
        if name == "float":
            return float(value)
        if name == "str":
            return str(value)
    except (TypeError, ValueError):
        raise ValueError(f"cannot coerce {value!r} to {name}")
    return value


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> TrainConfig:
    """Apply ``--set section.key=value`` strings, e.g. ``optim.peak_lr=3e-4``."""
    nested: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, _, value = item.partition("=")
        node = nested
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value.strip()
    if nested:
        _merge_into(cfg, nested)
    return cfg


def load_config(path: Optional[str] = None,
                overrides: Optional[list[str]] = None) -> TrainConfig:
    """Build a TrainConfig from an optional YAML file plus CLI overrides."""
    cfg = TrainConfig()
    if path:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _merge_into(cfg, raw)
    apply_overrides(cfg, overrides or [])

    # A handful of env vars stay honoured for cluster launchers that cannot
    # pass CLI flags through.
    env_map = {
        "MICRO_BATCH": ("optim", "micro_batch_size"),
        "GRAD_ACCUM": ("optim", "grad_accum_steps"),
        "MAX_STEPS": ("optim", "max_steps"),
        "NUM_WORKERS": ("data", "num_workers"),
        "USE_COMPILE": ("runtime", "compile"),
        "COMPILE_MODE": ("runtime", "compile_mode"),
        "OUT_DIR": ("runtime", "out_dir"),
    }
    for env, (section, key) in env_map.items():
        if env in os.environ:
            node = getattr(cfg, section)
            setattr(node, key, _coerce(os.environ[env],
                                       {f.name: f.type for f in fields(node)}[key]))

    cfg.validate()
    return cfg
