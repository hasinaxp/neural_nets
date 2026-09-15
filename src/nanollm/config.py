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
    "layer_schedule",
    "swiglu_hidden_dim",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture. Defaults describe the ~169M-parameter reference model,
    16 unique layers executed 24 times.

    n_dim / n_head is chosen so head_dim == 64: the flash-attention kernels are
    tuned for head_dim in (32, 64, 128) and fall back to a much slower math
    kernel otherwise. 896/14 is the same head geometry as Qwen2-0.5B.

    ``repeat_*`` loops a contiguous middle group of blocks: the group runs once,
    then its own output is fed back through the *same* weights again. Depth is
    what the loss curve responds to most strongly at this scale, and a looped
    group buys it at zero parameter cost -- the extra depth is free in VRAM for
    the weights and in inference memory, and paid for only in FLOPs and
    activations. Layers next to the embedding and next to the output head stay
    single-pass: those two ends do format conversion rather than the iterative
    refinement that rewards looping.
    """

    vocab_size: int = 32768        # padded to a multiple of 64 at build time
    n_dim: int = 896
    n_layer: int = 16              # unique blocks (weight sets)
    n_head: int = 14               # head_dim = 896 / 14 = 64
    n_kv_head: int = 2             # GQA: 7 query heads share each KV head
    n_seq: int = 2048

    # Looped middle group: blocks [repeat_start, repeat_end) run repeat_times
    # times in sequence. 4..11 twice -> 4 + 8 + 8 + 4 = 24 executed layers.
    repeat_start: int = 4
    repeat_end: int = 12           # exclusive
    repeat_times: int = 2          # 1 disables looping entirely

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

    @property
    def layer_schedule(self) -> tuple[int, ...]:
        """Block indices in execution order; len() is the effective depth."""
        return layer_schedule(self.n_layer, self.repeat_start,
                              self.repeat_end, self.repeat_times)

    @property
    def n_executed_layer(self) -> int:
        return len(self.layer_schedule)

    def validate(self) -> None:
        if self.n_dim % self.n_head:
            raise ValueError(f"n_dim={self.n_dim} not divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head:
            raise ValueError(
                f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")
        validate_repeat(self.n_layer, self.repeat_start, self.repeat_end,
                        self.repeat_times)

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


def validate_repeat(n_layer: int, start: int, end: int, times: int) -> None:
    if times < 1:
        raise ValueError(f"repeat_times={times} must be >= 1")
    if not 0 <= start <= end <= n_layer:
        raise ValueError(
            f"repeat span [{start}, {end}) is not inside [0, {n_layer})")
    if times > 1 and start == end:
        raise ValueError(
            f"repeat_times={times} with an empty span [{start}, {end}) repeats "
            f"nothing; set repeat_times=1 to disable looping")


def layer_schedule(n_layer: int, start: int, end: int,
                   times: int) -> tuple[int, ...]:
    """Block indices in execution order.

    The middle group is looped as a *group* -- 0 1 2 3 | 4..11 | 4..11 | 12..15
    -- not per layer (4 4 5 5 ...). The whole group seeing its own output is
    what makes the second pass a refinement step rather than a wider layer.
    """
    validate_repeat(n_layer, start, end, times)
    middle = list(range(start, end))
    return tuple(list(range(start)) + middle * times + list(range(end, n_layer)))


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
    # Where build_sft_cache() writes its normalised parquet. Configurable so a
    # run can be pointed at an alternative mixture (or a test at a throwaway
    # one) without editing module constants.
    sft_cache_dir: str = "dataset/sft/cache"

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
    max_steps: int = 38400

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
    worse at everything else. Default is one epoch: past 3-epoch runs showed val
    loss flat after the first (1.3986 -> 1.3968 over the last ~900 steps) while
    epochs 2-3 kept eroding generality.
    """

    init_from: str = "artifacts/pretrain_model.pt"
    # Matches the pretrained n_seq. At 1024 the back half of the context never
    # saw a chat template, a role marker, or an EOS during fine-tuning, so
    # long-form behaviour past 1024 tokens was whatever pretraining left there.
    # It was also the reason long multi-turn chats were dropped at render time
    # and why xsum had to stay disabled.
    seq_len: int = 2048

    # Halved to hold activation memory flat at 2x the sequence length; the
    # global batch (micro_batch * grad_accum = 32) is unchanged, so the loss
    # curve is comparable to the 1024-ctx runs.
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 1
    max_steps: int = 0             # 0 -> derive from epochs
    # Examples drawn per epoch after task resampling. 0 -> the dataset default
    # (sft.DEFAULT_EPOCH_EXAMPLES). This is the real length knob for an SFT
    # run: max_steps only truncates, whereas this changes the mixture the LR
    # schedule is fitted to.
    mixture_examples: int = 0

    peak_lr: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.05      # a longer ramp: the first steps out of a
                                   # converged checkpoint do the damage
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # -- loss weighting ------------------------------------------------------
    # The plain token mean does not spend the gradient where the mixture says
    # it should. Both knobs below are measured corrections; see LossWeighting
    # in nanollm.train.common for the numbers behind them. Validation is always
    # scored unweighted, so val loss stays comparable across these settings.
    #
    # Share of each example's loss mass pinned on its closing EOS. One token in
    # a 155-token reply is 0.6% of that example's gradient, which is why long
    # replies stop badly (P(EOS) 0.55 past 257 tokens vs 0.84 under 16) and the
    # model runs to the length cap instead of ending. Only ever an upweight:
    # short replies already stop at ~0.99 and are left alone. 0 disables.
    eos_loss_share: float = 0.02
    # Ceiling on that weight, so a 600-token chat reply cannot hand one
    # position an unbounded share and turn the model trigger-happy about
    # stopping early.
    eos_loss_cap: float = 12.0
    # "token": the historical behaviour -- every supervised token counts once,
    # so an example's pull scales with its reply length and TASK_WEIGHTS ends
    # up describing the example mix while the gradient follows the token mix
    # (chat 78% of the gradient against a nominal 24%; extractive_qa 0.4%
    # against 12%). "example": normalise each example to equal mass, so the
    # delivered gradient matches the mixture that was actually designed.
    loss_normalize: str = "token"

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
    # Batches of held-out data evaluated *per task*, on top of the aggregate
    # above. The aggregate is a weighted average over ten tasks, so a task that
    # never learns can sit inside a perfectly healthy-looking curve; the small
    # ones (shell, rewrite, writing) are the likeliest to be in that position
    # and the ones the mixture would need retuning for. 0 disables.
    val_task_batches: int = 4


@dataclass
class DPOConfig:
    """Direct preference optimisation against a frozen reference policy."""

    init_from: str = "artifacts/sft_model.pt"
    seq_len: int = 2048          # tracks sft.seq_len; DPO must see the same
                                 # context the policy was fine-tuned in

    # DPO runs both branches of every pair through the model, so a micro-batch
    # of N pairs is 2N sequences. Halved alongside the doubled context.
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    epochs: int = 1
    max_steps: int = 0
    mixture_pairs: int = 0         # 0 -> dpo.DEFAULT_EPOCH_PAIRS

    # Sequence log-probs are averaged over reply tokens, not summed: a summed
    # logratio grows with reply length, which made the gradient blow past
    # grad_clip on every step (norm ~18 vs clip 1.0) and biased the policy
    # toward longer replies. With per-token logratios beta has to rise by
    # roughly the mean reply length to keep the same preference signal.
    length_normalize: bool = True
    beta: float = 2.0             # KL strength; higher = stays nearer the ref
    label_smoothing: float = 0.0   # cDPO: assume this share of labels are noise
    sft_loss_weight: float = 0.1   # NLL on the chosen reply. Pure DPO can push
                                   # both log-probs down as long as the margin
                                   # grows; this anchors the chosen branch.

    peak_lr: float = 5e-6
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
            f"depth {self.model.n_layer} unique / "
            f"{self.model.n_executed_layer} executed | "
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
