"""Legacy config shim.

The real configuration lives in ``nanollm.config`` as typed dataclasses, and a
run is described by ``configs/base.yaml``. This module keeps the old ``CONFIG``
dict working for scripts that still read it.

    from config import CONFIG          # still works
    from nanollm.config import TrainConfig   # preferred
"""

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.config import (DataConfig, ModelConfig, OptimConfig,  # noqa: F401
                            RuntimeConfig, TrainConfig, load_config)

_DEFAULTS = ModelConfig()

CONFIG = {
    'vocab_size': _DEFAULTS.vocab_size,
    'embedding_dim': _DEFAULTS.n_dim,
    'n_layers': _DEFAULTS.n_layer,
    # 14 heads, not 16: 896/14 = 64, the head_dim the flash-attention kernels
    # are tuned for. 896/16 = 56 falls back to the much slower math kernel.
    'n_heads': _DEFAULTS.n_head,
    'n_kv_heads': _DEFAULTS.n_kv_head,
    'sequence_length': _DEFAULTS.n_seq,
    'seq_len': _DEFAULTS.n_seq,
}

__all__ = ["CONFIG", "TrainConfig", "ModelConfig", "DataConfig",
           "OptimConfig", "RuntimeConfig", "load_config"]
