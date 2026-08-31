"""Legacy import shim -- the model now lives in ``nanollm.model``.

    from simple_transformer import Transformer   # still works
    from nanollm.model import Transformer        # preferred
"""

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.model import (  # noqa: F401
    DEFAULT_EMBEDDING_DIM, DEFAULT_NUM_HEADS, DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_EXPERTS, DEFAULT_SEQ_LEN, DEFAULT_TOP_K, IGNORE_INDEX,
    AttentionGQA, Block, FFN, KVCache, RMSNorm, SwiGLU, Transformer,
    apply_rope, build_document_mask, default_n_kv_head, precompute_rope,
    rotate_half, swiglu_hidden_dim,
)

__all__ = [
    "Transformer", "RMSNorm", "AttentionGQA", "SwiGLU", "FFN", "Block",
    "KVCache", "IGNORE_INDEX", "swiglu_hidden_dim", "build_document_mask",
    "precompute_rope", "apply_rope", "rotate_half", "default_n_kv_head",
    "DEFAULT_SEQ_LEN", "DEFAULT_EMBEDDING_DIM", "DEFAULT_NUM_HEADS",
    "DEFAULT_NUM_LAYERS", "DEFAULT_NUM_EXPERTS", "DEFAULT_TOP_K",
]
