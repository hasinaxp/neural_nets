"""nanollm -- a compact, single-file-per-concern LLM training stack.

The reference model is ~186M parameters (896 dim, 18 layers, 14 heads,
head_dim 64, GQA 7:1, 32768 vocab, 2048 context).
"""

__version__ = "0.1.0"

from .config import (DataConfig, ModelConfig, OptimConfig, RuntimeConfig,
                     TrainConfig, load_config)
from .model import Transformer, IGNORE_INDEX
from .tokenizer import Tokenizer

__all__ = [
    "TrainConfig", "ModelConfig", "DataConfig", "OptimConfig", "RuntimeConfig",
    "load_config", "Transformer", "Tokenizer", "IGNORE_INDEX", "__version__",
]
