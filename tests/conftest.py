import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def tiny_config():
    from nanollm.config import ModelConfig
    # repeat_times=1: the general-behaviour tests want a plain stack, so the
    # looped schedule is exercised by its own fixture rather than silently
    # underpinning every other assertion.
    return ModelConfig(vocab_size=512, n_dim=64, n_layer=2, n_head=2,
                       n_kv_head=1, n_seq=64, loss_chunk_size=16,
                       repeat_start=0, repeat_end=0, repeat_times=1)


@pytest.fixture(scope="session")
def looped_config():
    """4 blocks with the middle two run twice: schedule (0, 1, 2, 1, 2, 3)."""
    from nanollm.config import ModelConfig
    return ModelConfig(vocab_size=512, n_dim=64, n_layer=4, n_head=2,
                       n_kv_head=1, n_seq=64, loss_chunk_size=16,
                       repeat_start=1, repeat_end=3, repeat_times=2)


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    from nanollm.model import Transformer
    torch.manual_seed(0)
    return Transformer.from_config(tiny_config)
