import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def tiny_config():
    from nanollm.config import ModelConfig
    return ModelConfig(vocab_size=512, n_dim=64, n_layer=2, n_head=2,
                       n_kv_head=1, n_seq=64, loss_chunk_size=16)


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    from nanollm.model import Transformer
    torch.manual_seed(0)
    return Transformer.from_config(tiny_config)
