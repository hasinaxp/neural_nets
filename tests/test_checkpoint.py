import os

import pytest
import torch

from nanollm.config import ModelConfig
from nanollm.model import Transformer
from nanollm.utils.checkpoint import (ArchitectureMismatch, atomic_save,
                                      load_checkpoint, save_checkpoint,
                                      unwrap_model)


@pytest.fixture
def small():
    return ModelConfig(vocab_size=128, n_dim=32, n_layer=2, n_head=2,
                       n_kv_head=1, n_seq=32)


def test_atomic_save_leaves_no_temp_file(tmp_path):
    path = tmp_path / "x.pt"
    atomic_save({"a": 1}, str(path))
    assert path.exists() and not (tmp_path / "x.pt.tmp").exists()


def test_save_load_roundtrip_restores_weights(tmp_path, small):
    torch.manual_seed(0)
    model = Transformer.from_config(small)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    idx = torch.randint(0, small.vocab_size, (2, 9))
    model.calculate_loss(idx[:, :-1], idx[:, 1:]).backward()
    opt.step()

    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model=model, optimizer=opt, step=7,
                    config={"model": small.__dict__}, history={"steps": [1]})

    restored = Transformer.from_config(small)
    ck = load_checkpoint(path, model=restored)
    assert ck["global_step"] == 7
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_load_rejects_mismatched_architecture(tmp_path, small):
    import dataclasses
    model = Transformer.from_config(small)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model=model, optimizer=opt, step=0, config={})

    wider = dataclasses.replace(small, n_dim=64)
    with pytest.raises(ArchitectureMismatch):
        load_checkpoint(path, model=Transformer.from_config(wider))


def test_state_dict_saved_without_wrapper_prefixes(tmp_path, small):
    """DDP/compile prefixes in checkpoint keys break every other loader."""
    model = Transformer.from_config(small)

    class FakeWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    wrapped = FakeWrapper(model)
    assert unwrap_model(wrapped) is model

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model=wrapped, optimizer=opt, step=0, config={})
    keys = torch.load(path, weights_only=False)["model_state_dict"].keys()
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in keys)


def test_load_pretrained_strips_prefixes(tmp_path, small):
    from nanollm.train.common import load_pretrained
    import logging

    model = Transformer.from_config(small)
    prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = str(tmp_path / "raw.pt")
    atomic_save(prefixed, path)

    target = Transformer.from_config(small)
    load_pretrained(target, path, torch.device("cpu"), logging.getLogger("t"))
    for a, b in zip(model.parameters(), target.parameters()):
        assert torch.equal(a, b)
