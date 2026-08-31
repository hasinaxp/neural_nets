"""End-to-end: shards -> training -> checkpoint -> resume -> generate."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from nanollm.data.shards import ShardWriter

REPO = os.path.join(os.path.dirname(__file__), "..")

COMMON = [
    "--set", "model.n_dim=64", "--set", "model.n_layer=2",
    "--set", "model.n_head=2", "--set", "model.n_kv_head=1",
    "--set", "model.n_seq=64", "--set", "model.loss_chunk_size=32",
    "--set", "optim.micro_batch_size=2", "--set", "optim.grad_accum_steps=2",
    "--set", "runtime.compile=false", "--set", "data.val_batches=2",
    "--set", "runtime.eval_every=4", "--set", "runtime.sample_every=1000",
    "--set", "runtime.plot_every=1000", "--set", "runtime.ckpt_every=4",
]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    tokens = root / "tokens"
    rng = np.random.default_rng(0)
    with ShardWriter(str(tokens), "tokens", vocab_size=320,
                     shard_tokens=4000) as w:
        w.add(rng.integers(0, 320, size=12000, dtype=np.int32))
    return root, tokens


def run_training(workspace, extra):
    root, tokens = workspace
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "nanollm.train.pretrain",
           "--set", f"data.data_dir={tokens}",
           "--set", f"runtime.out_dir={root/'art'}",
           "--set", f"runtime.log_dir={root/'logs'}",
           "--set", "model.vocab_size=320",
           "--set", f"data.tokenizer_file={REPO}/artifacts/tokenizer-32768.txt",
           ] + COMMON + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=REPO, env=env)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, "artifacts/tokenizer-32768.txt")),
    reason="needs the bundled tokenizer")
def test_train_then_resume(workspace):
    root, _ = workspace

    first = run_training(workspace, ["--set", "optim.max_steps=4"])
    assert first.returncode == 0, first.stderr[-3000:]
    ckpt = root / "art" / "pretrain_checkpoint_latest.pt"
    assert ckpt.exists()
    assert (root / "art" / "pretrain_model.pt").exists()

    step = torch.load(ckpt, weights_only=False)["global_step"]
    assert step == 4

    second = run_training(workspace, ["--set", "optim.max_steps=8"])
    assert second.returncode == 0, second.stderr[-3000:]
    assert "resumed from" in second.stderr or "resumed from" in second.stdout
    assert torch.load(ckpt, weights_only=False)["global_step"] == 8


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, "artifacts/tokenizer-32768.txt")),
    reason="needs the bundled tokenizer")
def test_checkpoint_carries_its_own_config(workspace):
    root, _ = workspace
    ck = torch.load(root / "art" / "pretrain_checkpoint_latest.pt",
                    weights_only=False)
    # A checkpoint must be loadable without the YAML that produced it.
    assert ck["config"]["model"]["n_dim"] == 64
    assert ck["config"]["model"]["n_layer"] == 2

    from nanollm.config import TrainConfig
    from nanollm.model import Transformer
    cfg = TrainConfig.from_dict(ck["config"])
    model = Transformer.from_config(cfg.model)
    model.load_state_dict(ck["model_state_dict"])

    out = model.generate(torch.tensor([[1, 2, 3]]), max_count=5, temperature=0.0)
    assert out.shape == (1, 8)
