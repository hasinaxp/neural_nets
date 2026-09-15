"""End-to-end SFT: cache -> tiny model -> per-task validation.

The per-task validation path is only exercised when a real run reaches its
first eval, so it is covered here rather than by unit tests on the pieces.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from nanollm.data.shards import ShardWriter

REPO = os.path.join(os.path.dirname(__file__), "..")
TOKENIZER = os.path.join(REPO, "artifacts/tokenizer-32768.txt")

TINY_MODEL = [
    "--set", "model.n_dim=64", "--set", "model.n_layer=2",
    "--set", "model.repeat_start=0", "--set", "model.repeat_end=0",
    "--set", "model.repeat_times=1",
    "--set", "model.n_head=2", "--set", "model.n_kv_head=1",
    "--set", "model.n_seq=128", "--set", "model.loss_chunk_size=32",
    "--set", "model.vocab_size=32768",
    "--set", "runtime.compile=false",
]

# Three tasks with obviously different shapes, so a per-task number that is
# identical across all of them would be a bug rather than a coincidence.
CONVERSATIONS = {
    "shell": ("Write a Linux command for this.\n\nlist files", "ls -la"),
    "writing": ("Draft this for me.\n\nA note to my neighbour about the bins.",
                "Hi! Just a quick note about the bin collection this week."),
    "rewrite": ("Rephrase the text below.\n\nthe cat sat on the mat",
                "The cat was sitting on the mat."),
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A pretrained checkpoint plus an SFT cache, both tiny."""
    root = tmp_path_factory.mktemp("sftrun")

    tokens = root / "tokens"
    rng = np.random.default_rng(0)
    with ShardWriter(str(tokens), "tokens", vocab_size=32768,
                     shard_tokens=4000) as w:
        w.add(rng.integers(0, 32768, size=12000, dtype=np.int32))

    cache = root / "cache"
    cache.mkdir()
    train_rows, val_rows = [], []
    for task, (prompt, reply) in CONVERSATIONS.items():
        messages = json.dumps([{"role": "user", "content": prompt},
                               {"role": "assistant", "content": reply}])
        for _ in range(64):
            train_rows.append({"source": task, "task": task,
                               "messages": messages})
        for _ in range(16):
            val_rows.append({"source": task, "task": task,
                             "messages": messages})
    pd.DataFrame(train_rows).to_parquet(cache / "sft_train.parquet")
    pd.DataFrame(val_rows).to_parquet(cache / "sft_val.parquet")
    return root, tokens, cache


def run_sft(workspace, extra):
    root, tokens, cache = workspace
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(REPO, "src") + os.pathsep
                         + env.get("PYTHONPATH", ""))
    # CPU only. These are 64-dim models that gain nothing from a GPU, and the
    # machine this runs on may well have a real pretraining run holding most of
    # the card -- a test has no business competing with it for memory.
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, "-m", "nanollm.train.sft",
           "--set", f"data.data_dir={tokens}",
           "--set", f"data.tokenizer_file={TOKENIZER}",
           "--set", f"runtime.out_dir={root/'art'}",
           "--set", f"runtime.log_dir={root/'logs'}",
           ] + TINY_MODEL + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=REPO, env=env)


@pytest.fixture(scope="module")
def pretrained(workspace):
    root, tokens, _ = workspace
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(REPO, "src") + os.pathsep
                         + env.get("PYTHONPATH", ""))
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, "-m", "nanollm.train.pretrain",
           "--set", f"data.data_dir={tokens}",
           "--set", f"data.tokenizer_file={TOKENIZER}",
           "--set", f"runtime.out_dir={root/'art'}",
           "--set", f"runtime.log_dir={root/'logs'}",
           "--set", "optim.max_steps=2", "--set", "optim.micro_batch_size=2",
           "--set", "optim.grad_accum_steps=1", "--set", "data.val_batches=1",
           "--set", "runtime.eval_every=2", "--set", "runtime.ckpt_every=2",
           "--set", "runtime.sample_every=1000",
           "--set", "runtime.plot_every=1000"] + TINY_MODEL
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                            cwd=REPO, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    path = root / "art" / "pretrain_model.pt"
    assert path.exists(), f"no checkpoint written: {result.stdout[-2000:]}"
    return path


@pytest.mark.skipif(not os.path.exists(TOKENIZER),
                    reason="needs the bundled tokenizer")
def test_sft_reports_per_task_validation(workspace, pretrained):
    root, _, cache = workspace
    result = run_sft(workspace, [
        "--set", f"data.sft_cache_dir={cache}",
        "--set", f"sft.init_from={pretrained}",
        "--set", "sft.seq_len=128", "--set", "sft.micro_batch_size=2",
        "--set", "sft.grad_accum_steps=1", "--set", "sft.max_steps=4",
        "--set", "sft.val_every=2", "--set", "sft.val_batches=2",
        "--set", "sft.val_task_batches=2", "--set", "sft.replay_frac=0.0",
        "--set", "sft.mixture_examples=48",
    ])
    assert result.returncode == 0, result.stderr[-4000:]
    out = result.stdout + result.stderr

    assert "per-task validation:" in out, out[-3000:]
    for task in CONVERSATIONS:
        assert f"{task}=" in out or f"{task:15s}" in out.replace("  ", " "), \
            f"{task} never reported\n{out[-3000:]}"


@pytest.mark.skipif(not os.path.exists(TOKENIZER),
                    reason="needs the bundled tokenizer")
def test_sft_runs_with_per_task_validation_disabled(workspace, pretrained):
    root, _, cache = workspace
    result = run_sft(workspace, [
        "--set", f"data.sft_cache_dir={cache}",
        "--set", f"sft.init_from={pretrained}",
        "--set", "sft.seq_len=128", "--set", "sft.micro_batch_size=2",
        "--set", "sft.grad_accum_steps=1", "--set", "sft.max_steps=2",
        "--set", "sft.val_every=2", "--set", "sft.val_batches=1",
        "--set", "sft.val_task_batches=0", "--set", "sft.replay_frac=0.0",
        "--set", "sft.mixture_examples=48",
    ])
    assert result.returncode == 0, result.stderr[-4000:]
    assert "per-task validation:" not in result.stdout
