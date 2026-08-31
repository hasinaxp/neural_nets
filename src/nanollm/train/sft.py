"""Supervised fine-tuning.

    python -m nanollm.train.sft --config configs/base.yaml
    torchrun --standalone --nproc_per_node=8 -m nanollm.train.sft --config configs/base.yaml

Loss is computed on assistant tokens only. A share of micro-batches is drawn
from the pretraining corpus instead (see ``SFTConfig.replay_frac``) to stop the
model forgetting how to model ordinary text.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
import time
from datetime import timedelta
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import TrainConfig, load_config
from ..data.loader import BatchStream, TokenCorpus
from ..data.shards import ShardIndex
from ..data.sft import SFTDataset, render_conversation
from ..model import Transformer
from ..utils.checkpoint import atomic_save, save_checkpoint, unwrap_model
from ..utils.distributed import (all_reduce_mean, cleanup_distributed,
                                 setup_distributed)
from ..utils.logging import MetricLogger, run_id, setup_logging
from ..utils.schedules import make_lr_fn
from .common import (build_optimizer, clip_and_step, configure_backends,
                     load_pretrained, load_tokenizer, pad_batch, peak_flops)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanollm.train.sft", description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--init-from", default=None,
                   help="checkpoint to fine-tune (default: sft.init_from)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


class ConversationStream:
    """Yields padded (xs, ys) micro-batches of rendered conversations.

    Rendering is cheap relative to a training step, so it runs inline; there is
    no worker pool to keep in sync with the training position on resume.
    """

    def __init__(self, dataset: SFTDataset, tokenizer, seq_len: int,
                 micro_batch_size: int, pad_id: int, rank: int = 0,
                 world_size: int = 1, epochs: int = 1, seed: int = 1337):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.pad_id = pad_id
        self.rank = rank
        self.world_size = world_size
        self.epochs = epochs
        self.seed = seed
        self.dropped = 0
        self.rendered = 0

    def _examples(self) -> Iterator[dict]:
        order = list(range(len(self.dataset)))
        for epoch in range(self.epochs):
            random.Random(self.seed + epoch).shuffle(order)
            # Each rank takes a disjoint stride of the batches, so no example
            # is seen twice per epoch across the world.
            for i in order[self.rank::self.world_size]:
                yield from self.dataset[i]

    def __iter__(self):
        rows: list[tuple[list[int], list[int]]] = []
        for example in self._examples():
            ids, mask = render_conversation(
                self.tokenizer, example["messages"], seq_len=self.seq_len)
            if ids is None or not any(mask):
                self.dropped += 1
                continue
            rows.append((ids, mask))
            self.rendered += 1
            if len(rows) == self.micro_batch_size:
                yield pad_batch(rows, self.pad_id)
                rows = []
        if rows:
            yield pad_batch(rows, self.pad_id)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.init_from:
        cfg.sft.init_from = args.init_from
    if args.max_steps is not None:
        cfg.sft.max_steps = args.max_steps
    if args.dry_run:
        cfg.sft.max_steps = 3
        cfg.runtime.compile = False

    env = setup_distributed()
    run = run_id(cfg.runtime.run_name)
    if env.enabled:
        obj = [run]
        dist.broadcast_object_list(obj, src=0)
        run = obj[0]

    log = setup_logging(cfg.runtime.log_dir, run, "sft", env.is_main, env.rank)
    device = env.device
    torch.manual_seed(cfg.runtime.seed + env.rank)

    amp_dtype = configure_backends(cfg, device)
    use_scaler = device.type == "cuda" and amp_dtype is torch.float16

    log.info(f"[0] SFT run {run} | device {device} | world size {env.world_size}")

    log.info("[1] Tokenizer")
    tokenizer, model_vocab, eos_id = load_tokenizer(cfg, log)
    cfg.model.vocab_size = model_vocab
    pad_id = tokenizer.special_tokens.get("<|PAD|>", eos_id)

    log.info("[2] Model")
    model = Transformer.from_config(cfg.model).to(device)
    load_pretrained(model, cfg.sft.init_from, device, log)
    log.info(f"    {model.get_param_count():,} parameters")

    log.info("[3] Data")
    train_ds = SFTDataset(batch_size=cfg.sft.micro_batch_size, split="train",
                          epochs=1, seed=cfg.data.seed)
    val_ds = SFTDataset(batch_size=cfg.sft.micro_batch_size, split="val",
                        seed=cfg.data.seed)
    for line in train_ds.describe().splitlines():
        log.info(f"    {line}")

    train_stream = ConversationStream(
        train_ds, tokenizer, cfg.sft.seq_len, cfg.sft.micro_batch_size, pad_id,
        rank=env.rank, world_size=env.world_size, epochs=cfg.sft.epochs,
        seed=cfg.data.seed)
    val_stream = ConversationStream(
        val_ds, tokenizer, cfg.sft.seq_len, cfg.sft.micro_batch_size, pad_id,
        epochs=1, seed=cfg.data.seed)

    val_batches = []
    for batch in val_stream:
        val_batches.append((batch[0].to(device), batch[1].to(device)))
        if len(val_batches) >= cfg.sft.val_batches:
            break
    log.info(f"    {len(val_batches)} fixed validation batches")

    # -- pretraining replay -------------------------------------------------
    replay_iter = None
    if cfg.sft.replay_frac > 0:
        try:
            index = ShardIndex.load(cfg.data.data_dir)
            corpus = TokenCorpus(index.paths, cfg.sft.replay_seq_len)
            replay_iter = iter(BatchStream(
                corpus, cfg.sft.micro_batch_size, seed=cfg.data.seed + 7,
                rank=env.rank, world_size=env.world_size,
                pin_memory=device.type == "cuda"))
            log.info(f"    replay: {cfg.sft.replay_frac:.0%} of micro-batches "
                     f"at seq_len {cfg.sft.replay_seq_len}")
        except Exception as e:
            log.warning(f"    replay disabled ({e}); the model will drift more")

    # -- steps --------------------------------------------------------------
    micro_per_step = cfg.sft.grad_accum_steps
    if cfg.sft.max_steps > 0:
        max_steps = cfg.sft.max_steps
    else:
        per_epoch_examples = len(train_ds.index) / max(1, env.world_size)
        max_steps = max(1, int(per_epoch_examples * cfg.sft.epochs
                               / (cfg.sft.micro_batch_size * micro_per_step)))
    log.info(f"    {max_steps:,} optimizer steps over {cfg.sft.epochs} epoch(s)")

    optimizer = build_optimizer(
        model, cfg.sft.peak_lr, cfg.sft.weight_decay,
        (cfg.optim.beta1, cfg.optim.beta2), cfg.optim.eps, device, log)
    scaler = torch.amp.GradScaler(enabled=use_scaler) if use_scaler else None
    lr_at = make_lr_fn(cfg.sft.peak_lr, max_steps,
                       max(1, int(cfg.sft.warmup_frac * max_steps)),
                       cfg.sft.min_lr_ratio, "cosine")

    metrics = MetricLogger(cfg.runtime.log_dir, run, enabled=env.is_main,
                           tensorboard=cfg.runtime.tensorboard,
                           wandb=cfg.runtime.wandb,
                           wandb_project=cfg.runtime.wandb_project,
                           config=cfg.to_dict(), logger=log)
    metrics.path = os.path.join(cfg.runtime.log_dir, "sft_metrics.json")

    if cfg.runtime.compile:
        try:
            torch._dynamo.config.suppress_errors = True
            # Variable-length SFT batches change shape every step, so this graph
            # must be dynamic or dynamo recompiles until it gives up.
            model.calculate_loss = torch.compile(
                model.calculate_loss, mode=cfg.runtime.compile_mode, dynamic=True)
            log.info("    torch.compile on (dynamic shapes)")
        except Exception as e:
            log.warning(f"    torch.compile failed, running eager: {e}")

    train_model = model
    if env.enabled:
        train_model = DDP(model, device_ids=[env.local_rank]
                          if device.type == "cuda" else None,
                          gradient_as_bucket_view=True)

    @torch.no_grad()
    def evaluate() -> float:
        train_model.eval()
        total = torch.zeros((), device=device)
        for xs, ys in val_batches:
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                total += model.calculate_loss(xs, ys).detach()
        train_model.train()
        return float(all_reduce_mean(total / max(1, len(val_batches)), env).item())

    log.info("[4] Training")
    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    all_params = [p for p in model.parameters() if p.requires_grad]
    replay_rng = random.Random(cfg.data.seed + env.rank)

    out_dir = cfg.runtime.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "sft_checkpoint_latest.pt")
    final_path = os.path.join(out_dir, "sft_model.pt")

    global_step = 0
    accum_micro = 0
    accum_loss = torch.zeros((), device=device)
    n_replay = 0
    started = time.time()
    step_timer = time.time()
    last_val = float("nan")

    pbar = None
    if env.is_main:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_steps, desc="sft", unit="step", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for xs, ys in train_stream:
            if global_step >= max_steps:
                break

            use_replay = (replay_iter is not None
                          and replay_rng.random() < cfg.sft.replay_frac)
            weight = cfg.sft.replay_loss_weight if use_replay else 1.0
            if use_replay:
                rx, ry = next(replay_iter)
                xs, ys = rx.to(device, non_blocking=True), ry.to(device, non_blocking=True)
                n_replay += 1
            else:
                xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)

            is_last_micro = (accum_micro + 1) == micro_per_step
            sync_ctx = (train_model.no_sync()
                        if env.enabled and not is_last_micro
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    loss = model.calculate_loss(xs, ys) * weight / micro_per_step
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            accum_loss += loss.detach()
            accum_micro += 1
            if not is_last_micro:
                continue

            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm, _finite = clip_and_step(
                all_params, optimizer, cfg.sft.grad_clip, scaler)

            step_loss = float(accum_loss.item())
            accum_loss.zero_()
            accum_micro = 0
            global_step += 1

            step_time = time.time() - step_timer
            step_timer = time.time()
            metrics.log_train(global_step, step_loss, lr, float(grad_norm.item()),
                              cfg.sft.micro_batch_size * micro_per_step / max(1e-6, step_time))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"})

            if global_step % cfg.runtime.log_every == 0:
                recent = metrics.history["train_loss"][-cfg.runtime.log_every:]
                avg = sum(recent) / max(1, len(recent))
                log.info(f"    step {global_step}/{max_steps} | loss {avg:.4f} "
                         f"| ppl {math.exp(min(20, avg)):.2f} | lr {lr:.2e} "
                         f"| replay {n_replay}/{global_step*micro_per_step}")

            if global_step % cfg.sft.val_every == 0:
                last_val = evaluate()
                metrics.log_val(global_step, last_val)
                log.info(f"    >> val loss {last_val:.4f} | "
                         f"ppl {math.exp(min(20, last_val)):.2f}")

            if global_step % cfg.runtime.ckpt_every == 0 and env.is_main:
                save_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                                step=global_step, config=cfg.to_dict(),
                                history=metrics.history, scaler=scaler,
                                extra={"val_loss": last_val, "stage": "sft"})
                metrics.save()
    finally:
        if pbar is not None:
            pbar.close()

    if accum_micro:
        optimizer.zero_grad(set_to_none=True)

    last_val = evaluate()
    metrics.log_val(global_step, last_val)
    log.info(f"Final val loss {last_val:.4f} | ppl {math.exp(min(20, last_val)):.2f}")
    log.info(f"    rendered {train_stream.rendered:,} conversations, "
             f"dropped {train_stream.dropped:,} (too long to fit)")

    if env.is_main:
        save_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                        step=global_step, config=cfg.to_dict(),
                        history=metrics.history, scaler=scaler,
                        extra={"val_loss": last_val, "stage": "sft"})
        atomic_save(unwrap_model(model).state_dict(), final_path)
        metrics.plot()
        metrics.save()
        log.info(f"Model saved to {final_path}")

    log.info(f"Done in {timedelta(seconds=int(time.time() - started))} "
             f"| steps {global_step}")
    metrics.close()
    cleanup_distributed(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
