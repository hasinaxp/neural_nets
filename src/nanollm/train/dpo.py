"""Direct preference optimisation.

    python -m nanollm.train.dpo --config configs/base.yaml
    torchrun --standalone --nproc_per_node=8 -m nanollm.train.dpo --config configs/base.yaml

Optimises

    -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

where pi/ref are summed log-probs of the chosen/rejected reply under the policy
and the frozen reference. The reference is a copy of the initial policy: it
never updates, and it is what keeps the policy from wandering arbitrarily far
in pursuit of margin.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import random
import sys
import time
from datetime import timedelta
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import load_config
from ..data.dpo import DPODataset, render_pair
from ..model import IGNORE_INDEX, Transformer
from ..utils.checkpoint import atomic_save, save_checkpoint, unwrap_model
from ..utils.distributed import (all_reduce_mean, cleanup_distributed,
                                 setup_distributed)
from ..utils.logging import MetricLogger, run_id, setup_logging
from ..utils.schedules import make_lr_fn
from .common import (build_optimizer, clip_and_step, configure_backends,
                     load_pretrained, load_tokenizer, peak_flops)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanollm.train.dpo", description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--init-from", default=None,
                   help="SFT checkpoint to start from (default: dpo.init_from)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def build_pair_batch(rows, pad_id: int):
    """Stack chosen and rejected branches into one (2B, T) batch.

    Both branches go through the model in a single forward pass -- separate
    passes would double the launch overhead and, under autocast, can give
    slightly different numerics for the shared prompt prefix.

    Returns (xs, ys, n_pairs). ys is IGNORE_INDEX everywhere except the reply.
    """
    seqs, targets = [], []
    for prompt_ids, chosen_ids, rejected_ids in rows:
        for reply in (chosen_ids, rejected_ids):
            ids = prompt_ids + reply
            # Loss on the reply only: the prompt is identical in both branches,
            # so including it adds the same constant to each and just adds noise.
            mask = [0] * len(prompt_ids) + [1] * len(reply)
            seqs.append(ids)
            targets.append(mask)

    width = max(len(s) for s in seqs)
    tokens = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(seqs, targets)):
        row = torch.tensor(ids, dtype=torch.long)
        tokens[i, :len(ids)] = row
        keep = torch.tensor(mask, dtype=torch.bool)
        labels[i, :len(ids)] = torch.where(
            keep, row, torch.full_like(row, IGNORE_INDEX))

    return tokens[:, :-1].contiguous(), labels[:, 1:].contiguous(), len(rows)


def sequence_logprobs(model, xs, ys) -> torch.Tensor:
    """Sum of log p(target) over unmasked positions, one value per sequence.

    Computed in chunks over the sequence for the same reason the pretraining
    loss is: a (B, T, vocab) fp32 logit tensor is the largest thing in the step.
    """
    hidden = model.forward_hidden(xs)
    chunk = model.loss_chunk_size or hidden.size(1)
    total = torch.zeros(xs.size(0), device=xs.device, dtype=torch.float32)
    for i in range(0, hidden.size(1), chunk):
        h = hidden[:, i:i + chunk]
        t = ys[:, i:i + chunk]
        logits = model.logit_proj(h).float()
        valid = t != IGNORE_INDEX
        safe = t.masked_fill(~valid, 0)
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        total = total + (picked * valid).sum(dim=-1)
    return total


def dpo_loss(policy_lp, ref_lp, beta: float, label_smoothing: float = 0.0):
    """policy_lp / ref_lp are (2B,) with chosen at even indices."""
    pi_c, pi_r = policy_lp[0::2], policy_lp[1::2]
    ref_c, ref_r = ref_lp[0::2], ref_lp[1::2]

    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    if label_smoothing > 0:
        # cDPO: treat a fraction of preference labels as flipped noise.
        loss = (-F.logsigmoid(logits) * (1 - label_smoothing)
                - F.logsigmoid(-logits) * label_smoothing)
    else:
        loss = -F.logsigmoid(logits)

    stats = {
        "margin": (pi_c - pi_r).mean().detach(),
        "reward_chosen": (beta * (pi_c - ref_c)).mean().detach(),
        "reward_rejected": (beta * (pi_r - ref_r)).mean().detach(),
        "accuracy": (logits > 0).float().mean().detach(),
    }
    return loss.mean(), stats


class PairStream:
    """Yields (xs, ys, n_pairs) micro-batches of preference pairs."""

    def __init__(self, dataset: DPODataset, tokenizer, seq_len: int,
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
            for i in order[self.rank::self.world_size]:
                yield from self.dataset[i]

    def __iter__(self):
        rows = []
        for example in self._examples():
            prompt_ids, chosen_ids, rejected_ids = render_pair(
                self.tokenizer, example["prompt"], example["chosen"],
                example["rejected"], seq_len=self.seq_len)
            if prompt_ids is None:
                self.dropped += 1
                continue
            rows.append((prompt_ids, chosen_ids, rejected_ids))
            self.rendered += 1
            if len(rows) == self.micro_batch_size:
                yield build_pair_batch(rows, self.pad_id)
                rows = []
        if rows:
            yield build_pair_batch(rows, self.pad_id)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.init_from:
        cfg.dpo.init_from = args.init_from
    if args.max_steps is not None:
        cfg.dpo.max_steps = args.max_steps
    if args.dry_run:
        cfg.dpo.max_steps = 3
        cfg.runtime.compile = False

    env = setup_distributed()
    run = run_id(cfg.runtime.run_name)
    if env.enabled:
        obj = [run]
        dist.broadcast_object_list(obj, src=0)
        run = obj[0]

    log = setup_logging(cfg.runtime.log_dir, run, "dpo", env.is_main, env.rank)
    device = env.device
    torch.manual_seed(cfg.runtime.seed + env.rank)

    amp_dtype = configure_backends(cfg, device)
    use_scaler = device.type == "cuda" and amp_dtype is torch.float16

    log.info(f"[0] DPO run {run} | device {device} | world size {env.world_size}")

    log.info("[1] Tokenizer")
    tokenizer, model_vocab, eos_id = load_tokenizer(cfg, log)
    cfg.model.vocab_size = model_vocab
    pad_id = tokenizer.special_tokens.get("<|PAD|>", eos_id)

    log.info("[2] Policy and reference")
    model = Transformer.from_config(cfg.model).to(device)
    load_pretrained(model, cfg.dpo.init_from, device, log)

    # The reference is a frozen copy of the starting policy. eval() matters as
    # much as requires_grad_(False): dropout in the reference would make the
    # implicit reward noisy on every step.
    ref_model = copy.deepcopy(model).to(device)
    ref_model.requires_grad_(False)
    ref_model.eval()
    log.info(f"    policy {model.get_param_count():,} params | reference frozen")

    log.info("[3] Data")
    train_ds = DPODataset(batch_size=cfg.dpo.micro_batch_size, split="train",
                          epochs=1, seed=cfg.data.seed)
    val_ds = DPODataset(batch_size=cfg.dpo.micro_batch_size, split="val",
                        seed=cfg.data.seed)
    for line in train_ds.describe().splitlines():
        log.info(f"    {line}")

    train_stream = PairStream(train_ds, tokenizer, cfg.dpo.seq_len,
                              cfg.dpo.micro_batch_size, pad_id, rank=env.rank,
                              world_size=env.world_size, epochs=cfg.dpo.epochs,
                              seed=cfg.data.seed)
    val_stream = PairStream(val_ds, tokenizer, cfg.dpo.seq_len,
                            cfg.dpo.micro_batch_size, pad_id, epochs=1,
                            seed=cfg.data.seed)
    val_batches = []
    for batch in val_stream:
        val_batches.append((batch[0].to(device), batch[1].to(device)))
        if len(val_batches) >= cfg.dpo.val_batches:
            break
    log.info(f"    {len(val_batches)} fixed validation batches")

    micro_per_step = cfg.dpo.grad_accum_steps
    if cfg.dpo.max_steps > 0:
        max_steps = cfg.dpo.max_steps
    else:
        per_rank = len(train_ds.index) / max(1, env.world_size)
        max_steps = max(1, int(per_rank * cfg.dpo.epochs
                               / (cfg.dpo.micro_batch_size * micro_per_step)))
    log.info(f"    {max_steps:,} optimizer steps | beta {cfg.dpo.beta} "
             f"| sft weight {cfg.dpo.sft_loss_weight}")

    optimizer = build_optimizer(
        model, cfg.dpo.peak_lr, cfg.dpo.weight_decay,
        (cfg.optim.beta1, cfg.optim.beta2), cfg.optim.eps, device, log)
    scaler = torch.amp.GradScaler(enabled=use_scaler) if use_scaler else None
    lr_at = make_lr_fn(cfg.dpo.peak_lr, max_steps,
                       max(1, int(cfg.dpo.warmup_frac * max_steps)),
                       cfg.dpo.min_lr_ratio, "cosine")

    metrics = MetricLogger(cfg.runtime.log_dir, run, enabled=env.is_main,
                           tensorboard=cfg.runtime.tensorboard,
                           wandb=cfg.runtime.wandb,
                           wandb_project=cfg.runtime.wandb_project,
                           config=cfg.to_dict(), logger=log)
    metrics.path = os.path.join(cfg.runtime.log_dir, "dpo_metrics.json")

    train_model = model
    if env.enabled:
        train_model = DDP(model, device_ids=[env.local_rank]
                          if device.type == "cuda" else None,
                          gradient_as_bucket_view=True)

    autocast = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype,
                                      enabled=device.type == "cuda")

    def compute(xs, ys):
        with torch.no_grad(), autocast():
            ref_lp = sequence_logprobs(ref_model, xs, ys)
        with autocast():
            policy_lp = sequence_logprobs(model, xs, ys)
        loss, stats = dpo_loss(policy_lp, ref_lp, cfg.dpo.beta,
                               cfg.dpo.label_smoothing)
        if cfg.dpo.sft_loss_weight > 0:
            # Anchor the chosen branch. Pure DPO is satisfied by pushing both
            # log-probs down as long as the margin grows, which degrades the
            # policy while the loss looks healthy.
            n_tokens = (ys[0::2] != IGNORE_INDEX).sum().clamp(min=1)
            nll = -policy_lp[0::2].sum() / n_tokens
            loss = loss + cfg.dpo.sft_loss_weight * nll
            stats["nll_chosen"] = nll.detach()
        return loss, stats

    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        model.eval()
        total = torch.zeros((), device=device)
        acc = torch.zeros((), device=device)
        for xs, ys in val_batches:
            loss, stats = compute(xs, ys)
            total += loss.detach()
            acc += stats["accuracy"]
        model.train()
        n = max(1, len(val_batches))
        return (float(all_reduce_mean(total / n, env).item()),
                float(all_reduce_mean(acc / n, env).item()))

    log.info("[4] Training")
    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    all_params = [p for p in model.parameters() if p.requires_grad]

    out_dir = cfg.runtime.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "dpo_checkpoint_latest.pt")
    final_path = os.path.join(out_dir, "dpo_model.pt")

    global_step = 0
    accum_micro = 0
    accum = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0}
    started = time.time()
    last_val = float("nan")

    pbar = None
    if env.is_main:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_steps, desc="dpo", unit="step", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for xs, ys, _n_pairs in train_stream:
            if global_step >= max_steps:
                break
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            is_last_micro = (accum_micro + 1) == micro_per_step
            sync_ctx = (train_model.no_sync()
                        if env.enabled and not is_last_micro
                        else contextlib.nullcontext())
            with sync_ctx:
                loss, stats = compute(xs, ys)
                scaled = loss / micro_per_step
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            accum["loss"] += float(scaled.detach().item())
            accum["accuracy"] += float(stats["accuracy"].item()) / micro_per_step
            accum["margin"] += float(stats["margin"].item()) / micro_per_step
            accum_micro += 1
            if not is_last_micro:
                continue

            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm, _finite = clip_and_step(
                all_params, optimizer, cfg.dpo.grad_clip, scaler)

            global_step += 1
            metrics.log_train(global_step, accum["loss"], lr,
                              float(grad_norm.item()), 0.0)
            metrics.log(global_step, accuracy=accum["accuracy"],
                        margin=accum["margin"])
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{accum['loss']:.4f}",
                                  "acc": f"{accum['accuracy']:.2f}"})

            if global_step % cfg.runtime.log_every == 0:
                log.info(f"    step {global_step}/{max_steps} "
                         f"| loss {accum['loss']:.4f} "
                         f"| pref acc {accum['accuracy']:.3f} "
                         f"| margin {accum['margin']:.3f} | lr {lr:.2e}")

            accum = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0}
            accum_micro = 0

            if global_step % cfg.dpo.val_every == 0:
                last_val, val_acc = evaluate()
                metrics.log_val(global_step, last_val)
                log.info(f"    >> val loss {last_val:.4f} | pref acc {val_acc:.3f}")

            if global_step % cfg.runtime.ckpt_every == 0 and env.is_main:
                save_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                                step=global_step, config=cfg.to_dict(),
                                history=metrics.history, scaler=scaler,
                                extra={"val_loss": last_val, "stage": "dpo"})
                metrics.save()
    finally:
        if pbar is not None:
            pbar.close()

    if accum_micro:
        optimizer.zero_grad(set_to_none=True)

    last_val, val_acc = evaluate()
    metrics.log_val(global_step, last_val)
    log.info(f"Final val loss {last_val:.4f} | preference accuracy {val_acc:.3f}")
    log.info(f"    rendered {train_stream.rendered:,} pairs, "
             f"dropped {train_stream.dropped:,}")

    if env.is_main:
        save_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                        step=global_step, config=cfg.to_dict(),
                        history=metrics.history, scaler=scaler,
                        extra={"val_loss": last_val, "stage": "dpo"})
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
