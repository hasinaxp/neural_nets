"""Pretraining entry point.

Single GPU::

    python -m nanollm.train.pretrain --config configs/base.yaml

Multi-GPU (one process per GPU, torchrun handles the env)::

    torchrun --standalone --nproc_per_node=8 -m nanollm.train.pretrain \
        --config configs/base.yaml

Every knob is in the YAML; ``--set section.key=value`` overrides individual
fields without editing it.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import TrainConfig, load_config
from ..data.loader import BatchStream, CudaPrefetcher, TokenCorpus
from ..data.shards import ShardIndex
from ..model import Transformer
from ..tokenizer import Tokenizer
from ..utils.checkpoint import (ArchitectureMismatch, atomic_save,
                                load_checkpoint, save_checkpoint, set_aside,
                                unwrap_model)
from ..utils.distributed import (all_reduce_mean, cleanup_distributed,
                                 setup_distributed)
from ..utils.logging import MetricLogger, run_id, setup_logging
from ..utils.schedules import make_lr_fn
from .common import (build_optimizer, clip_and_step, configure_backends,
                     load_tokenizer, peak_flops)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanollm.train.pretrain",
        description="Pretrain a small decoder-only transformer.")
    p.add_argument("--config", default=None, help="path to a YAML config")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override a config field, e.g. --set optim.peak_lr=3e-4")
    p.add_argument("--resume", default=None,
                   help="'auto' (default), 'never', or a checkpoint path")
    p.add_argument("--max-steps", type=int, default=None,
                   help="shorthand for --set optim.max_steps=N")
    p.add_argument("--dry-run", action="store_true",
                   help="build everything, run 3 steps, exit")
    return p


def build_corpora(cfg: TrainConfig, log) -> tuple[TokenCorpus, TokenCorpus]:
    """Split shards into train and validation sets.

    Validation is whole held-out *shards*, not a random slice of the training
    shards -- otherwise every validation window overlaps training windows and
    the number stops meaning anything.
    """
    index = ShardIndex.load(cfg.data.data_dir)
    paths = index.paths
    n_val = min(cfg.data.val_shards, max(0, len(paths) - 1))
    if n_val == 0 and len(paths) == 1:
        log.warning(
            "    only one shard: validation reuses it, so val loss will track "
            "train loss. Prepare more shards for a clean split.")
        train_paths, val_paths = paths, paths
    else:
        train_paths, val_paths = paths[:-n_val], paths[-n_val:]

    train = TokenCorpus(train_paths, cfg.model.n_seq)
    val = TokenCorpus(val_paths, cfg.model.n_seq)
    log.info(f"    train: {len(train_paths)} shards, {train.total_tokens:,} tokens")
    log.info(f"    val:   {len(val_paths)} shards, {val.total_tokens:,} tokens")
    return train, val


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.max_steps is not None:
        cfg.optim.max_steps = args.max_steps
    if args.resume is not None:
        cfg.runtime.resume = args.resume
    if args.dry_run:
        cfg.optim.max_steps = 3
        cfg.runtime.compile = False

    env = setup_distributed()
    run = run_id(cfg.runtime.run_name)
    if env.enabled:
        # Every rank must agree on the run id or they write to different dirs.
        obj = [run]
        dist.broadcast_object_list(obj, src=0)
        run = obj[0]

    log = setup_logging(cfg.runtime.log_dir, run, "train", env.is_main, env.rank)
    device = env.device

    torch.manual_seed(cfg.runtime.seed + env.rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.runtime.seed + env.rank)

    amp_dtype = configure_backends(cfg, device)
    use_scaler = device.type == "cuda" and amp_dtype is torch.float16

    log.info(f"[0] run {run} | device {device} | world size {env.world_size}")
    if device.type == "cuda":
        log.info(f"    GPU: {torch.cuda.get_device_name(device)}")
    log.info(f"    AMP dtype {amp_dtype} (grad scaler: {use_scaler})")

    # -- tokenizer + data ---------------------------------------------------
    log.info("[1] Tokenizer")
    tokenizer, model_vocab, eos_id = load_tokenizer(cfg, log)
    cfg.model.vocab_size = model_vocab

    log.info("[2] Data")
    train_corpus, val_corpus = build_corpora(cfg, log)

    # -- model --------------------------------------------------------------
    log.info("[3] Model")
    model = Transformer.from_config(cfg.model).to(device)
    for p in model.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()

    param_count = model.get_param_count()
    tokens_per_step = cfg.optim.tokens_per_step(cfg.model.n_seq, env.world_size)
    log.info(f"    {cfg.summary(env.world_size)}")
    log.info(f"    actual params: {param_count:,} | n_kv_head {model.n_kv_head} "
             f"| head_dim {cfg.model.head_dim}")

    optimizer = build_optimizer(
        model, cfg.optim.peak_lr, cfg.optim.weight_decay,
        (cfg.optim.beta1, cfg.optim.beta2), cfg.optim.eps, device, log)
    scaler = torch.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    metrics = MetricLogger(
        cfg.runtime.log_dir, run, enabled=env.is_main,
        tensorboard=cfg.runtime.tensorboard, wandb=cfg.runtime.wandb,
        wandb_project=cfg.runtime.wandb_project, config=cfg.to_dict(), logger=log)

    # -- resume -------------------------------------------------------------
    os.makedirs(cfg.runtime.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.runtime.out_dir, "pretrain_checkpoint_latest.pt")
    final_path = os.path.join(cfg.runtime.out_dir, "pretrain_model.pt")

    global_step = 0
    resume_from = None
    if cfg.runtime.resume == "auto":
        resume_from = ckpt_path if os.path.exists(ckpt_path) else None
    elif cfg.runtime.resume not in ("never", ""):
        resume_from = cfg.runtime.resume

    if resume_from and os.path.exists(resume_from):
        try:
            ck = load_checkpoint(
                resume_from, model=model, optimizer=optimizer, scaler=scaler,
                map_location=device, log=log.warning,
                expect_arch={
                    "vocab_size": cfg.model.vocab_size, "n_dim": cfg.model.n_dim,
                    "n_layer": cfg.model.n_layer, "n_head": cfg.model.n_head,
                    "n_kv_head": cfg.model.n_kv_head, "n_seq": cfg.model.n_seq,
                })
            global_step = int(ck.get("global_step", 0))
            metrics.load(ck.get("history") or {})
            log.info(f"    resumed from {resume_from} at step {global_step}")
        except ArchitectureMismatch as e:
            if env.is_main:
                set_aside(resume_from, f"arch-{run}", log.warning)
            env.barrier()
            log.warning(f"    {e}; starting fresh at step 0")
        except Exception as e:
            log.warning(f"    could not load {resume_from}: {e}; starting fresh")
    else:
        log.info("    no checkpoint to resume; starting at step 0")

    # -- compile / DDP ------------------------------------------------------
    # Order matters: compile the inner module, then wrap in DDP. Compiling the
    # DDP wrapper instead makes dynamo trace the gradient hooks.
    if cfg.runtime.compile:
        try:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 32
            model.calculate_loss = torch.compile(
                model.calculate_loss, mode=cfg.runtime.compile_mode, dynamic=False)
            log.info(f"    torch.compile on (mode={cfg.runtime.compile_mode}); "
                     f"first step pays a one-off warmup")
        except Exception as e:
            log.warning(f"    torch.compile failed, running eager: {e}")

    train_model = model
    if env.enabled:
        train_model = DDP(model, device_ids=[env.local_rank]
                          if device.type == "cuda" else None,
                          gradient_as_bucket_view=True)

    # -- streams ------------------------------------------------------------
    train_stream = BatchStream(
        train_corpus, cfg.optim.micro_batch_size, seed=cfg.data.seed,
        rank=env.rank, world_size=env.world_size,
        start_step=global_step * cfg.optim.grad_accum_steps,
        pin_memory=device.type == "cuda")
    batches = CudaPrefetcher(train_stream, device,
                             enabled=device.type == "cuda")

    # Validation batches are fixed for the life of the run, so val loss is
    # comparable across steps and across runs with the same seed.
    val_stream = BatchStream(val_corpus, cfg.optim.micro_batch_size,
                             seed=cfg.data.seed + 99991, rank=0, world_size=1,
                             pin_memory=False)
    val_batches = [(x.to(device), y.to(device))
                   for x, y in val_stream.take(cfg.data.val_batches)]
    log.info(f"    {len(val_batches)} fixed validation batches")

    device_peak = peak_flops(device, env.world_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # -- loop ---------------------------------------------------------------
    log.info("[4] Training")
    train_model.train()
    optimizer.zero_grad(set_to_none=True)

    lr_at = make_lr_fn(
        cfg.optim.peak_lr, cfg.optim.max_steps,
        max(1, int(cfg.optim.warmup_frac * cfg.optim.max_steps)),
        cfg.optim.min_lr_ratio, cfg.optim.schedule, cfg.optim.decay_frac)

    all_params = [p for p in model.parameters() if p.requires_grad]
    accum_loss = torch.zeros((), device=device)
    accum_micro = 0
    skipped = 0
    started = time.time()
    step_timer = time.time()

    # Per-step scalars stay on the GPU and are read back once every log_every
    # steps; a .item() per step would drain the pipeline every step.
    pending: list[tuple] = []

    def flush(force: bool = False) -> Optional[tuple]:
        nonlocal skipped
        if not pending:
            return None
        packed = torch.stack([row for *_, row in pending]).cpu()   # one sync
        last = None
        for (step, lr_v, tps, mfu, _), row in zip(pending, packed):
            loss_v, gn, finite = float(row[0]), float(row[1]), float(row[2])
            if finite < 0.5:
                skipped += 1
                log.warning(f"    non-finite grad norm at step {step}; "
                            f"update skipped ({skipped} total)")
            metrics.log_train(step, loss_v, lr_v, gn, tps, mfu)
            last = (step, loss_v, lr_v, gn, tps, mfu)
        pending.clear()
        return last

    @torch.no_grad()
    def evaluate() -> float:
        train_model.eval()
        total = torch.zeros((), device=device)
        for xs, ys in val_batches:
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                total += model.calculate_loss(xs, ys).detach()
        train_model.train()
        mean = total / max(1, len(val_batches))
        return float(all_reduce_mean(mean, env).item())

    @torch.inference_mode()
    def sample(step: int, max_count: int = 80) -> None:
        """Log a sample. Never fatal: this is a progress readout, and a run
        that has trained and saved must not fail on the way out."""
        if not env.is_main:
            return
        model.eval()
        try:
            seed = torch.tensor([[tokenizer.special_tokens["<|BOS|>"]]],
                                dtype=torch.long, device=device)
            budget = max(1, min(max_count, cfg.model.n_seq - seed.size(1)))
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                out = model.generate(seed, max_count=budget, eos_token_id=eos_id,
                                     valid_vocab_size=tokenizer.vocab_size)
            log.info(f"    sample @ {step}: "
                     f"{tokenizer.decode(out[0].cpu().tolist())}")
        except Exception as e:
            log.warning(f"    sampling at step {step} failed: {e}")
        finally:
            model.train()

    def checkpoint(path: str) -> None:
        if not env.is_main:
            return
        save_checkpoint(
            path, model=model, optimizer=optimizer, step=global_step,
            config=cfg.to_dict(), history=metrics.history, scaler=scaler,
            extra={"val_loss": last_val, "tokenizer_vocab": tokenizer.vocab_size})

    last_val = float("nan")
    pbar = None
    if env.is_main:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=cfg.optim.max_steps, initial=global_step,
                        desc="pretrain", unit="step", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for xs, ys in batches:
            if global_step >= cfg.optim.max_steps:
                break

            # DDP: skip the all-reduce on every micro-step but the last, or
            # gradients are synchronised grad_accum_steps times per step.
            is_last_micro = (accum_micro + 1) == cfg.optim.grad_accum_steps
            sync_ctx = (train_model.no_sync()
                        if env.enabled and not is_last_micro
                        else contextlib.nullcontext())

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    loss = model.calculate_loss(xs, ys) / cfg.optim.grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            accum_loss += loss.detach()      # on-device; .item() would sync
            accum_micro += 1
            if not is_last_micro:
                continue

            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            grad_norm, finite = clip_and_step(
                all_params, optimizer, cfg.optim.grad_clip, scaler)

            row = torch.stack((accum_loss.detach().float(), grad_norm, finite))
            accum_loss.zero_()
            accum_micro = 0
            global_step += 1

            step_time = time.time() - step_timer
            step_timer = time.time()
            tps = tokens_per_step / max(1e-6, step_time)
            mfu = (model.estimate_flops_per_token() * tps / device_peak
                   if device_peak else 0.0)
            pending.append((global_step, lr, tps, mfu, row))
            if pbar is not None:
                pbar.update(1)

            if global_step % cfg.runtime.log_every == 0:
                last = flush()
                if last and env.is_main:
                    step, loss_v, lr_v, gn, tps_v, mfu_v = last
                    recent = metrics.history["train_loss"][-cfg.runtime.log_every:]
                    avg = sum(recent) / max(1, len(recent))
                    msg = (f"    step {step}/{cfg.optim.max_steps} | loss {avg:.4f} "
                           f"| ppl {math.exp(min(20, avg)):.1f} | lr {lr_v:.2e} "
                           f"| gn {gn:.2f} | {tps_v/1e3:.1f}k tok/s")
                    if mfu_v:
                        msg += f" | mfu {mfu_v*100:.1f}%"
                    if device.type == "cuda":
                        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                        total_gb = torch.cuda.get_device_properties(
                            device).total_memory / 1024**3
                        msg += f" | mem {peak_gb:.1f}/{total_gb:.0f}GB"
                    log.info(msg)
                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{avg:.4f}", "lr": f"{lr_v:.2e}",
                            "tok/s": f"{tps_v/1e3:.1f}k",
                            "mfu": f"{mfu_v*100:.0f}%" if mfu_v else "n/a"})

            if global_step % cfg.runtime.eval_every == 0:
                last_val = evaluate()
                metrics.log_val(global_step, last_val)
                log.info(f"    >> val loss {last_val:.4f} | "
                         f"ppl {math.exp(min(20, last_val)):.2f}")

            if global_step % cfg.runtime.sample_every == 0:
                sample(global_step)

            # Plot/checkpoint read history, so drain staged metrics first.
            if global_step % cfg.runtime.plot_every == 0:
                flush()
                metrics.plot()
                metrics.save()

            if global_step % cfg.runtime.ckpt_every == 0:
                flush()
                checkpoint(ckpt_path)
                log.info(f"    checkpoint saved at step {global_step}")

            if global_step % cfg.runtime.snapshot_every == 0:
                flush()
                checkpoint(os.path.join(cfg.runtime.out_dir,
                                        f"pretrain_step{global_step}.pt"))
    finally:
        if pbar is not None:
            pbar.close()
        batches.close()
        flush()

    # A partial accumulation is a fraction of a step; drop it rather than
    # applying an update at the wrong effective batch size.
    if accum_micro:
        optimizer.zero_grad(set_to_none=True)

    last_val = evaluate()
    metrics.log_val(global_step, last_val)
    log.info(f"Final val loss {last_val:.4f} | ppl {math.exp(min(20, last_val)):.2f}")

    checkpoint(ckpt_path)
    if env.is_main:
        atomic_save(unwrap_model(model).state_dict(), final_path)
        metrics.plot()
        metrics.save()
        with open(os.path.join(cfg.runtime.log_dir, f"config_{run}.yaml"), "w") as f:
            f.write(cfg.to_yaml())

    elapsed = time.time() - started
    log.info(f"Done in {timedelta(seconds=int(elapsed))} | steps {global_step} "
             f"| skipped {skipped} | tokens {global_step * tokens_per_step:,}")
    if device.type == "cuda":
        log.info(f"Peak GPU memory: "
                 f"{torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
    if env.is_main:
        log.info(f"Model saved to {final_path}")
        sample(global_step)

    metrics.close()
    cleanup_distributed(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
