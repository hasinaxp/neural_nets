"""torchrun-compatible distributed setup.

Single-GPU is the same code path with world_size 1, so nothing branches on
"are we distributed" beyond this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

__all__ = ["DistEnv", "setup_distributed", "cleanup_distributed", "all_reduce_mean"]


@dataclass
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier(device_ids=[self.local_rank]
                         if self.device.type == "cuda" else None)


def setup_distributed() -> DistEnv:
    """Initialise from torchrun's env vars, or fall back to single process."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return DistEnv(device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistEnv(rank=rank, local_rank=local_rank, world_size=world_size,
                   device=device, enabled=True)


def cleanup_distributed(env: DistEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """Average a scalar across ranks. No-op when not distributed."""
    if not env.enabled:
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / env.world_size
