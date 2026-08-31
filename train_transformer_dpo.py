#!/usr/bin/env python
"""Legacy DPO entry point.

The loop moved to ``nanollm.train.dpo`` -- config-driven, resumable and
multi-GPU. This wrapper keeps the old command working and forwards arguments.
The previous script is kept at ``rough/train_transformer_dpo_legacy.py``.

    python train_transformer_dpo.py
    torchrun --nproc_per_node=8 -m nanollm.train.dpo --config configs/base.yaml
"""

import os
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.train.dpo import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--config" not in argv and os.path.exists("configs/base.yaml"):
        argv = ["--config", "configs/base.yaml"] + argv
    sys.exit(main(argv))
