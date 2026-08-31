#!/usr/bin/env python
"""Legacy SFT entry point.

The loop moved to ``nanollm.train.sft`` -- config-driven, resumable and
multi-GPU. This wrapper keeps the old command working and forwards arguments.
The previous script is kept at ``rough/train_transformer_sft_legacy.py``.

    python train_transformer_sft.py
    torchrun --nproc_per_node=8 -m nanollm.train.sft --config configs/base.yaml
"""

import os
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.train.sft import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--config" not in argv and os.path.exists("configs/base.yaml"):
        argv = ["--config", "configs/base.yaml"] + argv
    sys.exit(main(argv))
