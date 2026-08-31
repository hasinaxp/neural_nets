#!/usr/bin/env python
"""Legacy pretraining entry point.

The training loop moved to ``nanollm.train.pretrain``, which takes a config
file and supports multi-GPU. This wrapper keeps ``python
train_transformer_pretraining.py`` working and forwards any arguments.

    python train_transformer_pretraining.py                    # configs/base.yaml
    python train_transformer_pretraining.py --set optim.peak_lr=3e-4
    torchrun --nproc_per_node=8 -m nanollm.train.pretrain --config configs/base.yaml
"""

import os
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.train.pretrain import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    default_config = "configs/base.yaml"
    if "--config" not in argv and os.path.exists(default_config):
        argv = ["--config", default_config] + argv
    sys.exit(main(argv))
