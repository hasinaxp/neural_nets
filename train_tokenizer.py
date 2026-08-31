#!/usr/bin/env python
"""Legacy entry point -- see ``scripts/train_tokenizer.py`` for the full CLI."""

import os
import runpy
import sys

if __name__ == "__main__":
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "scripts", "train_tokenizer.py"),
        run_name="__main__")
