#!/usr/bin/env python
"""Legacy entry point -- see ``scripts/download_data.py pretrain``.

The old fixed file-index lists are gone; downloads are now driven by a size
budget split across sources, which survives the upstream repos being re-sharded.
"""

import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.data.download_pretrain import download_corpus

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    download_corpus(budget_gb=budget)
