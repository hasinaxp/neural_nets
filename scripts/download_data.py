#!/usr/bin/env python
"""Download corpora for every training stage.

    python scripts/download_data.py pretrain --budget-gb 40
    python scripts/download_data.py sft
    python scripts/download_data.py dpo
    python scripts/download_data.py all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    pre = sub.add_parser("pretrain", help="web-scale text for pretraining")
    pre.add_argument("--out-dir", default="dataset/raw")
    pre.add_argument("--budget-gb", type=float, default=40.0,
                     help="total download budget, split across sources by weight")
    pre.add_argument("--sources", nargs="*", default=None)
    pre.add_argument("--copy", action="store_true",
                     help="copy out of the hub cache instead of symlinking")

    sft = sub.add_parser("sft", help="instruction / chat data")
    sft.add_argument("--force", action="store_true")

    dpo = sub.add_parser("dpo", help="preference pairs")
    dpo.add_argument("--force", action="store_true")

    allp = sub.add_parser("all", help="every stage")
    allp.add_argument("--budget-gb", type=float, default=40.0)

    args = p.parse_args()

    if args.stage in ("pretrain", "all"):
        from nanollm.data.download_pretrain import download_corpus
        download_corpus(
            out_dir=getattr(args, "out_dir", "dataset/raw"),
            budget_gb=args.budget_gb,
            sources=getattr(args, "sources", None),
            symlink=not getattr(args, "copy", False),
        )

    if args.stage in ("sft", "all"):
        from nanollm.data.download_sft import main as sft_main
        print("\n=== SFT ===")
        sft_main()

    if args.stage in ("dpo", "all"):
        from nanollm.data.download_dpo import main as dpo_main
        print("\n=== DPO ===")
        dpo_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
