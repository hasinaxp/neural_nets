#!/usr/bin/env python
"""Tokenise the raw corpus into binary shards.

    python scripts/prepare_data.py --workers 16

Run this once. Training reads the shards it produces; nothing in the training
loop touches the raw parquet files or the BPE tokenizer again.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanollm.config import load_config
from nanollm.data.prepare import prepare_shards
from nanollm.tokenizer import Tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--shard-tokens", type=int, default=100_000_000,
                   help="tokens per shard file (default 100M ~ 200MB at uint16)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="stop after this many tokens; handy for a quick corpus")
    p.add_argument("--no-wikipedia", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   help="stop cleanly when the disk has less than this free")
    args = p.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None, [])
    raw_dir = args.raw_dir or cfg.data.raw_dir
    out_dir = args.out_dir or cfg.data.data_dir
    tokenizer_file = args.tokenizer or cfg.data.tokenizer_file

    if not os.path.exists(tokenizer_file):
        print(f"tokenizer not found: {tokenizer_file}\n"
              f"train one first:  python scripts/train_tokenizer.py", file=sys.stderr)
        return 1
    if not os.path.isdir(raw_dir):
        print(f"raw corpus not found: {raw_dir}\n"
              f"download it first:  python dataset_pretrain_download.py", file=sys.stderr)
        return 1

    tok = Tokenizer(vocab_size=cfg.model.vocab_size)
    tok.load(tokenizer_file)
    eos_id = tok.special_tokens.get("<|EOS|>") or tok.special_tokens["<|BOS|>"]
    vocab_size = max(tok.vocab) + 1

    print(f"raw       : {raw_dir}")
    print(f"out       : {out_dir}")
    print(f"tokenizer : {tokenizer_file} (vocab {vocab_size}, eos {eos_id})")
    print(f"workers   : {args.workers}")

    from nanollm.data.shards import free_bytes
    free_gb = free_bytes(out_dir) / 1024 ** 3
    if args.max_tokens:
        need_gb = args.max_tokens * 2 / 1024 ** 3
        print(f"disk      : {free_gb:.1f}GB free | "
              f"~{need_gb:.1f}GB needed for {args.max_tokens/1e9:.1f}B tokens")
        if need_gb > free_gb - args.min_free_gb:
            print(f"  WARNING: that will not fit. Lower --max-tokens to about "
                  f"{max(0, (free_gb - args.min_free_gb)) * 1024**3 / 2 / 1e9:.1f}B.")
    else:
        cap = max(0.0, (free_gb - args.min_free_gb)) * 1024 ** 3 / 2 / 1e9
        print(f"disk      : {free_gb:.1f}GB free | no --max-tokens set, so this "
              f"runs until the corpus ends or ~{cap:.1f}B tokens fill the disk")

    index = prepare_shards(
        raw_dir=raw_dir,
        out_dir=out_dir,
        tokenizer_file=tokenizer_file,
        vocab_size=vocab_size,
        eos_id=eos_id,
        shard_tokens=args.shard_tokens,
        num_workers=args.workers,
        max_tokens=args.max_tokens,
        include_wikipedia=not args.no_wikipedia,
        min_free_gb=args.min_free_gb,
    )
    print(f"\nindex: {os.path.join(index.dir, 'index.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
