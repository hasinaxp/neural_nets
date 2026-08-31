#!/usr/bin/env python
"""Train the BPE tokenizer on a sample of the raw corpus.

    python scripts/train_tokenizer.py --vocab-size 32768 --sample-mb 200
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanollm.data.sources import iter_all_documents
from nanollm.tokenizer import Tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--raw-dir", default="dataset/raw")
    p.add_argument("--out", default=None)
    p.add_argument("--sample-mb", type=int, default=200,
                   help="how much text to fit on; BPE gains flatten out well "
                        "before the full corpus")
    p.add_argument("--text-file", default=None,
                   help="train on a plain text file instead of the corpus")
    args = p.parse_args()

    out = args.out or f"artifacts/tokenizer-{args.vocab_size}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tok = Tokenizer(vocab_size=args.vocab_size)

    if args.text_file:
        print(f"training on {args.text_file}")
        tok.train_from_file(args.text_file)
    else:
        budget = args.sample_mb * 1024 * 1024
        parts, total = [], 0
        for _source, doc in iter_all_documents(raw_dir=args.raw_dir):
            parts.append(doc)
            total += len(doc)
            if total >= budget:
                break
        if not parts:
            print(f"no documents under {args.raw_dir}", file=sys.stderr)
            return 1
        print(f"training on {total/1e6:.1f}MB from {len(parts):,} documents")
        tok.train_from_text("\n\n".join(parts))

    tok.save(out)
    print(f"saved {out} ({len(tok.merges)} merges, vocab {max(tok.vocab)+1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
