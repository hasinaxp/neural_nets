#!/usr/bin/env python
"""Download and clean the mid-training textbook corpus.

    python scripts/download_midtrain.py
    python scripts/download_midtrain.py --sources openstax ncert
    python scripts/download_midtrain.py --skip-download      # re-clean only

Writes parquet with a ``text`` column into --out-dir, which is the format
``scripts/prepare_data.py`` already reads. See nanollm/data/midtrain.py for
what this stage does and does not fix.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanollm.data.midtrain import (MIDTRAIN_SOURCES, build_midtrain_corpus,
                                   download_midtrain)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="dataset/midtrain_raw",
                   help="where the source repos are downloaded")
    p.add_argument("--out-dir", default="dataset/raw_midtrain",
                   help="where the cleaned parquet is written")
    p.add_argument("--sources", nargs="*", default=None,
                   choices=list(MIDTRAIN_SOURCES),
                   help=f"default: all of {', '.join(MIDTRAIN_SOURCES)}")
    p.add_argument("--skip-download", action="store_true",
                   help="clean what is already in --cache-dir")
    args = p.parse_args()

    names = args.sources or list(MIDTRAIN_SOURCES)
    print("Sources:")
    for name in names:
        source = MIDTRAIN_SOURCES[name]
        print(f"  {name:16s} {source.repo_id:<42} {source.note}")
    print()

    if not args.skip_download:
        download_midtrain(args.cache_dir, names)
        print()

    build_midtrain_corpus(args.cache_dir, args.out_dir, names)
    print(f"\nNext:\n"
          f"  python scripts/prepare_data.py --raw-dir {args.out_dir} \\\n"
          f"      --out-dir dataset/tokens_midtrain --no-wikipedia\n"
          f"  python -m nanollm.train.pretrain --config configs/midtrain.yaml \\\n"
          f"      --init-from artifacts/pretrain_checkpoint_latest.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
