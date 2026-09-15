#!/usr/bin/env python
"""Mix shards from one token set into another, updating the manifest.

Mid-training on a small curated corpus drifts the model off general text the
same way pure SFT does, and the pretrain loop has no replay knob -- so the
rehearsal has to happen in the data. This links a few of the original
pretraining shards into the mid-training set and rewrites its ``index.json``
so the loader actually sees them.

    python scripts/mix_shards.py --into dataset/tokens_midtrain \\
        --from dataset/tokens --shards 2

``--shards 2`` at 100M tokens each is ~200M tokens of original corpus against a
~46M-token textbook set, i.e. the textbooks are ~19% of what the stage sees.
Raise it to anneal more gently, lower it to push harder toward the textbooks.
Shards are linked, not copied, so this costs no disk.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanollm.data.shards import ShardIndex


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--into", required=True, help="shard set to add to")
    p.add_argument("--from", dest="source", required=True, help="shard set to take from")
    p.add_argument("--shards", type=int, default=2, help="how many to link in")
    p.add_argument("--prefix", default="mixin", help="name prefix for the links")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = p.parse_args()

    target = ShardIndex.load(args.into)
    source = ShardIndex.load(args.source)
    if source.dtype != target.dtype or source.vocab_size != target.vocab_size:
        print(f"refusing to mix: {args.source} is {source.dtype}/"
              f"{source.vocab_size}, {args.into} is {target.dtype}/"
              f"{target.vocab_size}", file=sys.stderr)
        return 1

    take = source.shards[:max(0, args.shards)]
    if not take:
        print("nothing to mix", file=sys.stderr)
        return 1

    added, added_tokens = [], 0
    per_shard = source.total_tokens // max(1, len(source.shards))
    for i, name in enumerate(take):
        link = f"{args.prefix}_{i:05d}.bin"
        dst = os.path.join(args.into, link)
        if link in target.shards:
            print(f"  {link} already mixed in, skipping")
            continue
        src = os.path.abspath(os.path.join(source.dir, name))
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy:
            import shutil
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)
        added.append(link)
        added_tokens += per_shard
        print(f"  {link} -> {src}")

    if not added:
        print("nothing added")
        return 0

    target.shards = target.shards + added
    target.total_tokens += added_tokens
    target.meta = dict(target.meta or {})
    target.meta["mixed_in"] = target.meta.get("mixed_in", []) + [
        {"source": args.source, "shards": added}]
    target.save()
    print(f"\n{args.into}: {len(target.shards)} shards, "
          f"~{target.total_tokens/1e6:.0f}M tokens "
          f"({added_tokens/max(1,target.total_tokens):.0%} mixed in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
