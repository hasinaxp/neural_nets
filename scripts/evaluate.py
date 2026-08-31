#!/usr/bin/env python
"""Evaluate a checkpoint: held-out perplexity, plus multiple choice if given.

    python scripts/evaluate.py --checkpoint artifacts/pretrain_model.pt
    python scripts/evaluate.py --checkpoint artifacts/sft_model.pt \
        --mc-file dataset/eval/hellaswag.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from nanollm.config import TrainConfig, load_config
from nanollm.data.loader import BatchStream, TokenCorpus
from nanollm.data.shards import ShardIndex
from nanollm.eval import (evaluate_multiple_choice, evaluate_perplexity,
                          load_jsonl_mc)
from nanollm.model import Transformer
from nanollm.tokenizer import Tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="artifacts/pretrain_model.pt")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--mc-file", default=None, help="multiple-choice JSONL")
    p.add_argument("--mc-limit", type=int, default=None)
    p.add_argument("--out", default=None, help="write results as JSON here")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = blob.get("config") if isinstance(blob, dict) else None
    cfg = (TrainConfig.from_dict(saved) if isinstance(saved, dict) and "model" in saved
           else load_config(args.config, []))
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

    model = Transformer.from_config(cfg.model).to(device)
    model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v
                           for k, v in state.items()})
    model.eval()

    tokenizer = Tokenizer(vocab_size=cfg.model.vocab_size)
    tokenizer.load(cfg.data.tokenizer_file)

    results = {"checkpoint": args.checkpoint,
               "params": model.get_param_count()}

    # Held-out perplexity on the validation shards.
    index = ShardIndex.load(cfg.data.data_dir)
    val_paths = index.paths[-max(1, cfg.data.val_shards):]
    corpus = TokenCorpus(val_paths, cfg.model.n_seq)
    stream = BatchStream(corpus, args.micro_batch, seed=cfg.data.seed + 99991,
                         pin_memory=False)
    ppl = evaluate_perplexity(model, stream.take(args.batches), device)
    results["perplexity"] = ppl
    print(f"perplexity  : {ppl['perplexity']:.3f}  "
          f"(loss {ppl['loss']:.4f}, {ppl['bits_per_token']:.3f} bits/token, "
          f"{ppl['tokens']:,} tokens)")

    if args.mc_file:
        examples = load_jsonl_mc(args.mc_file, limit=args.mc_limit)
        mc = evaluate_multiple_choice(model, tokenizer, examples, device,
                                      log_every=200)
        results["multiple_choice"] = mc
        print(f"mc accuracy : {mc['accuracy']:.4f} length-normalised "
              f"| {mc['accuracy_raw']:.4f} raw | n={mc['n']}")
        for task, acc in mc.get("by_task", {}).items():
            print(f"  {task:20s} {acc:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
