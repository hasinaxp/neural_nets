#!/usr/bin/env python
"""Sample from a trained checkpoint.

    python scripts/generate.py --prompt "The history of" --tokens 200
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from nanollm.config import ModelConfig, TrainConfig, load_config
from nanollm.model import Transformer
from nanollm.tokenizer import Tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="artifacts/pretrain_checkpoint_latest.pt")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # A checkpoint carries its own config, so the CLI --config is only a
    # fallback for bare state_dict files.
    saved = ck.get("config")
    if isinstance(saved, dict) and "model" in saved:
        cfg = TrainConfig.from_dict(saved)
    else:
        cfg = load_config(args.config, [])
    state = ck.get("model_state_dict", ck)

    tok = Tokenizer(vocab_size=cfg.model.vocab_size)
    tok.load(cfg.data.tokenizer_file)

    model = Transformer.from_config(cfg.model).to(device)
    model.load_state_dict(state)
    model.eval()

    bos = tok.special_tokens["<|BOS|>"]
    ids = [bos] + (tok.encode(args.prompt) if args.prompt else [])
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.samples):
        out = model.generate(
            idx, max_count=args.tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            eos_token_id=tok.special_tokens.get("<|EOS|>"),
            valid_vocab_size=tok.vocab_size)
        print(f"--- sample {i+1} ---")
        print(tok.decode(out[0].cpu().tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
