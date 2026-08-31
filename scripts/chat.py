#!/usr/bin/env python
"""Chat with a fine-tuned checkpoint.

    python scripts/chat.py                          # newest of dpo/sft model
    python scripts/chat.py --checkpoint artifacts/sft_model.pt
    python scripts/chat.py --prompt "Explain gravity." --once

Slash commands inside the session:

    /reset          clear the conversation
    /retry          resample the last reply
    /undo           drop the last exchange
    /history        print the conversation as the model sees it
    /tokens         show prompt length vs context
    /set k=v        change a sampling knob (temperature, top_k, top_p,
                    min_p, repetition_penalty, max_new_tokens)
    /params         show current sampling settings
    /help  /exit
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from nanollm.chat import ChatSession, SamplingParams, stream_reply
from nanollm.config import TrainConfig, load_config
from nanollm.model import Transformer
from nanollm.tokenizer import Tokenizer

# Preference order: the most post-trained checkpoint that exists.
DEFAULT_CHECKPOINTS = [
    "artifacts/dpo_model.pt",
    "artifacts/dpo_checkpoint_latest.pt",
    "artifacts/sft_model.pt",
    "artifacts/sft_checkpoint_latest.pt",
]

TUNABLE = {"temperature": float, "top_k": int, "top_p": float, "min_p": float,
           "repetition_penalty": float, "max_new_tokens": int}


def pick_checkpoint(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for path in DEFAULT_CHECKPOINTS:
        if os.path.exists(path):
            return path
    return None


def load_model(path: str, fallback_config: str, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    saved = blob.get("config") if isinstance(blob, dict) else None
    if isinstance(saved, dict) and "model" in saved:
        cfg = TrainConfig.from_dict(saved)
    else:
        cfg = load_config(fallback_config if os.path.exists(fallback_config) else None, [])
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}

    model = Transformer.from_config(cfg.model).to(device)
    model.load_state_dict(state)
    model.eval()

    stage = blob.get("stage") if isinstance(blob, dict) else None
    step = blob.get("global_step") if isinstance(blob, dict) else None
    return model, cfg, stage, step


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help=f"default: first of {', '.join(DEFAULT_CHECKPOINTS)}")
    p.add_argument("--config", default="configs/base.yaml",
                   help="only used if the checkpoint carries no config")
    p.add_argument("--prompt", default=None, help="send one message")
    p.add_argument("--once", action="store_true",
                   help="with --prompt, answer and exit")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    path = pick_checkpoint(args.checkpoint)
    if path is None:
        target = args.checkpoint or " / ".join(DEFAULT_CHECKPOINTS)
        print(f"No chat checkpoint found ({target}).\n"
              f"Train one first:\n"
              f"  python -m nanollm.train.sft --config configs/base.yaml\n"
              f"  python -m nanollm.train.dpo --config configs/base.yaml\n\n"
              f"To sample from a base (pretrain-only) model instead, use:\n"
              f"  python scripts/generate.py --prompt '...'", file=sys.stderr)
        return 1

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, stage, step = load_model(path, args.config, device)
    tokenizer = Tokenizer(vocab_size=cfg.model.vocab_size)
    tokenizer.load(cfg.data.tokenizer_file)

    amp_dtype = torch.bfloat16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    params = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        min_p=args.min_p, repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens)
    session = ChatSession(tokenizer=tokenizer, n_seq=cfg.model.n_seq)

    def respond() -> str:
        prompt_ids, dropped = session.build_prompt(params.max_new_tokens)
        if dropped:
            print(f"[dropped {dropped} old turn(s) to fit the "
                  f"{cfg.model.n_seq}-token context]")
        print("assistant: ", end="", flush=True)
        pieces = []
        try:
            for delta in stream_reply(model, tokenizer, prompt_ids, params,
                                      device, amp_dtype):
                pieces.append(delta)
                print(delta, end="", flush=True)
        except KeyboardInterrupt:
            print("  [interrupted]", end="")
        print()
        return "".join(pieces)

    # -- one-shot ----------------------------------------------------------
    if args.prompt and args.once:
        session.add_user(args.prompt)
        respond()
        return 0

    # -- interactive -------------------------------------------------------
    label = f"{stage or 'model'}" + (f" @ step {step}" if step else "")
    print(f"nanollm chat -- {os.path.basename(path)} ({label})")
    print(f"{model.get_param_count()/1e6:.0f}M params | ctx {cfg.model.n_seq} "
          f"| {device} | {params.describe()}")
    print("/help for commands, /exit to quit.\n")

    if args.prompt:
        session.add_user(args.prompt)
        print(f"you: {args.prompt}")
        session.add_assistant(respond())

    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            command, rest = command.lower(), rest.strip()

            if command in ("exit", "quit", "q"):
                break
            if command == "help":
                print(__doc__.split("Slash commands", 1)[1])
                continue
            if command == "reset":
                session.reset()
                print("[conversation cleared]")
                continue
            if command == "undo":
                session.undo()
                print(f"[{len(session.messages)} message(s) left]")
                continue
            if command == "retry":
                if not any(m["role"] == "user" for m in session.messages):
                    print("[nothing to retry]")
                    continue
                last_user = [m for m in session.messages if m["role"] == "user"][-1]
                session.undo()
                session.add_user(last_user["content"])
                session.add_assistant(respond())
                continue
            if command == "history":
                if not session.messages:
                    print("[empty]")
                for m in session.messages:
                    print(f"  <|{m['role'].upper()}|> {m['content']}")
                continue
            if command == "tokens":
                ids, dropped = session.build_prompt(params.max_new_tokens)
                print(f"[prompt {len(ids)} tokens of {cfg.model.n_seq} "
                      f"| reply budget {params.max_new_tokens} "
                      f"| {dropped} turn(s) dropped]")
                continue
            if command == "params":
                print(f"[{params.describe()}]")
                continue
            if command == "set":
                key, _, value = rest.partition("=")
                key, value = key.strip(), value.strip()
                if key not in TUNABLE:
                    print(f"[unknown setting {key!r}; "
                          f"one of {', '.join(sorted(TUNABLE))}]")
                    continue
                try:
                    setattr(params, key, TUNABLE[key](value))
                except ValueError:
                    print(f"[{value!r} is not a valid {TUNABLE[key].__name__}]")
                    continue
                print(f"[{key} = {getattr(params, key)}]")
                continue
            print(f"[unknown command /{command}; try /help]")
            continue

        session.add_user(line)
        session.add_assistant(respond())

    print("bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
