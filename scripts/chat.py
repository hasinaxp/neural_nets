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
    /think [on|off] ask for a "Thinking: ... / Answer: ..." scratchpad before
                    the answer (toggles when given no argument). Trained on
                    maths and multiple-choice reasoning, where the source data
                    carries a real rationale -- elsewhere the model will
                    usually just answer.
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
from nanollm.data.sft import ANSWER_PREFIX, THINK_PREFIX, THINK_PROMPTS
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
    # Defaults tuned for a 169M model, which is far more sensitive to sampling
    # temperature than a frontier one: at 0.8 this checkpoint answered "who
    # wrote Pride and Prejudice" with "Elizabeth Taylor", and at greedy with
    # "Jane Austen". The knowledge is there; a hot tail is what loses it.
    #
    # min_p rather than a lower temperature alone: cutting temperature on its
    # own drives the model into repetition loops (at 0.3 it produced
    # "Dear Mr./Ms./Mrs./Mr./..." until the token cap). min_p keeps the
    # distribution sharp where the model is confident and still leaves a tail
    # where it genuinely is not, and the repetition penalty covers the rest.
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--think", action="store_true",
                   help="start with think mode on (see /think)")
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
    think_mode = args.think

    # SFT teaches a plain-text scratchpad -- "Thinking: ...\nAnswer: ..." --
    # because the tokenizer's six special tokens are fixed by the pretrained
    # checkpoint and a seventh would invalidate every merge id. Dim the working
    # so the answer is what stands out, but never hide it: the whole reason to
    # show reasoning from a model this small is so a reader can see when it is
    # nonsense.
    DIM, RESET = "\033[2m", "\033[0m"

    def ask(text: str) -> str:
        """Prefix a think instruction when think mode is on."""
        if not think_mode:
            return text
        return f"{THINK_PROMPTS[0]}\n\n{text}"

    def respond() -> str:
        prompt_ids, dropped = session.build_prompt(params.max_new_tokens)
        if dropped:
            print(f"[dropped {dropped} old turn(s) to fit the "
                  f"{cfg.model.n_seq}-token context]")
        print("assistant: ", end="", flush=True)
        pieces = []
        dimmed = False
        try:
            for delta in stream_reply(model, tokenizer, prompt_ids, params,
                                      device, amp_dtype):
                pieces.append(delta)
                # Decide on the text so far, not on this fragment: the prefixes
                # are several tokens long and arrive split across deltas.
                so_far = "".join(pieces)
                inside = (THINK_PREFIX in so_far
                          and ANSWER_PREFIX not in so_far)
                if inside and not dimmed:
                    print(DIM, end="", flush=True)
                    dimmed = True
                elif dimmed and not inside:
                    print(RESET, end="", flush=True)
                    dimmed = False
                print(delta, end="", flush=True)
        except KeyboardInterrupt:
            print("  [interrupted]", end="")
        finally:
            if dimmed:
                print(RESET, end="", flush=True)
        print()
        return "".join(pieces)

    # -- one-shot ----------------------------------------------------------
    if args.prompt and args.once:
        session.add_user(ask(args.prompt))
        respond()
        return 0

    # -- interactive -------------------------------------------------------
    label = f"{stage or 'model'}" + (f" @ step {step}" if step else "")
    print(f"nanollm chat -- {os.path.basename(path)} ({label})")
    print(f"{model.get_param_count()/1e6:.0f}M params | ctx {cfg.model.n_seq} "
          f"| {device} | {params.describe()}")
    print("/help for commands, /exit to quit.\n")

    if args.prompt:
        session.add_user(ask(args.prompt))
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
                session.add_user(last_user["content"])   # already prefixed
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
            if command == "think":
                if rest in ("on", "off"):
                    think_mode = rest == "on"
                elif not rest:
                    think_mode = not think_mode
                else:
                    print("[usage: /think [on|off]]")
                    continue
                print(f"[think mode {'on' if think_mode else 'off'}]")
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

        session.add_user(ask(line))
        session.add_assistant(respond())

    print("bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
