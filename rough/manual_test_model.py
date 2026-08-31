"""Interactive chat REPL for a trained checkpoint.

Defaults to the SFT checkpoint and the SFT chat template:

    <|BOS|> <|USER|> ...prompt... <|ASSISTANT|> ...reply... <|EOS|>

Pass --raw (or /raw in the REPL) to talk to a pretrain checkpoint instead,
which gets plain text completion with no role markers.
"""

import argparse
import os
import re

import torch

from tokenizer import Tokenizer
from simple_transformer import Transformer
from config import CONFIG

ARTIFACTS_DIR = "artifacts"
DEFAULT_CHECKPOINT = f"{ARTIFACTS_DIR}/dpo_checkpoint_latest.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def infer_arch(state_dict):
    """Recover the model geometry from the weights themselves.

    Checkpoints written by the training scripts carry a `config` dict, but the
    bare `state_dict` snapshots do not, and CONFIG is not a reliable stand-in
    (its key names have drifted from what the scripts read).
    """
    vocab_size, n_dim = state_dict["l_embeddings.weight"].shape
    n_layer = 1 + max(int(m.group(1)) for m in
                      (re.match(r"blocks\.(\d+)\.", k) for k in state_dict)
                      if m)
    # q_norm is per-head-dim, so it names head_dim directly.
    head_dim = state_dict["blocks.0.attn.q_norm.weight"].shape[0]
    n_head = n_dim // head_dim
    n_kv_head = state_dict["blocks.0.attn.k_proj.weight"].shape[0] // head_dim
    return {
        "vocab_size": int(vocab_size),
        "n_dim": int(n_dim),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_kv_head": int(n_kv_head),
    }


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state, saved = ck["model_state_dict"], ck.get("config") or {}
        step, val = ck.get("global_step"), ck.get("val_loss")
    else:
        state, saved, step, val = ck, {}, None, None

    arch = infer_arch(state)
    # n_seq is not recoverable from the weights (RoPE is computed on the fly),
    # so take the saved value, else CONFIG, else the training default.
    n_seq = saved.get("n_seq") or CONFIG.get("sequence_length") or 1024

    model = Transformer(
        vocab_size=arch["vocab_size"],
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_dim=arch["n_dim"],
        n_seq=n_seq,
        n_kv_head=arch["n_kv_head"],
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    desc = (f"{arch['n_layer']}L x {arch['n_dim']}d x {arch['n_head']}h "
            f"(kv {arch['n_kv_head']}), vocab {arch['vocab_size']}, seq {n_seq}")
    if step is not None:
        desc += f" | step {step}"
    if isinstance(val, float):
        desc += f" | val {val:.4f}"
    return model, n_seq, desc


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_chat_ids(tokenizer, history, prompt, max_prompt_len):
    """Render history + new prompt, ending at the <|ASSISTANT|> marker.

    Same layout as sft_dataset.render_conversation, but left open for the model
    to complete: the assistant marker is the last token, and no EOS follows.
    Oldest turns are dropped until the prompt fits.
    """
    sp = tokenizer.special_tokens
    bos, eos = sp["<|BOS|>"], sp["<|EOS|>"]
    user, assistant = sp["<|USER|>"], sp["<|ASSISTANT|>"]

    turns = [[user] + tokenizer.encode(u)
             + [assistant] + tokenizer.encode(a) + [eos]
             for u, a in history]
    current = [user] + tokenizer.encode(prompt) + [assistant]

    while turns and 1 + sum(len(t) for t in turns) + len(current) > max_prompt_len:
        turns.pop(0)                       # drop the oldest exchange
    ids = [bos] + [t for turn in turns for t in turn] + current
    if len(ids) > max_prompt_len:
        # A single prompt too long for the window: keep the tail, which holds
        # the actual question, plus the markers that frame it.
        ids = [bos, user] + ids[-(max_prompt_len - 2):]
    return ids


def build_raw_ids(tokenizer, prompt, max_prompt_len):
    ids = [tokenizer.special_tokens["<|BOS|>"]] + tokenizer.encode(prompt)
    return ids[:max_prompt_len]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(model, tokenizer, ids, opts, valid_vocab_size, n_seq):
    eos = tokenizer.special_tokens["<|EOS|>"]
    max_new = min(opts["max_new"], n_seq - len(ids))
    if max_new <= 0:
        raise ValueError(f"prompt is {len(ids)} tokens, no room left in {n_seq}")

    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(
        x,
        max_count=max_new,
        temperature=opts["temperature"],
        top_k=opts["top_k"],
        top_p=opts["top_p"],
        repetition_penalty=opts["repetition_penalty"],
        eos_token_id=eos,
        valid_vocab_size=valid_vocab_size,
    )
    new = out[0, len(ids):].tolist()
    if eos in new:
        new = new[:new.index(eos)]         # cut before decode; decode() would
    return tokenizer.decode(new).strip()   # otherwise print "<|EOS|>" verbatim


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  /help                 show this message
  /reset                clear the conversation history
  /raw                  toggle raw completion mode (no chat template)
  /maxlen N             max new tokens          (current: {max_new})
  /temp F               sampling temperature    (current: {temperature})
  /topk N               top-k, 0 disables       (current: {top_k})
  /topp F               top-p                   (current: {top_p})
  /penalty F            repetition penalty      (current: {repetition_penalty})
  /settings             show the above
  /history              show the conversation so far
  /exit, /quit          leave
Anything else is sent to the model.
"""


def repl(model, tokenizer, opts, valid_vocab_size, n_seq):
    history = []                          # [(user, assistant), ...]
    print(HELP_TEXT.format(**opts))

    def set_num(line, key, cast, check=lambda v: True):
        parts = line.split()
        if len(parts) != 2:
            print(f"usage: {parts[0]} <value>")
            return
        try:
            value = cast(parts[1])
        except ValueError:
            print(f"not a number: {parts[1]}")
            return
        if not check(value):
            print(f"out of range: {value}")
            return
        opts[key] = value
        print(f"{key} = {value}")

    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            print(HELP_TEXT.format(**opts))
            continue
        if line == "/settings":
            print(HELP_TEXT.format(**opts))
            continue
        if line == "/reset":
            history.clear()
            print("history cleared")
            continue
        if line == "/raw":
            opts["raw"] = not opts["raw"]
            print(f"raw completion mode {'on' if opts['raw'] else 'off'}")
            continue
        if line == "/history":
            if not history:
                print("(empty)")
            for u, a in history:
                print(f"  user:      {u}")
                print(f"  assistant: {a}")
            continue
        if line.startswith("/maxlen"):
            set_num(line, "max_new", int, lambda v: v > 0)
            continue
        if line.startswith("/temp"):
            set_num(line, "temperature", float, lambda v: v > 0)
            continue
        if line.startswith("/topk"):
            set_num(line, "top_k", int, lambda v: v >= 0)
            continue
        if line.startswith("/topp"):
            set_num(line, "top_p", float, lambda v: 0 < v <= 1)
            continue
        if line.startswith("/penalty"):
            set_num(line, "repetition_penalty", float, lambda v: v > 0)
            continue
        if line.startswith("/"):
            print(f"unknown command: {line.split()[0]} (try /help)")
            continue

        room = n_seq - opts["max_new"]
        if opts["raw"]:
            ids = build_raw_ids(tokenizer, line, room)
        else:
            ids = build_chat_ids(tokenizer, history, line, room)

        try:
            reply = generate(model, tokenizer, ids, opts, valid_vocab_size, n_seq)
        except Exception as e:
            print(f"generation failed: {e}")
            continue

        print(reply if reply else "(empty reply)")
        if not opts["raw"]:
            history.append((line, reply))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"checkpoint to chat with (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer file (default: from CONFIG vocab_size)")
    p.add_argument("--raw", action="store_true",
                   help="plain completion, no chat template (for pretrain checkpoints)")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--penalty", type=float, default=1.1,
                   help="repetition penalty, 1.0 disables")
    p.add_argument("--prompt", default=None,
                   help="answer this single prompt and exit (no REPL)")
    args = p.parse_args()

    print(f"Using device: {DEVICE}")

    vocab_size = CONFIG.get("vocab_size", 20000)
    tok_file = args.tokenizer or f"{ARTIFACTS_DIR}/tokenizer-{vocab_size}.txt"
    if not os.path.exists(tok_file):
        raise FileNotFoundError(f"{tok_file} not found -- train the tokenizer first")
    print(f"Loading tokenizer from {tok_file}...")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.load(tok_file)
    actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else vocab_size

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint} not found")
    print(f"Loading model from {args.checkpoint}...")
    model, n_seq, desc = load_model(args.checkpoint)
    print(f"  {desc}")
    print(f"  {model.get_param_count():,} params")
    if model.vocab_size < actual_vocab_size:
        raise ValueError(
            f"checkpoint vocab {model.vocab_size} is smaller than the "
            f"tokenizer's {actual_vocab_size} -- mismatched tokenizer file")

    opts = {
        "max_new": args.max_new,
        "temperature": args.temp,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.penalty,
        "raw": args.raw,
    }

    if args.prompt is not None:
        room = n_seq - opts["max_new"]
        ids = (build_raw_ids(tokenizer, args.prompt, room) if opts["raw"]
               else build_chat_ids(tokenizer, [], args.prompt, room))
        print(generate(model, tokenizer, ids, opts, actual_vocab_size, n_seq))
        return

    repl(model, tokenizer, opts, actual_vocab_size, n_seq)


if __name__ == "__main__":
    main()
