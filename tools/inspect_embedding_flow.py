"""Visualize how a token's representation is transformed layer by layer.

Captures the residual stream at every depth, then renders one image covering:
geometry (PCA trajectory, rotation, norm growth, effective dimensionality,
anisotropy) and function (logit lens: what the model would predict if you read
out at each layer).

    python inspect_embedding_flow.py
    python inspect_embedding_flow.py --prompt "The capital of France is"
    python inspect_embedding_flow.py --checkpoint artifacts/sft_model.pt --seq-len 256

The logit lens works because the output head reads the residual stream through
final_norm. Applying that same readout to an intermediate layer shows what the
model "currently believes" the next token is at that depth.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# This file lives in tools/; put the repo root and src/ on the path so the
# legacy shims (config, simple_transformer, ...) and nanollm both resolve.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tokenizer import Tokenizer
from config import CONFIG
from simple_transformer import Transformer


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path, device):
    ck = torch.load(path, map_location="cpu")
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}

    emb_key = next(k for k in state if "l_embeddings.weight" in k)
    vocab, n_dim = state[emb_key].shape
    n_layer = 1 + max(
        int(k.split(".")[1]) for k in state
        if k.startswith(("blocks.", "attentions.")) and k.split(".")[1].isdigit()
    )
    k_key = next(k for k in state if k.endswith("k_proj.weight"))
    kv_dim = state[k_key].shape[0]

    n_head = cfg.get("n_head") or CONFIG.get("n_heads") or 10
    if n_dim % n_head:
        n_head = next(h for h in (16, 12, 10, 8, 4, 2) if n_dim % h == 0)
    n_kv_head = max(1, kv_dim // (n_dim // n_head))
    n_seq = cfg.get("n_seq") or CONFIG.get("seq_len", 1024)

    print(f"  vocab={vocab} n_dim={n_dim} n_layer={n_layer} n_head={n_head} n_seq={n_seq}")
    model = Transformer(vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                        n_dim=n_dim, n_seq=n_seq, n_kv_head=n_kv_head)
    model.load_state_dict(state, strict=False)
    step = ck.get("global_step") if isinstance(ck, dict) else None
    return model.to(device).eval(), step


def get_blocks(model):
    if hasattr(model, "blocks"):
        return list(model.blocks)
    return None


def load_tokens(tokenizer, seq_len, batch, text_file=None):
    chunk = seq_len + 1
    eos = tokenizer.special_tokens["<|EOS|>"]
    buffer, seqs = [], []

    if text_file and os.path.exists(text_file):
        with open(text_file, errors="ignore") as f:
            buffer = tokenizer.encode(f.read())
    else:
        try:
            from pretrain_dataset import PretrainTextDataset
            ds = PretrainTextDataset(batch_size=16, min_chunk_size=1024,
                                     max_chunk_size=2048)
            for i in range(len(ds)):
                for t in ds[i]:
                    if isinstance(t, str):
                        buffer.extend(tokenizer.encode(t))
                        buffer.append(eos)
                if len(buffer) >= chunk * batch:
                    break
        except Exception as e:
            print(f"  could not load dataset ({e})")

    while len(buffer) >= chunk and len(seqs) < batch:
        seqs.append(buffer[:chunk])
        buffer = buffer[chunk:]

    if len(seqs) < batch:
        print("  !! not enough real tokens; falling back to random ids")
        return None
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_stream(model, idx, device):
    """Residual stream after the embedding and after every block, plus each
    block's attention and FFN write vectors."""
    blocks = get_blocks(model)
    states = []
    attn_writes = []
    ffn_writes = []

    x = model.l_embeddings(idx.to(device))
    states.append(x.clone())

    cos, sin = model._get_rope(model.n_seq, x.device)

    if blocks is None:
        raise RuntimeError(
            "This script needs the `blocks` layout of simple_transformer.py")

    for block in blocks:
        a = block.attn(block.attn_norm(x), cos, sin, 0, None, None)
        attn_writes.append(a.clone())
        x = x + a
        f = block.ffn(block.ffn_norm(x))
        ffn_writes.append(f.clone())
        x = x + f
        states.append(x.clone())

    return states, attn_writes, ffn_writes


@torch.no_grad()
def logit_lens(model, states, targets):
    """Read out every layer through final_norm + the output head."""
    per_layer = []
    for h in states:
        logits = model.logit_proj(model.final_norm(h)).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1)).item()
        top1 = logits.argmax(-1)
        acc = (top1 == targets).float().mean().item()
        # rank of the true token: how close is it to the top?
        true_logit = logits.gather(-1, targets.unsqueeze(-1))
        rank = (logits > true_logit).sum(-1).float().mean().item()
        probs = logits.softmax(-1)
        ent = -(probs * (probs + 1e-12).log()).sum(-1).mean().item()
        per_layer.append({"loss": loss, "acc": acc, "rank": rank,
                          "entropy": ent, "top1": top1})
    return per_layer


@torch.no_grad()
def prompt_trace(model, tokenizer, prompt, device, topk=3):
    """Layer-by-layer top-k predictions for the last position of a prompt."""
    ids = tokenizer.encode(prompt)
    if not ids:
        return None, None
    ids = ids[-min(len(ids), model.n_seq):]
    idx = torch.tensor([ids], dtype=torch.long)
    states, _, _ = capture_stream(model, idx, device)

    rows = []
    for depth, h in enumerate(states):
        logits = model.logit_proj(model.final_norm(h[:, -1])).float()
        probs = logits.softmax(-1)[0]
        vals, tops = probs.topk(topk)
        preds = []
        for v, t in zip(vals.tolist(), tops.tolist()):
            piece = tokenizer.decode([t]).replace("\n", "\\n")
            preds.append(f"{piece[:12]!r}:{v:.2f}")
        rows.append((depth, "  ".join(preds)))
    return ids, rows


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def geometry(states, attn_writes, ffn_writes, max_tokens=800):
    n = len(states)
    flat = [h.reshape(-1, h.size(-1)).float().cpu() for h in states]
    n_tok = flat[0].size(0)
    sel = torch.randperm(n_tok)[:min(max_tokens, n_tok)]
    flat = [f[sel] for f in flat]

    stats = {
        "norm_med": [], "norm_q1": [], "norm_q3": [],
        "cos_prev": [], "cos_final": [], "cos_embed": [],
        "eff_dim": [], "anisotropy": [],
    }

    final = F.normalize(flat[-1], dim=-1)
    embed = F.normalize(flat[0], dim=-1)

    for i, f in enumerate(flat):
        norms = f.norm(dim=-1)
        stats["norm_med"].append(norms.median().item())
        stats["norm_q1"].append(norms.quantile(0.25).item())
        stats["norm_q3"].append(norms.quantile(0.75).item())

        u = F.normalize(f, dim=-1)
        stats["cos_prev"].append(
            (u * F.normalize(flat[i - 1], dim=-1)).sum(-1).mean().item()
            if i > 0 else 1.0)
        stats["cos_final"].append((u * final).sum(-1).mean().item())
        stats["cos_embed"].append((u * embed).sum(-1).mean().item())

        # effective dimensionality: participation ratio of the PCA spectrum,
        # normalized so 1.0 = variance spread evenly over all directions
        centered = f - f.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(centered)
        lam = sv.pow(2)
        stats["eff_dim"].append(
            (lam.sum().pow(2) / lam.pow(2).sum()).item() / f.size(-1))

        # anisotropy: mean cosine between random token pairs. High = all tokens
        # point the same way, i.e. representation collapse.
        perm = torch.randperm(u.size(0))
        stats["anisotropy"].append((u * u[perm]).sum(-1).mean().item())

    # block write geometry: does a block rotate the residual or just scale it?
    write_cos_attn, write_cos_ffn, write_rel_attn, write_rel_ffn = [], [], [], []
    for i, (a, fw) in enumerate(zip(attn_writes, ffn_writes)):
        resid = states[i].reshape(-1, states[i].size(-1)).float().cpu()[sel]
        av = a.reshape(-1, a.size(-1)).float().cpu()[sel]
        fv = fw.reshape(-1, fw.size(-1)).float().cpu()[sel]
        write_cos_attn.append(
            (F.normalize(av, dim=-1) * F.normalize(resid, dim=-1)).sum(-1).mean().item())
        write_cos_ffn.append(
            (F.normalize(fv, dim=-1) * F.normalize(resid, dim=-1)).sum(-1).mean().item())
        write_rel_attn.append((av.norm(dim=-1) / resid.norm(dim=-1).clamp(min=1e-6)).median().item())
        write_rel_ffn.append((fv.norm(dim=-1) / resid.norm(dim=-1).clamp(min=1e-6)).median().item())

    stats["write_cos_attn"] = write_cos_attn
    stats["write_cos_ffn"] = write_cos_ffn
    stats["write_rel_attn"] = write_rel_attn
    stats["write_rel_ffn"] = write_rel_ffn

    # joint PCA across all layers so positions are comparable between layers
    stacked = torch.cat(flat, dim=0)
    stacked = stacked - stacked.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(stacked, q=3)
    proj = [(f - stacked.mean(0, keepdim=True) * 0) @ V[:, :2] for f in flat]
    proj = [p.numpy() for p in proj]

    return stats, proj, sel, n


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def build_figure(args, meta, stats, proj, lens, trace_ids, trace_rows, tokenizer):
    n = len(stats["norm_med"])
    depths = np.arange(n)
    layers = np.arange(1, n)

    fig = plt.figure(figsize=(19, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.25)
    fig.suptitle(
        f"Residual stream transformation — {os.path.basename(args.checkpoint)} — "
        f"step {meta['step']} — {n - 1} layers",
        fontsize=14, y=0.99)

    cmap = plt.get_cmap("viridis")

    # 1: PCA trajectory of the token cloud through depth
    ax = fig.add_subplot(gs[0, 0])
    for i, p in enumerate(proj):
        ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.25, color=cmap(i / max(1, n - 1)))
    # follow a few individual tokens through every layer
    for tok in range(min(4, proj[0].shape[0])):
        path = np.array([p[tok] for p in proj])
        ax.plot(path[:, 0], path[:, 1], lw=1.0, color="crimson", alpha=0.8)
        ax.scatter(path[0, 0], path[0, 1], s=22, marker="o",
                   facecolors="none", edgecolors="crimson", lw=1.2)
        ax.scatter(path[-1, 0], path[-1, 1], s=26, marker="X", color="crimson")
    ax.set_title("Token cloud in joint PCA space\nred = 4 tracked tokens "
                 "(o start, X end)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max(1, n - 1)))
    fig.colorbar(sm, ax=ax, fraction=0.046, label="depth")

    # 2: per-layer PCA spread, small multiples
    ax = fig.add_subplot(gs[0, 1])
    spreads = [p.std(axis=0).mean() for p in proj]
    ax.plot(depths, spreads, marker="o", ms=3, color="tab:blue")
    ax.set_title("Spread of the token cloud (PC1-2 std)")
    ax.set_xlabel("depth (0 = embedding)")
    ax.grid(alpha=0.3)

    # 3: norm growth with spread
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(depths, stats["norm_med"], marker="o", ms=3, color="tab:blue",
            label="median")
    ax.fill_between(depths, stats["norm_q1"], stats["norm_q3"], alpha=0.25,
                    color="tab:blue", label="IQR")
    ax.set_yscale("log")
    growth = stats["norm_med"][-1] / max(1e-9, stats["norm_med"][0])
    ax.set_title(f"Residual norm per token\ngrowth {growth:.1f}x")
    ax.set_xlabel("depth")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 4: how much each layer rotates the representation
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(layers, stats["cos_prev"][1:], marker="o", ms=3, color="tab:orange")
    ax.set_title("cos(layer, previous layer)\n1.0 = layer changed nothing")
    ax.set_xlabel("layer")
    ax.set_ylim(min(0.5, min(stats["cos_prev"][1:]) - 0.05), 1.005)
    ax.grid(alpha=0.3)

    # 5: convergence toward the final representation
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(depths, stats["cos_final"], marker="o", ms=3, label="cos to final")
    ax.plot(depths, stats["cos_embed"], marker="s", ms=3, label="cos to embedding")
    ax.set_title("Where the representation locks in")
    ax.set_xlabel("depth")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 6: block write geometry
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(layers, stats["write_cos_attn"], marker="o", ms=3, label="attn write")
    ax.plot(layers, stats["write_cos_ffn"], marker="s", ms=3, label="ffn write")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_title("cos(block output, incoming residual)\n~0 = writes new "
                 "information, ~1 = amplifies")
    ax.set_xlabel("layer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 7: relative write magnitude
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(layers, stats["write_rel_attn"], marker="o", ms=3, label="attn")
    ax.plot(layers, stats["write_rel_ffn"], marker="s", ms=3, label="ffn")
    ax.set_yscale("log")
    ax.set_title("Write magnitude / residual magnitude")
    ax.set_xlabel("layer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 8: effective dimensionality + anisotropy
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(depths, stats["eff_dim"], marker="o", ms=3, color="tab:green",
            label="effective dim (frac of n_dim)")
    ax2 = ax.twinx()
    ax2.plot(depths, stats["anisotropy"], marker="s", ms=3, color="tab:red",
             label="anisotropy")
    ax.set_ylabel("effective dim", labelpad=1)
    ax2.set_ylabel("anisotropy", color="tab:red", labelpad=1)
    ax.set_title("Dimensionality vs collapse")
    ax.set_xlabel("depth")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # 9: logit lens loss + accuracy
    ax = fig.add_subplot(gs[2, 2])
    if lens:
        ax.plot(depths, [d["loss"] for d in lens], marker="o", ms=3,
                color="tab:blue", label="readout loss")
        ax.set_ylabel("cross entropy", labelpad=1)
        ax3 = ax.twinx()
        ax3.plot(depths, [100 * d["acc"] for d in lens], marker="s", ms=3,
                 color="tab:green", label="top-1 %")
        ax3.set_ylabel("top-1 %", color="tab:green", labelpad=1)
        ax.set_title("Logit lens: prediction quality by depth")
        ax.set_xlabel("depth")
        ax.legend(fontsize=7, loc="upper right")
        ax3.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)

    # 10: logit lens rank + entropy
    ax = fig.add_subplot(gs[3, 0])
    if lens:
        ax.plot(depths, [d["rank"] for d in lens], marker="o", ms=3,
                color="tab:purple", label="mean rank of true token")
        ax.set_yscale("log")
        ax.set_ylabel("rank", labelpad=1)
        ax4 = ax.twinx()
        ax4.plot(depths, [d["entropy"] for d in lens], marker="s", ms=3,
                 color="tab:orange", label="output entropy")
        ax4.set_ylabel("entropy", color="tab:orange", labelpad=1)
        ax.set_title("Logit lens: rank of the correct token")
        ax.set_xlabel("depth")
        ax.legend(fontsize=7, loc="upper right")
        ax4.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)

    # 11: agreement with the final prediction, per position
    ax = fig.add_subplot(gs[3, 1])
    if lens:
        final_top1 = lens[-1]["top1"]
        agree = np.stack([
            (d["top1"] == final_top1).float().mean(0).cpu().numpy()
            for d in lens
        ])
        im = ax.imshow(agree, aspect="auto", cmap="magma", vmin=0, vmax=1,
                       origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Agreement with final prediction\n(fraction of sequences)")
        ax.set_xlabel("position in sequence")
        ax.set_ylabel("depth")

    # 12: layer-by-layer prediction trace for the prompt
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    if trace_rows:
        decoded = tokenizer.decode(trace_ids).replace("\n", " ")
        header = f'prompt: {decoded[-46:]!r}\ndepth  top-3 next-token predictions'
        ax.text(0, 1.0, header, va="top", fontsize=8, family="monospace",
                weight="bold", transform=ax.transAxes)
        y = 0.86
        # thin the list if the model is deep, keeping first and last layers
        rows = trace_rows
        if len(rows) > 18:
            keep = set(range(3)) | set(range(len(rows) - 3, len(rows)))
            stride = max(1, len(rows) // 12)
            keep |= {i for i in range(len(rows)) if i % stride == 0}
            rows = [r for i, r in enumerate(rows) if i in keep]
        for depth, text in rows:
            ax.text(0, y, f"{depth:>4}   {text[:52]}", va="top", fontsize=7,
                    family="monospace", transform=ax.transAxes)
            y -= 0.048
            if y < 0.01:
                break

    return fig


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/pretrain_checkpoint_latest.pt")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", default="logs/embedding_flow.png")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--text-file", default=None)
    p.add_argument("--prompt", default="The history of the city began when")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = CONFIG.get("vocab_size", 20000)
    tok_path = args.tokenizer or f"artifacts/tokenizer-{vocab_size}.txt"
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.load(tok_path)

    print(f"Loading {args.checkpoint}")
    model, step = load_model(args.checkpoint, device)

    print("Loading tokens")
    block = load_tokens(tokenizer, args.seq_len, args.batch, args.text_file)
    if block is None:
        block = torch.randint(0, min(model.vocab_size, vocab_size),
                              (args.batch, args.seq_len + 1))

    xs = block[:, :-1]
    ys = block[:, 1:].to(device)

    print("Capturing residual stream")
    states, attn_writes, ffn_writes = capture_stream(model, xs, device)
    print(f"  {len(states)} depths x {tuple(states[0].shape)}")

    print("Logit lens")
    lens = logit_lens(model, states, ys)

    print("Geometry")
    stats, proj, _, _ = geometry(states, attn_writes, ffn_writes)

    print("Prompt trace")
    trace_ids, trace_rows = prompt_trace(model, tokenizer, args.prompt, device)

    print("Rendering")
    fig = build_figure(args, {"step": step}, stats, proj, lens,
                       trace_ids, trace_rows, tokenizer)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}")

    print("\nlogit lens summary (readout loss by depth):")
    for d, entry in enumerate(lens):
        bar = "#" * int(40 * entry["loss"] / max(e["loss"] for e in lens))
        print(f"  depth {d:>3}  loss {entry['loss']:6.3f}  "
              f"top1 {100*entry['acc']:5.1f}%  {bar}")


if __name__ == "__main__":
    main()