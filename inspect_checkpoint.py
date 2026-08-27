import argparse
import collections
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
    q_key = next(k for k in state if k.endswith("q_proj.weight"))
    k_key = next(k for k in state if k.endswith("k_proj.weight"))
    kv_dim = state[k_key].shape[0]

    n_head = cfg.get("n_head") or CONFIG.get("n_heads") or 10
    if n_dim % n_head:
        n_head = next(h for h in (16, 12, 10, 8, 4, 2) if n_dim % h == 0)
    head_dim = n_dim // n_head
    n_kv_head = max(1, kv_dim // head_dim)

    n_seq = cfg.get("n_seq") or CONFIG.get("seq_len", 1024)

    print(f"  vocab={vocab} n_dim={n_dim} n_layer={n_layer} "
          f"n_head={n_head} n_kv_head={n_kv_head} n_seq={n_seq}")

    model = Transformer(vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                        n_dim=n_dim, n_seq=n_seq, n_kv_head=n_kv_head)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {list(missing)[:6]}")
    if unexpected:
        print(f"  unexpected keys: {list(unexpected)[:6]}")

    step = ck.get("global_step") if isinstance(ck, dict) else None
    logged = None
    if isinstance(ck, dict) and isinstance(ck.get("history"), dict):
        losses = ck["history"].get("train_loss") or []
        logged = losses[-1] if losses else None

    return model.to(device).eval(), step, logged


def get_blocks(model):
    """Uniform (attn, ffn, attn_norm, ffn_norm) tuples for either layout."""
    if hasattr(model, "blocks"):
        return [(b.attn, b.ffn, b.attn_norm, b.ffn_norm) for b in model.blocks]
    return list(zip(model.attentions, model.ffns, model.attn_norms, model.ffn_norms))


def load_batches(tokenizer, seq_len, batch, n_batches, text_file=None):
    """Real tokens if we can get them. Activation stats on random ids are
    meaningless, so say so loudly if we fall back."""
    chunk = seq_len + 1
    eos = tokenizer.special_tokens["<|EOS|>"]
    need = batch * n_batches
    seqs, buffer = [], []

    if text_file and os.path.exists(text_file):
        with open(text_file, errors="ignore") as f:
            buffer = tokenizer.encode(f.read())
        while len(buffer) >= chunk and len(seqs) < need:
            seqs.append(buffer[:chunk])
            buffer = buffer[chunk:]
    else:
        try:
            from pretrain_dataset import PretrainTextDataset
            ds = PretrainTextDataset(batch_size=16, min_chunk_size=1024,
                                     max_chunk_size=2048)
            for i in range(len(ds)):
                for text in ds[i]:
                    if isinstance(text, str):
                        buffer.extend(tokenizer.encode(text))
                        buffer.append(eos)
                while len(buffer) >= chunk:
                    seqs.append(buffer[:chunk])
                    buffer = buffer[chunk:]
                    if len(seqs) >= need:
                        break
                if len(seqs) >= need:
                    break
        except Exception as e:
            print(f"  could not load real data ({e})")

    if not seqs:
        print("  !! WARNING: using RANDOM tokens. Activation, attention and")
        print("     loss panels below are NOT meaningful.")
        return None

    blocks = [torch.tensor(seqs[i:i + batch], dtype=torch.long)
              for i in range(0, len(seqs) - batch + 1, batch)]
    print(f"  {len(blocks)} batches of {batch} x {seq_len} real tokens")
    return blocks


# ---------------------------------------------------------------------------
# Instrumented forward pass
# ---------------------------------------------------------------------------

class Probe:
    """Captures residual norms, block outputs, FFN activations and attention
    probabilities during a forward pass."""

    def __init__(self, model):
        self.model = model
        self.blocks = get_blocks(model)
        self.reset()
        self.handles = []

    def reset(self):
        n = len(self.blocks)
        self.resid_rms = [[] for _ in range(n + 1)]
        self.attn_out_rms = [[] for _ in range(n)]
        self.ffn_out_rms = [[] for _ in range(n)]
        self.ffn_active = [None] * n
        self.attn_entropy = [None] * n
        self.attn_distance = [None] * n

    def __enter__(self):
        for i, (attn, ffn, _, _) in enumerate(self.blocks):
            self.handles.append(attn.register_forward_hook(self._attn_hook(i)))
            self.handles.append(ffn.register_forward_hook(self._ffn_hook(i)))
            inner = getattr(ffn, "ffn", ffn)          # FFN wrapper vs raw SwiGLU
            if hasattr(inner, "g"):
                self.handles.append(inner.g.register_forward_hook(self._gate_hook(i)))
        if hasattr(self.model, "blocks"):
            for i, b in enumerate(self.model.blocks):
                self.handles.append(b.register_forward_hook(self._resid_hook(i + 1)))
        self.handles.append(
            self.model.l_embeddings.register_forward_hook(self._resid_hook(0)))

        # SDPA hides the attention matrix, so wrap it and recompute the softmax
        # for analysis only. Capped by --seq-len to keep this affordable.
        self._orig_sdpa = F.scaled_dot_product_attention
        probe = self

        def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=None, **kw):
            probe._capture_attention(q, k, is_causal)
            return probe._orig_sdpa(q, k, v, attn_mask=attn_mask,
                                    dropout_p=dropout_p, is_causal=is_causal,
                                    scale=scale, **kw)

        F.scaled_dot_product_attention = patched
        self._layer_cursor = 0
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        F.scaled_dot_product_attention = self._orig_sdpa

    @staticmethod
    def _rms(t):
        return t.float().pow(2).mean(dim=-1).sqrt().mean().item()

    def _resid_hook(self, idx):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            self.resid_rms[idx].append(self._rms(t))
        return hook

    def _attn_hook(self, idx):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            self.attn_out_rms[idx].append(self._rms(t))
        return hook

    def _ffn_hook(self, idx):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            self.ffn_out_rms[idx].append(self._rms(t))
        return hook

    def _gate_hook(self, idx):
        def hook(_m, _i, out):
            act = F.silu(out.float())
            # a unit is "active" for a token if |silu(gate)| is non-trivial
            frac = (act.abs() > 1e-2).float().mean(dim=(0, 1)).cpu().numpy()
            prev = self.ffn_active[idx]
            self.ffn_active[idx] = frac if prev is None else (prev + frac) / 2
        return hook

    def _capture_attention(self, q, k, is_causal):
        idx = self._layer_cursor % len(self.blocks)
        self._layer_cursor += 1
        with torch.no_grad():
            scale = 1.0 / math.sqrt(q.size(-1))
            logits = (q.float() @ k.float().transpose(-1, -2)) * scale
            T = logits.size(-1)
            if is_causal:
                mask = torch.ones(T, T, dtype=torch.bool, device=logits.device).tril()
                logits = logits.masked_fill(~mask, float("-inf"))
            p = logits.softmax(dim=-1)

            ent = -(p * (p + 1e-12).log()).sum(-1)          # (B, H, T)
            # normalize by the entropy of attending uniformly over the causal
            # window, so early positions aren't unfairly scored as collapsed
            window = torch.arange(1, T + 1, device=p.device).float().log()
            norm_ent = (ent / window.clamp(min=1e-6)).mean(dim=(0, 2))

            pos = torch.arange(T, device=p.device).float()
            dist = (pos[None, None, :, None] - pos[None, None, None, :]).abs()
            mean_dist = (p * dist).sum(-1).mean(dim=(0, 2))

            self.attn_entropy[idx] = norm_ent.cpu().numpy()
            self.attn_distance[idx] = mean_dist.cpu().numpy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

PROJ_GROUPS = {
    "q_proj": "Q", "k_proj": "K", "v_proj": "V", "wo": "attn out",
    "g": "FFN gate", "u": "FFN up", "d": "FFN down",
}


def weight_stats(model):
    blocks = get_blocks(model)
    rms = {label: [] for label in PROJ_GROUPS.values()}
    stable_rank = {label: [] for label in PROJ_GROUPS.values()}

    for attn, ffn, _, _ in blocks:
        inner = getattr(ffn, "ffn", ffn)
        modules = {"q_proj": attn.q_proj, "k_proj": attn.k_proj,
                   "v_proj": attn.v_proj, "wo": attn.wo}
        for name in ("g", "u", "d"):
            if hasattr(inner, name):
                modules[name] = getattr(inner, name)

        for key, label in PROJ_GROUPS.items():
            m = modules.get(key)
            if m is None:
                rms[label].append(np.nan)
                stable_rank[label].append(np.nan)
                continue
            w = m.weight.detach().float()
            rms[label].append(w.pow(2).mean().sqrt().item())
            # stable rank = ||W||_F^2 / ||W||_2^2; low means rank collapse
            sv = torch.linalg.svdvals(w)
            stable_rank[label].append(
                (sv.pow(2).sum() / sv[0].pow(2)).item() / min(w.shape))

    gains = {"attn_norm": [], "ffn_norm": []}
    for _, _, an, fn in blocks:
        gains["attn_norm"].append(an.weight.detach().float().cpu().numpy())
        gains["ffn_norm"].append(fn.weight.detach().float().cpu().numpy())

    return rms, stable_rank, gains


@torch.no_grad()
def loss_structure(model, batches, device, vocab):
    """Per-position loss and loss bucketed by target-token frequency."""
    pos_sum = None
    pos_n = 0
    tgt_counter = collections.Counter()
    per_token = []

    for block in batches:
        x = block[:, :-1].to(device)
        y = block[:, 1:].to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             y.reshape(-1), reduction="none").view(y.shape)
        s = ce.sum(dim=0).cpu().numpy()
        pos_sum = s if pos_sum is None else pos_sum + s
        pos_n += y.size(0)
        tgt_counter.update(y.flatten().cpu().tolist())
        per_token.append((y.flatten().cpu().numpy(), ce.flatten().cpu().numpy()))

    pos_loss = pos_sum / pos_n

    ranks = {tok: i for i, (tok, _) in enumerate(tgt_counter.most_common())}
    n_ranks = max(1, len(ranks))
    bucket_sum = np.zeros(10)
    bucket_n = np.zeros(10)
    for toks, losses in per_token:
        b = np.array([min(9, int(10 * ranks.get(int(t), n_ranks) / n_ranks))
                      for t in toks])
        for i in range(10):
            sel = b == i
            bucket_sum[i] += losses[sel].sum()
            bucket_n[i] += sel.sum()
    freq_loss = bucket_sum / np.maximum(1, bucket_n)

    overall = float(sum(s.sum() for _, s in per_token) /
                    sum(len(s) for _, s in per_token))

    # unigram floor from these same targets
    total = sum(tgt_counter.values())
    unigram_H = -sum((c / total) * math.log(c / total)
                     for c in tgt_counter.values())

    return pos_loss, freq_loss, overall, unigram_H, len(tgt_counter)


@torch.no_grad()
def embedding_stats(model, tokenizer):
    w = model.l_embeddings.weight.detach().float()
    norms = w.norm(dim=1).cpu().numpy()
    real_vocab = (max(tokenizer.vocab) + 1) if tokenizer.vocab else len(norms)
    return norms, real_vocab


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def build_figure(args, meta, rms, stable_rank, gains, probe, pos_loss,
                 freq_loss, overall, unigram_H, distinct, emb_norms, real_vocab):
    n_layer = len(rms["Q"])
    layers = np.arange(1, n_layer + 1)
    verdicts = []

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.26)
    fig.suptitle(
        f"Checkpoint internals — {os.path.basename(args.checkpoint)} — "
        f"step {meta['step']} — {meta['params']:,} params",
        fontsize=14, y=0.985)

    # 1: weight RMS by depth
    ax = fig.add_subplot(gs[0, 0])
    for label, vals in rms.items():
        ax.plot(layers, vals, marker="o", ms=3, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_title("Weight RMS by depth")
    ax.set_xlabel("layer")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    wo = np.array(rms["attn out"], dtype=float)
    q = np.array(rms["Q"], dtype=float)
    if np.nanmax(wo) / max(1e-9, np.nanmin(wo)) > 20:
        verdicts.append(("warn", "attn-out weight RMS varies >20x across depth"))
    if np.nanmean(wo) > np.nanmean(q):
        verdicts.append(("warn", "residual-path weights larger than Q; init "
                                 "scaling may have been skipped"))

    # 2: RMSNorm gains
    ax = fig.add_subplot(gs[0, 1])
    for name, arrs in gains.items():
        means = [a.mean() for a in arrs]
        stds = [a.std() for a in arrs]
        ax.errorbar(layers, means, yerr=stds, marker="o", ms=3, lw=1.2,
                    capsize=2, label=name)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_title("RMSNorm gain (mean ± std)\ndotted = untrained init")
    ax.set_xlabel("layer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    all_gain_std = np.mean([a.std() for arrs in gains.values() for a in arrs])
    if all_gain_std < 0.01:
        verdicts.append(("bad", "norm gains still at init — layers barely trained"))

    # 3: residual stream growth
    ax = fig.add_subplot(gs[0, 2])
    resid = [np.mean(v) for v in probe.resid_rms if v]
    if resid:
        ax.plot(range(len(resid)), resid, marker="o", ms=3, color="tab:blue")
        ax.set_yscale("log")
        growth = resid[-1] / max(1e-9, resid[0])
        ax.set_title(f"Residual stream RMS\ngrowth {growth:.1f}x")
        if growth > 50:
            verdicts.append(("warn", f"residual norm grows {growth:.0f}x — "
                                     "check init scaling"))
        elif growth < 1.5:
            verdicts.append(("warn", "residual norm nearly flat — layers may "
                                     "be contributing little"))
    ax.set_xlabel("depth (0 = embedding)")
    ax.grid(alpha=0.3)

    # 4: per-block contribution
    ax = fig.add_subplot(gs[0, 3])
    a_rms = np.array([np.mean(v) if v else np.nan for v in probe.attn_out_rms])
    f_rms = np.array([np.mean(v) if v else np.nan for v in probe.ffn_out_rms])
    base = np.array(resid[:-1]) if len(resid) > 1 else np.ones_like(a_rms)
    if len(base) == len(a_rms):
        ax.plot(layers, a_rms / base, marker="o", ms=3, label="attn / residual")
        ax.plot(layers, f_rms / base, marker="s", ms=3, label="ffn / residual")
        ax.set_yscale("log")
        ratios = np.nan_to_num(np.maximum(a_rms / base, f_rms / base))
        dead = [int(i) + 1 for i, r in enumerate(ratios) if r < 0.01]
        if dead:
            verdicts.append(("bad", f"near-dead layers (contribution <1%): {dead}"))
    ax.set_title("Block output vs residual\n(dead layers show as ~0)")
    ax.set_xlabel("layer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 5: attention entropy heatmap
    ax = fig.add_subplot(gs[1, 0])
    ent = [e for e in probe.attn_entropy if e is not None]
    if ent:
        mat = np.stack(ent)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Attention entropy / uniform\n0 = one token, 1 = uniform")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        collapsed = float((mat < 0.15).mean())
        if collapsed > 0.5:
            verdicts.append(("warn", f"{100*collapsed:.0f}% of heads are "
                                     "near-collapsed onto single tokens"))
        if float((mat > 0.9).mean()) > 0.5:
            verdicts.append(("bad", "most heads attend almost uniformly — "
                                    "attention has not learned structure"))

    # 6: attention distance
    ax = fig.add_subplot(gs[1, 1])
    dist = [d for d in probe.attn_distance if d is not None]
    if dist:
        mat = np.stack(dist)
        ax.plot(layers[:len(mat)], mat.mean(axis=1), marker="o", ms=3,
                color="tab:purple", label="mean over heads")
        ax.fill_between(layers[:len(mat)], mat.min(axis=1), mat.max(axis=1),
                        alpha=0.2, color="tab:purple", label="head range")
        ax.set_title("Mean attended distance (tokens)")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        if mat.mean() < 3:
            verdicts.append(("warn", "heads look only ~1-2 tokens back; model "
                                     "is behaving like an n-gram"))

    # 7: FFN dead units
    ax = fig.add_subplot(gs[1, 2])
    act = [a for a in probe.ffn_active if a is not None]
    if act:
        dead_frac = [float((a < 0.01).mean()) for a in act]
        ax.bar(layers[:len(dead_frac)], dead_frac, color="tab:red", alpha=0.7)
        ax.set_title("Dead FFN units per layer\n(active for <1% of tokens)")
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
        if max(dead_frac) < 0.01:
            ax.text(0.5, 0.5, "no dead units", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        if max(dead_frac) > 0.5:
            verdicts.append(("warn", f"up to {100*max(dead_frac):.0f}% of FFN "
                                     "units are dead in some layer"))

    # 8: stable rank
    ax = fig.add_subplot(gs[1, 3])
    for label, vals in stable_rank.items():
        ax.plot(layers, vals, marker="o", ms=3, lw=1.2, label=label)
    ax.set_title("Normalized stable rank\n(low = rank collapse)")
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    flat = [v for vals in stable_rank.values() for v in vals if not np.isnan(v)]
    if flat and np.mean(flat) < 0.05:
        verdicts.append(("warn", "very low stable rank — weights are nearly "
                                 "low-rank, capacity underused"))

    # 9: embedding norms
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(emb_norms[:real_vocab], bins=60, alpha=0.75, label="real tokens")
    if real_vocab < len(emb_norms):
        ax.hist(emb_norms[real_vocab:], bins=20, alpha=0.75, color="tab:red",
                label="padded rows")
    ax.axvline(emb_norms[0], color="k", ls="--", lw=1,
               label=f"token 0 ({emb_norms[0]:.3f})")
    ax.set_title("Embedding row norms")
    ax.set_xlabel("L2 norm")
    ax.legend(fontsize=6)
    if emb_norms[0] < 1e-6:
        verdicts.append(("bad", "token 0 embedding is all zeros — padding_idx "
                                "is frozen and, if tied, its logit too"))
    tiny = int((emb_norms[:real_vocab] < 0.1 * np.median(emb_norms[:real_vocab])).sum())
    if tiny > 0.05 * real_vocab:
        verdicts.append(("warn", f"{tiny} token embeddings are near zero — "
                                 "unused vocabulary"))

    # 10: per-position loss
    ax = fig.add_subplot(gs[2, 1])
    if pos_loss is not None:
        ax.plot(np.arange(1, len(pos_loss) + 1), pos_loss, lw=0.8,
                color="tab:blue", alpha=0.5)
        k = max(1, len(pos_loss) // 64)
        smooth = np.convolve(pos_loss, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(smooth)) + k, smooth, lw=1.8, color="tab:red")
        ax.set_xscale("log")
        ax.set_title("Loss vs position in sequence\n(should decrease)")
        ax.set_xlabel("position")
        ax.grid(alpha=0.3)
        early = pos_loss[:len(pos_loss) // 8].mean()
        late = pos_loss[-len(pos_loss) // 4:].mean()
        drop = early - late
        ax.axhline(late, color="k", ls=":", lw=0.8)
        if drop < 0.05:
            verdicts.append(("bad", f"loss flat across position ({drop:+.3f}) — "
                                    "model is not using long context"))
        else:
            verdicts.append(("ok", f"context helps: loss drops {drop:.2f} nats "
                                   "from early to late positions"))

    # 11: loss by target frequency
    ax = fig.add_subplot(gs[2, 2])
    if freq_loss is not None:
        ax.bar(np.arange(10), freq_loss, color="tab:green", alpha=0.75)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{i*10}" for i in range(10)], fontsize=6)
        ax.set_title("Loss by target frequency decile\n(0 = most frequent)")
        ax.set_xlabel("decile")
        ax.grid(alpha=0.3, axis="y")
        if freq_loss[0] < 0.5 and freq_loss[-1] > 6:
            verdicts.append(("warn", "only frequent tokens are predicted well — "
                                     "close to a unigram model"))

    # 12: verdict
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    lines = [f"step: {meta['step']}"]
    if meta["logged"] is not None:
        lines.append(f"logged train loss: {meta['logged']:.4f}")
    if overall is not None:
        lines.append(f"recomputed loss: {overall:.4f}  (ppl {math.exp(min(20, overall)):.1f})")
        lines.append(f"unigram floor: {unigram_H:.4f}")
        lines.append(f"beats unigram by: {unigram_H - overall:.4f} nats")
        lines.append(f"distinct targets seen: {distinct}")
        if meta["logged"] is not None:
            gap = abs(meta["logged"] - overall)
            # a single logged minibatch is noisy; only a large, structured gap
            # (especially a clean 1/GRAD_ACCUM ratio) indicates a logging bug
            ratio = overall / max(1e-9, meta["logged"])
            if gap > 0.5 and any(abs(ratio - r) < 0.15 for r in (2, 4, 8)):
                verdicts.insert(0, ("bad", f"recomputed loss is {ratio:.1f}x the "
                                           f"logged value - GRAD_ACCUM division "
                                           f"bug in the logging path"))
            elif gap > 1.0:
                verdicts.insert(0, ("warn", f"logged {meta['logged']:.2f} vs "
                                            f"recomputed {overall:.2f}; check the "
                                            f"eval data matches training data"))
        if overall > unigram_H - 0.3:
            verdicts.insert(0, ("bad", "model barely beats token frequency"))
    lines.append("")

    order = {"bad": 0, "warn": 1, "ok": 2}
    colors = {"bad": "#b00020", "warn": "#b06000", "ok": "#0a6b2d"}
    marks = {"bad": "FAIL", "warn": "WARN", "ok": "OK  "}

    ax.text(0, 1.0, "\n".join(lines), va="top", fontsize=8.5, family="monospace",
            transform=ax.transAxes)
    y = 1.0 - 0.055 * (len(lines) + 1)
    ax.text(0, y, "findings", va="top", fontsize=9, weight="bold",
            transform=ax.transAxes)
    y -= 0.05
    if not verdicts:
        verdicts = [("ok", "no anomalies detected")]
    for kind, msg in sorted(verdicts, key=lambda v: order[v[0]]):
        wrapped = [msg[i:i + 44] for i in range(0, len(msg), 44)]
        ax.text(0, y, f"{marks[kind]} {wrapped[0]}", va="top", fontsize=7.5,
                family="monospace", color=colors[kind], transform=ax.transAxes)
        y -= 0.042
        for cont in wrapped[1:]:
            ax.text(0, y, f"     {cont}", va="top", fontsize=7.5,
                    family="monospace", color=colors[kind], transform=ax.transAxes)
            y -= 0.042
        if y < 0.02:
            break

    return fig, verdicts


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/pretrain_checkpoint_latest.pt")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", default="logs/checkpoint_internals.png")
    p.add_argument("--seq-len", type=int, default=256,
                   help="analysis length; attention capture is O(T^2)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--text-file", default=None,
                   help="use this text instead of the pretrain dataset")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = CONFIG.get("vocab_size", 20000)
    tok_path = args.tokenizer or f"artifacts/tokenizer-{vocab_size}.txt"
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.load(tok_path)

    print(f"Loading {args.checkpoint}")
    model, step, logged = load_model(args.checkpoint, device)

    print("Loading data")
    batches = load_batches(tokenizer, args.seq_len, args.batch, args.batches,
                           args.text_file)

    print("Weight statistics")
    rms, stable_rank, gains = weight_stats(model)
    emb_norms, real_vocab = embedding_stats(model, tokenizer)

    probe = Probe(model)
    pos_loss = freq_loss = overall = None
    unigram_H = float("nan")
    distinct = 0

    if batches:
        print("Instrumented forward pass")
        with probe, torch.no_grad():
            for block in batches[:2]:
                model(block[:, :-1].to(device))
        print("Loss structure")
        pos_loss, freq_loss, overall, unigram_H, distinct = loss_structure(
            model, batches, device, model.vocab_size)

    meta = {"step": step, "logged": logged, "params": model.get_param_count()}

    print("Rendering")
    fig, verdicts = build_figure(args, meta, rms, stable_rank, gains, probe,
                                 pos_loss, freq_loss, overall, unigram_H,
                                 distinct, emb_norms, real_vocab)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {args.out}\n")
    for kind, msg in sorted(verdicts, key=lambda v: {"bad": 0, "warn": 1, "ok": 2}[v[0]]):
        print(f"  [{kind.upper():4s}] {msg}")


if __name__ == "__main__":
    main()