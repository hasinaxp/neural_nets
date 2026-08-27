

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SEQ_LEN = 1024
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 12
DEFAULT_NUM_EXPERTS = 1
DEFAULT_TOP_K = 1

IGNORE_INDEX = -100


class RMSNorm(nn.Module):

    def __init__(self, n_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)



def precompute_rope(head_dim: int, seq_len: int, device, base: float = 10000.0):
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D), already in x's dtype."""
    c = cos[None, None]
    s = sin[None, None]
    return x * c + rotate_half(x) * s


class KVCache:

    def __init__(self, batch_size, max_seq_len, num_layers, n_kv_head,
                 head_dim, device, dtype, **_legacy):
        self.max_seq_len = max_seq_len
        shape = (num_layers, batch_size, n_kv_head, max_seq_len, head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)

    def update(self, layer_idx, start_pos, k, v):
        t = k.size(2)
        end = start_pos + t
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} > {self.max_seq_len}")
        self.k[layer_idx, :, :, start_pos:end] = k.to(self.k.dtype)
        self.v[layer_idx, :, :, start_pos:end] = v.to(self.v.dtype)
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]


class AttentionGQA(nn.Module):

    def __init__(self, n_dim, n_head, layer_idx, rope_theta=10000.0,
                 n_kv_head=None):
        super().__init__()
        if n_dim % n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        self.n_dim = n_dim
        self.n_head = n_head
        self.head_dim = n_dim // n_head
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta

        n_kv_head = n_kv_head or max(1, n_head // 4)
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head

        # Legacy attribute names some callers construct KVCache from.
        self.qk_rope_head_dim = self.head_dim
        self.kv_lora_rank = self.head_dim

        self.q_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_dim, n_dim, bias=False)

        # QK-norm: normalizes per head before RoPE. Keeps attention logits from
        # drifting at high LR, which is the main instability at depth.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None, attn_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        c = cos[start_pos:start_pos + T].to(q.dtype)
        s = sin[start_pos:start_pos + T].to(q.dtype)
        q = apply_rope(q, c, s)
        k = apply_rope(k, c, s)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, start_pos, k, v)

        if self.n_rep > 1:
            # expand, not repeat_interleave: no copy, SDPA broadcasts fine
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)

        if attn_mask is not None:
            # Explicit mask (e.g. intra-document); is_causal must be False.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # THE FIX: causal whenever there is more than one query position.
            # The old condition disabled it during cached prefill, making the
            # prompt bidirectional at inference but causal in training.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, n_dim, hidden_dim):
        super().__init__()
        self.g = nn.Linear(n_dim, hidden_dim, bias=False)
        self.u = nn.Linear(n_dim, hidden_dim, bias=False)
        self.d = nn.Linear(hidden_dim, n_dim, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class FFN(nn.Module):
    """Kept for name compatibility; a dense SwiGLU block."""

    def __init__(self, n_dim, num_experts=1, top_k=1, num_shared_experts=1,
                 hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = swiglu_hidden_dim(n_dim)
        self.ffn = SwiGLU(n_dim, hidden_dim)

    def forward(self, x):
        return self.ffn(x)


def swiglu_hidden_dim(n_dim: int, multiple_of: int = 256) -> int:
    """8/3 * n_dim keeps SwiGLU param-matched to a 4x GELU FFN."""
    hidden = int(8 * n_dim / 3)
    return ((hidden + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-norm attention + FFN. A real module, so torch.compile can trace it
    once instead of seeing a freshly-built closure every step."""

    def __init__(self, n_dim, n_head, layer_idx, rope_theta, n_kv_head,
                 hidden_dim, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(n_dim)
        self.attn = AttentionGQA(n_dim, n_head, layer_idx, rope_theta, n_kv_head)
        self.ffn_norm = RMSNorm(n_dim)
        self.ffn = SwiGLU(n_dim, hidden_dim)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None, attn_mask=None):
        x = x + self.resid_dropout(
            self.attn(self.attn_norm(x), cos, sin, start_pos, kv_cache, attn_mask))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def default_n_kv_head(n_head: int) -> int:
    if n_head <= 4:
        kv = 1
    elif n_head <= 8:
        kv = 2
    elif n_head <= 16:
        kv = 4
    else:
        kv = 8
    while n_head % kv:
        kv -= 1
    return kv


def build_document_mask(idx: torch.Tensor, eos_id: int, n_head: int = 1):
    """Block-diagonal causal mask so tokens cannot attend across an EOS.

    You pack several documents into every 1024-token sequence; plain causal
    attention lets a token read the tail of the previous document. Most open
    models tolerate this. Pass the result as `attn_mask` if you'd rather not.
    Returns a bool mask of shape (B, 1, T, T), True = attend.
    """
    B, T = idx.shape
    # document id increments after each EOS
    doc = (idx == eos_id).cumsum(dim=1)
    doc = doc - (idx == eos_id).long()          # EOS belongs to its own document
    same_doc = doc[:, :, None] == doc[:, None, :]
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    return (same_doc & causal).unsqueeze(1)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer=DEFAULT_NUM_LAYERS,
        n_head=DEFAULT_NUM_HEADS,
        n_dim=DEFAULT_EMBEDDING_DIM,
        n_seq=DEFAULT_SEQ_LEN,
        num_experts=DEFAULT_NUM_EXPERTS,      # accepted, unused (dense)
        top_k=DEFAULT_TOP_K,                  # accepted, unused (dense)
        rope_theta=10000.0,
        num_shared_experts=1,                 # accepted, unused (dense)
        dropout=0.0,
        n_kv_head=None,
        tie_embeddings=True,
        activation_checkpointing=False,        # was hardcoded True
        loss_chunk_size=512,
        z_loss_weight=1e-4,
        debug_token_range=False,
        init_std=0.02,
    ):
        super().__init__()
        if n_dim % n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        head_dim = n_dim // n_head
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        if head_dim not in (32, 64, 128):
            # not fatal, just slow: the flash kernels are tuned for these
            import warnings
            warnings.warn(
                f"head_dim={head_dim} misses the fast attention kernels; "
                f"n_head={n_dim // 64} would give head_dim 64 at n_dim={n_dim}",
                stacklevel=2,
            )

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_seq = n_seq
        self.rope_theta = rope_theta
        self.activation_checkpointing = activation_checkpointing
        self.loss_chunk_size = loss_chunk_size
        self.z_loss_weight = z_loss_weight
        self.debug_token_range = debug_token_range
        self.init_std = init_std

        self.n_kv_head = n_kv_head or default_n_kv_head(n_head)
        if n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")

        # No padding_idx: with tied embeddings it permanently freezes the row-0
        # output logit at 0.0, so token 0 can never be predicted or learned.
        self.l_embeddings = nn.Embedding(vocab_size, n_dim)
        self.emb_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        hidden_dim = swiglu_hidden_dim(n_dim)
        self.blocks = nn.ModuleList([
            Block(n_dim, n_head, i, rope_theta, self.n_kv_head, hidden_dim, dropout)
            for i in range(n_layer)
        ])

        self.final_norm = RMSNorm(n_dim)
        self.logit_proj = nn.Linear(n_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.logit_proj.weight = self.l_embeddings.weight

        self._rope_cos = None
        self._rope_sin = None
        self._rope_cached_len = 0
        self._rope_device = None

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "d.weight")):
                nn.init.normal_(p, mean=0.0, std=init_std / math.sqrt(2 * n_layer))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


    def _get_rope(self, seq_len, device):
        if (self._rope_cos is None
                or seq_len > self._rope_cached_len
                or device != self._rope_device):
            self._rope_cos, self._rope_sin = precompute_rope(
                self.n_dim // self.n_head, seq_len, device, self.rope_theta)
            self._rope_cached_len = seq_len
            self._rope_device = device
        return self._rope_cos, self._rope_sin

    # -- forward ------------------------------------------------------------

    def forward_hidden(self, idx, start_pos=0, kv_cache=None, attn_mask=None):
        """Everything up to and including final_norm. Split out so the loss can
        project in chunks and generation can project only the last position."""
        if self.debug_token_range and idx.numel():
            # Two host syncs; enable for the first few steps, then turn off.
            lo, hi = idx.min().item(), idx.max().item()
            if lo < 0 or hi >= self.vocab_size:
                raise ValueError(
                    f"Token id range [{lo}, {hi}] outside [0, {self.vocab_size - 1}]")

        B, T = idx.shape
        if kv_cache is None and T > self.n_seq:
            raise ValueError(f"Sequence length {T} exceeds n_seq={self.n_seq}")

        x = self.emb_dropout(self.l_embeddings(idx))
        max_len = self.n_seq if kv_cache is None else kv_cache.max_seq_len
        cos, sin = self._get_rope(max_len, x.device)

        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, start_pos, kv_cache, attn_mask,
                    use_reentrant=False)
            else:
                x = block(x, cos, sin, start_pos, kv_cache, attn_mask)

        return self.final_norm(x)

    def forward(self, idx, start_pos=0, kv_cache=None, attn_mask=None):
        """Returns (logits, None). The second element is vestigial -- it used to
        carry an MoE aux loss that was always literally 0.0."""
        x = self.forward_hidden(idx, start_pos, kv_cache, attn_mask)
        return self.logit_proj(x), None


    def _loss_chunk(self, h, targets):
        """Projection + CE for one slice. Returns (ce_sum, z_sum, n_valid)
        packed into one tensor so it can be checkpointed."""
        logits = self.logit_proj(h).float()
        flat = logits.reshape(-1, self.vocab_size)
        tgt = targets.reshape(-1)
        ce = F.cross_entropy(flat, tgt, reduction="sum",
                             ignore_index=IGNORE_INDEX)
        if self.z_loss_weight:
            valid = tgt != IGNORE_INDEX
            z = torch.logsumexp(flat, dim=-1)
            z_sum = (z[valid] ** 2).sum()
        else:
            z_sum = ce.new_zeros(())
        n = (tgt != IGNORE_INDEX).sum().to(ce.dtype)
        return torch.stack((ce, z_sum, n))

    def calculate_loss(self, xs, ys, attn_mask=None, **_legacy):
        """Token-level cross entropy, computed in chunks over the sequence.

        The old version built a (B*T, vocab) fp32 logit tensor -- 659M floats at
        batch 32 x 1024 x 20k vocab -- and autograd kept it plus the softmax
        intermediates. That single tensor dominated activation memory. Here each
        chunk's logits are recomputed in the backward pass, so peak memory is
        one chunk instead of the whole batch. Cost is one extra logit matmul,
        roughly 4% of step FLOPs at this scale; it buys back the ~35% that
        activation checkpointing was costing.
        """
        h = self.forward_hidden(xs, attn_mask=attn_mask)

        T = h.size(1)
        chunk = self.loss_chunk_size or T
        totals = None
        for i in range(0, T, chunk):
            hs = h[:, i:i + chunk]
            ts = ys[:, i:i + chunk]
            if self.training and torch.is_grad_enabled():
                part = torch.utils.checkpoint.checkpoint(
                    self._loss_chunk, hs, ts, use_reentrant=False)
            else:
                part = self._loss_chunk(hs, ts)
            totals = part if totals is None else totals + part

        ce_sum, z_sum, n = totals[0], totals[1], totals[2].clamp(min=1.0)
        loss = ce_sum / n
        if self.z_loss_weight:
            # Penalizes drift in log Z. Cheap insurance against the logits
            # slowly inflating at a high peak LR.
            loss = loss + self.z_loss_weight * (z_sum / n)
        return loss


    def make_kv_cache(self, batch_size, max_seq_len, device=None, dtype=None):
        p = self.l_embeddings.weight
        return KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_layers=self.n_layer,
            n_kv_head=self.n_kv_head,
            head_dim=self.n_dim // self.n_head,
            device=device or p.device,
            dtype=dtype or p.dtype,
        )

    def _sample(self, logits, temperature, top_k, top_p, min_p):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-5)

        if top_k:
            k = min(top_k, logits.size(-1))
            kth = logits.topk(k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if min_p and min_p > 0:
            probs = F.softmax(logits, dim=-1)
            thresh = min_p * probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(probs < thresh, float("-inf"))

        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumulative = F.softmax(sorted_logits, dim=-1).cumsum(-1)
            remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
            remove = remove.scatter(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_count=128,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        min_p=0.0,
        eos_token_id=None,
        repetition_penalty=1.0,
        valid_vocab_size=None,
    ):
        """Autoregressive sampling with a KV cache.

        `valid_vocab_size` masks the padded tail of the vocab (pass your
        tokenizer's real size if you rounded the model vocab up).
        """
        B, T = idx.shape
        if T + max_count > self.n_seq:
            raise ValueError(
                f"Generation needs {T + max_count} positions but n_seq={self.n_seq}")

        cache = self.make_kv_cache(B, T + max_count, idx.device)

        # Prefill: project only the last position, not all T.
        h = self.forward_hidden(idx, start_pos=0, kv_cache=cache)
        logits = self.logit_proj(h[:, -1]).float()

        pos = T
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)

        for _ in range(max_count):
            if valid_vocab_size is not None and valid_vocab_size < self.vocab_size:
                logits[:, valid_vocab_size:] = float("-inf")

            if repetition_penalty != 1.0:
                gathered = torch.gather(logits, 1, idx)
                gathered = torch.where(gathered > 0,
                                       gathered / repetition_penalty,
                                       gathered * repetition_penalty)
                logits = logits.scatter(1, idx, gathered)

            next_token = self._sample(logits, temperature, top_k, top_p, min_p)

            if eos_token_id is not None:
                # once a sequence is finished, keep emitting EOS
                next_token = torch.where(done[:, None],
                                         torch.full_like(next_token, eos_token_id),
                                         next_token)
                done = done | (next_token.squeeze(1) == eos_token_id)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and bool(done.all()):
                break

            h = self.forward_hidden(next_token, start_pos=pos, kv_cache=cache)
            logits = self.logit_proj(h[:, -1]).float()
            pos += 1

        return idx

    def get_param_count(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.l_embeddings.weight.numel()
        return n

    def param_groups(self, weight_decay=0.1):
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def estimate_flops_per_token(self):
        n = self.get_param_count()
        attn = 12 * self.n_layer * self.n_dim * self.n_seq
        return 6 * n + attn