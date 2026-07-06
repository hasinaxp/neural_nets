import math
import random

import torch
import torch.nn.functional as F

DEFAULT_SEQ_LEN = 256
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_KV_HEADS = 8  # set < DEFAULT_NUM_HEADS to enable GQA


class RMSNorm(torch.nn.Module):
    def __init__(self, n_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def precompute_rope(head_dim: int, seq_len: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()  # each (T, head_dim)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: (1, 1, T, head_dim) broadcast against (B, n_head, T, head_dim)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(torch.nn.Module):
    def __init__(
        self,
        n_dim: int = DEFAULT_EMBEDDING_DIM,
        n_head: int = DEFAULT_NUM_HEADS,
        n_kv_head: int = DEFAULT_NUM_KV_HEADS,
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_dim // n_head
        assert self.head_dim * n_head == n_dim, "n_dim must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.wq = torch.nn.Linear(n_dim, n_head * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(n_head * self.head_dim, n_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(out)


class FNN(torch.nn.Module):
    """SwiGLU feed-forward, hidden dim chosen to match param count of a
    standard 2-matrix GELU MLP at the same n_dim (~1.5x expansion)."""

    def __init__(self, n_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        super().__init__()
        self.n_dim = n_dim
        hidden = max(1, int(2 * (n_dim * 1.5) / 3))
        self.w_gate = torch.nn.Linear(n_dim, hidden, bias=False)
        self.w_up = torch.nn.Linear(n_dim, hidden, bias=False)
        self.w_down = torch.nn.Linear(hidden, n_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = DEFAULT_NUM_LAYERS,
        n_head: int = DEFAULT_NUM_HEADS,
        n_kv_head: int = DEFAULT_NUM_KV_HEADS,
        n_dim: int = DEFAULT_EMBEDDING_DIM,
        n_seq: int = DEFAULT_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_seq = n_seq
        self.head_dim = n_dim // n_head

        self.l_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_dim, padding_idx=0
        )
        self.attentions = torch.nn.ModuleList(
            [
                Attention(n_dim=n_dim, n_head=n_head, n_kv_head=n_kv_head)
                for _ in range(n_layer)
            ]
        )
        self.ffns = torch.nn.ModuleList([FNN(n_dim=n_dim) for _ in range(n_layer)])
        self.attn_norms = torch.nn.ModuleList([RMSNorm(n_dim) for _ in range(n_layer)])
        self.ffn_norms = torch.nn.ModuleList([RMSNorm(n_dim) for _ in range(n_layer)])
        self.final_norm = RMSNorm(n_dim)

        self.logit_proj = torch.nn.Linear(n_dim, vocab_size, bias=False)
        self.logit_proj.weight = self.l_embeddings.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init on residual-stream output projections, GPT-2 style
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.l_embeddings(idx)

        cos, sin = precompute_rope(self.head_dim, T, device=idx.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        for i in range(self.n_layer):
            x = x + self.attentions[i](self.attn_norms[i](x), cos, sin)
            x = x + self.ffns[i](self.ffn_norms[i](x))

        x = self.final_norm(x)
        logits = self.logit_proj(x)
        return logits

    def generate(self, idx, max_count=128):
        for i in range(max_count):
            logits = self.forward(idx)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def calculate_loss(self, xs, ys):
        logits = self.forward(xs)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), ys.view(-1)
        )
