import math
import torch
import torch.nn.functional as F


DEFAULT_SEQ_LEN = 2048
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 12
DEFAULT_NUM_EXPERTS = 1
DEFAULT_TOP_K = 1


class RMSNorm(torch.nn.Module):
    def __init__(self, n_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim, seq_len, device, base=10000.0):
    if head_dim % 2:
        raise ValueError("head_dim must be even")

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )

    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class KVCache:
    def __init__(
        self,
        batch_size,
        max_seq_len,
        num_layers,
        kv_lora_rank,
        qk_rope_head_dim,
        device,
        dtype,
        n_kv_head=1,
        head_dim=None,
    ):
        head_dim = head_dim or qk_rope_head_dim

        self.max_seq_len = max_seq_len

        self.k = torch.empty(
            num_layers,
            batch_size,
            n_kv_head,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        self.v = torch.empty_like(self.k)

    def update(self, layer_idx, start_pos, k, v):
        t = k.size(2)
        end = start_pos + t

        if end > self.max_seq_len:
            raise RuntimeError("KV cache overflow")

        self.k[layer_idx, :, :, start_pos:end] = k
        self.v[layer_idx, :, :, start_pos:end] = v

        return (
            self.k[layer_idx, :, :, :end],
            self.v[layer_idx, :, :, :end],
        )


class AttentionGQA(torch.nn.Module):
    """
    API-compatible name.
    Internally: GQA + QK-Norm + RoPE.
    """

    def __init__(
        self,
        n_dim,
        n_head,
        layer_idx,
        rope_theta=10000.0,
        n_kv_head=None,
    ):
        super().__init__()

        if n_dim % n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")

        self.n_dim = n_dim
        self.n_head = n_head
        self.head_dim = n_dim // n_head
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta

        n_kv_head = n_kv_head or max(1, n_head // 4)

        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.n_kv_head = n_kv_head

        # Compatibility fields.
        self.qk_rope_head_dim = self.head_dim
        self.kv_lora_rank = self.head_dim

        self.q_proj = torch.nn.Linear(
            n_dim, n_dim, bias=False
        )

        self.k_proj = torch.nn.Linear(
            n_dim, n_kv_head * self.head_dim, bias=False
        )

        self.v_proj = torch.nn.Linear(
            n_dim, n_kv_head * self.head_dim, bias=False
        )

        self.wo = torch.nn.Linear(
            n_dim, n_dim, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        c = cos[start_pos:start_pos + T][None, None]
        s = sin[start_pos:start_pos + T][None, None]

        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s

        if kv_cache is not None:
            k, v = kv_cache.update(
                self.layer_idx,
                start_pos,
                k,
                v,
            )

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(kv_cache is None and T > 1),
        )

        out = out.transpose(1, 2).contiguous().view(
            B, T, self.n_dim
        )

        return self.wo(out), 0.0


class SwiGLU(torch.nn.Module):
    def __init__(self, n_dim, hidden_dim):
        super().__init__()

        self.g = torch.nn.Linear(
            n_dim, hidden_dim, bias=False
        )

        self.u = torch.nn.Linear(
            n_dim, hidden_dim, bias=False
        )

        self.d = torch.nn.Linear(
            hidden_dim, n_dim, bias=False
        )

    def forward(self, x):
        return self.d(
            F.silu(self.g(x)) * self.u(x)
        )


class FFN(torch.nn.Module):
    """
    API-compatible dense FFN.
    """

    def __init__(
        self,
        n_dim,
        num_experts,
        top_k,
        num_shared_experts=1,
        hidden_dim=None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(8 * n_dim / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.ffn = SwiGLU(
            n_dim,
            hidden_dim,
        )

    def forward(self, x):
        return self.ffn(x), 0.0


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer=DEFAULT_NUM_LAYERS,
        n_head=DEFAULT_NUM_HEADS,
        n_dim=DEFAULT_EMBEDDING_DIM,
        n_seq=DEFAULT_SEQ_LEN,
        num_experts=DEFAULT_NUM_EXPERTS,
        top_k=DEFAULT_TOP_K,
        rope_theta=10000.0,
        num_shared_experts=1,
        dropout=0.0,
    ):
        super().__init__()

        if n_dim % n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_seq = n_seq
        self.rope_theta = rope_theta
        # Enable activation checkpointing to save memory during training
        self.activation_checkpointing = True

        # GQA ratio.
        if n_head <= 4:
            n_kv_head = 1
        elif n_head <= 8:
            n_kv_head = 2
        elif n_head <= 16:
            n_kv_head = 4
        else:
            n_kv_head = 8

        while n_head % n_kv_head:
            n_kv_head -= 1

        self.n_kv_head = n_kv_head

        self.l_embeddings = torch.nn.Embedding(
            vocab_size,
            n_dim,
            padding_idx=0,
        )

        self.emb_dropout = torch.nn.Dropout(dropout)

        self.attentions = torch.nn.ModuleList([
            AttentionGQA(
                n_dim,
                n_head,
                i,
                rope_theta,
                n_kv_head,
            )
            for i in range(n_layer)
        ])

        hidden_dim = int(8 * n_dim / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.ffns = torch.nn.ModuleList([
            FFN(
                n_dim,
                num_experts,
                top_k,
                num_shared_experts,
                hidden_dim,
            )
            for _ in range(n_layer)
        ])

        self.attn_norms = torch.nn.ModuleList([
            RMSNorm(n_dim)
            for _ in range(n_layer)
        ])

        self.ffn_norms = torch.nn.ModuleList([
            RMSNorm(n_dim)
            for _ in range(n_layer)
        ])

        self.resid_dropout = torch.nn.Dropout(dropout)
        self.final_norm = RMSNorm(n_dim)

        self.logit_proj = torch.nn.Linear(
            n_dim,
            vocab_size,
            bias=False,
        )

        # Weight tying.
        self.logit_proj.weight = self.l_embeddings.weight

        self._rope_cos = None
        self._rope_sin = None
        self._rope_cached_len = 0
        self._rope_device = None

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "d.weight")):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layer),
                )

    def _init_weights(self, module):
        if isinstance(
            module,
            (torch.nn.Linear, torch.nn.Embedding),
        ):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _get_rope(self, seq_len, device):
        if (
            self._rope_cos is None
            or seq_len > self._rope_cached_len
            or device != self._rope_device
        ):
            self._rope_cos, self._rope_sin = precompute_rope(
                self.attentions[0].head_dim,
                seq_len,
                device,
                self.rope_theta,
            )

            self._rope_cached_len = seq_len
            self._rope_device = device

        return self._rope_cos, self._rope_sin

    def forward(self, idx, start_pos=0, kv_cache=None):
        # Gives a useful error instead of a mysterious CUDA assert.
        if idx.numel():
            lo = idx.min().item()
            hi = idx.max().item()

            if lo < 0 or hi >= self.vocab_size:
                raise ValueError(
                    f"Token id range [{lo}, {hi}] outside "
                    f"[0, {self.vocab_size - 1}]"
                )

        B, T = idx.shape

        if kv_cache is None and T > self.n_seq:
            raise ValueError(
                f"Sequence length {T} exceeds n_seq={self.n_seq}"
            )

        x = self.emb_dropout(
            self.l_embeddings(idx)
        )

        max_len = (
            self.n_seq
            if kv_cache is None
            else kv_cache.max_seq_len
        )

        cos, sin = self._get_rope(
            max_len,
            x.device,
        )

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        x_ = x

        for i in range(self.n_layer):

            def layer_fn(inp):
                # Attention block
                residual = inp
                if i % 4:
                    residual = (residual + inp) / 2.0

                y, _ = self.attentions[i](
                    self.attn_norms[i](inp),
                    cos,
                    sin,
                    start_pos,
                    kv_cache,
                )

                out1 = residual + self.resid_dropout(y)

                # FFN block
                y2, a2 = self.ffns[i](
                    self.ffn_norms[i](out1)
                )

                out2 = out1 + self.resid_dropout(y2)

                return out2, torch.tensor(a2, device=out2.device, dtype=out2.dtype)

            if self.activation_checkpointing and self.training:
                out = torch.utils.checkpoint.checkpoint(layer_fn, x, use_reentrant=False)
                x, a = out[0], out[1]
            else:
                x, a = layer_fn(x)

            aux_loss = aux_loss + a

        x = self.final_norm(x)

        return self.logit_proj(x), aux_loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_count=128,
        temperature=1.0,
        top_p=0.9,
    ):
        B, T = idx.shape

        if T + max_count > self.n_seq:
            raise ValueError(
                f"Generation requires {T + max_count} tokens, "
                f"but n_seq={self.n_seq}"
            )

        attn = self.attentions[0]

        cache = KVCache(
            batch_size=B,
            max_seq_len=T + max_count,
            num_layers=self.n_layer,
            kv_lora_rank=attn.kv_lora_rank,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            device=idx.device,
            dtype=self.l_embeddings.weight.dtype,
            n_kv_head=self.n_kv_head,
            head_dim=attn.head_dim,
        )

        logits, _ = self.forward(
            idx,
            start_pos=0,
            kv_cache=cache,
        )

        pos = T

        for _ in range(max_count):

            logits = logits[:, -1]
            logits = logits / max(
                temperature,
                1e-5,
            )

            if top_p < 1.0:
                probs = F.softmax(logits, dim=-1)

                sorted_probs, sorted_idx = torch.sort(
                    probs,
                    descending=True,
                )

                cumulative = sorted_probs.cumsum(-1)

                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False

                remove = remove.scatter(
                    1,
                    sorted_idx,
                    remove,
                )

                logits = logits.masked_fill(
                    remove,
                    float("-inf"),
                )

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )

            logits, _ = self.forward(
                next_token,
                start_pos=pos,
                kv_cache=cache,
            )

            pos += 1

        return idx

    def get_param_count(self):
        return sum(
            p.numel()
            for p in self.parameters()
        )

    def calculate_loss(
        self,
        xs,
        ys,
        aux_loss_weight=0.01,
    ):
        logits, aux_loss = self.forward(xs)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            ys.reshape(-1),
        )

        return loss + aux_loss_weight * aux_loss