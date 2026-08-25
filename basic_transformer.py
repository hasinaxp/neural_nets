import math
import torch
import torch.nn.functional as F

DEFAULT_SEQ_LEN = 256
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K = 2


class RMSNorm(torch.nn.Module):
    def __init__(self, n_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_dim))

    def forward(self, x):
        # Upcast to fp32 for the norm statistics; standard for stability under bf16/fp16.
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(input_dtype)


def precompute_rope(head_dim: int, seq_len: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class KVCache:
    """
    MLA compressed KV cache.
    Stores the low-rank 'c_kv' and the *shared* (not per-head) 'k_rope' instead of
    expanded K and V, cutting KV-cache memory by up to ~90% vs standard MHA.

    Cache tensors are (num_layers, batch, max_seq_len, dim). The caller (Transformer)
    tracks the write position explicitly via `start_pos` — the cache itself holds no
    mutable position state, since a single forward pass calls `update()` once per
    layer at the *same* position and a shared counter would advance once per layer
    instead of once per token.
    """
    def __init__(self, batch_size: int, max_seq_len: int, num_layers: int,
                 kv_lora_rank: int, qk_rope_head_dim: int, device, dtype):
        self.c_kv_cache = torch.zeros(num_layers, batch_size, max_seq_len, kv_lora_rank, device=device, dtype=dtype)
        self.k_rope_cache = torch.zeros(num_layers, batch_size, max_seq_len, qk_rope_head_dim, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def update(self, layer_idx: int, start_pos: int, new_c_kv: torch.Tensor, new_k_rope: torch.Tensor):
        B, T_new, _ = new_c_kv.shape
        end_pos = start_pos + T_new
        self.c_kv_cache[layer_idx, :, start_pos:end_pos] = new_c_kv
        self.k_rope_cache[layer_idx, :, start_pos:end_pos] = new_k_rope
        return self.c_kv_cache[layer_idx, :, :end_pos], self.k_rope_cache[layer_idx, :, :end_pos]


class AttentionMLA(torch.nn.Module):
    """
    DeepSeek-style Multi-Head Latent Attention (MLA):
    - Low-rank joint compression of K/V into `c_kv` (kv_lora_rank)
    - Decoupled RoPE: query rope keys are per-head, key rope is a single shared
      vector broadcast to all heads — this is what keeps the KV cache small,
      since only `kv_lora_rank + qk_rope_head_dim` needs to be cached per token,
      independent of the number of heads.
    - QK-Norm for training stability.
    """
    def __init__(self, n_dim: int, n_head: int, layer_idx: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.head_dim = n_dim // n_head
        assert self.head_dim * n_head == n_dim, "n_dim must be divisible by n_head"

        self.qk_rope_head_dim = max(8, (self.head_dim // 2) // 2 * 2)
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = self.head_dim

        self.q_lora_rank = max(16, n_dim // 4)
        self.kv_lora_rank = max(8, n_dim // 8)
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta

        # Q projections (per-head rope, as in DeepSeek-V2/V3)
        self.q_down_proj = torch.nn.Linear(n_dim, self.q_lora_rank, bias=False)
        self.q_up_proj = torch.nn.Linear(self.q_lora_rank, n_head * self.qk_nope_head_dim, bias=False)
        self.q_rope_proj = torch.nn.Linear(n_dim, n_head * self.qk_rope_head_dim, bias=False)

        # KV projections. k_rope is a SINGLE shared head (not n_head * dim) — this is
        # the core MLA trick that keeps the cache small.
        self.kv_down_proj = torch.nn.Linear(n_dim, self.kv_lora_rank, bias=False)
        self.kv_up_proj = torch.nn.Linear(self.kv_lora_rank, n_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.k_rope_proj = torch.nn.Linear(n_dim, self.qk_rope_head_dim, bias=False)

        # Output projection
        self.wo = torch.nn.Linear(n_head * self.v_head_dim, n_dim, bias=False)

        # QK-Norm for stability
        self.q_nope_norm = RMSNorm(self.qk_nope_head_dim)
        self.k_nope_norm = RMSNorm(self.qk_nope_head_dim)

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None):
        B, T_new, C = x.shape

        # 1. Query projections
        q_down = self.q_down_proj(x)
        q_nope = self.q_up_proj(q_down).view(B, T_new, self.n_head, self.qk_nope_head_dim)
        q_rope = self.q_rope_proj(x).view(B, T_new, self.n_head, self.qk_rope_head_dim)

        # 2. KV low-rank compression + shared rope key
        c_kv = self.kv_down_proj(x)
        k_rope_new = self.k_rope_proj(x).view(B, T_new, self.qk_rope_head_dim)

        # 3. KV cache handling
        if kv_cache is not None:
            c_kv_cached, k_rope_cached = kv_cache.update(self.layer_idx, start_pos, c_kv, k_rope_new)
            T_cached = c_kv_cached.shape[1]
            kv_up = self.kv_up_proj(c_kv_cached).view(B, T_cached, self.n_head, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_up.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_rope = k_rope_cached
        else:
            T_cached = T_new
            kv_up = self.kv_up_proj(c_kv).view(B, T_new, self.n_head, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_up.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_rope = k_rope_new

        # Transpose to (B, n_head, T, dim)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)
        k_rope = k_rope.unsqueeze(1)  # (B, 1, T_cached, qk_rope_head_dim) — shared across heads

        # Apply QK-Norm
        q_nope = self.q_nope_norm(q_nope)
        k_nope = self.k_nope_norm(k_nope)

        # Apply RoPE at correct positions
        cos_q = cos[start_pos: start_pos + T_new].unsqueeze(0).unsqueeze(0)
        sin_q = sin[start_pos: start_pos + T_new].unsqueeze(0).unsqueeze(0)
        cos_k = cos[:T_cached].unsqueeze(0).unsqueeze(0)
        sin_k = sin[:T_cached].unsqueeze(0).unsqueeze(0)

        q_rope = (q_rope * cos_q) + (rotate_half(q_rope) * sin_q)
        k_rope = (k_rope * cos_k) + (rotate_half(k_rope) * sin_k)
        k_rope = k_rope.expand(-1, self.n_head, -1, -1)  # broadcast the shared rope key to all heads

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Causal masking is needed whenever there's more than one new query position
        # (full-sequence forward, with or without an attached-but-empty cache). For
        # single-token decode (T_new == 1) there is nothing "in the future" to mask.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T_new > 1))

        out = out.transpose(1, 2).contiguous().view(B, T_new, self.n_head * self.v_head_dim)
        return self.wo(out), 0.0  # 0.0 aux loss for attention


class SwiGLU(torch.nn.Module):
    """Hardware-optimized SwiGLU with dimensions rounded to multiples of 256."""
    def __init__(self, n_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(n_dim, hidden_dim, bias=False)
        self.w_up = torch.nn.Linear(n_dim, hidden_dim, bias=False)
        self.w_down = torch.nn.Linear(hidden_dim, n_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(torch.nn.Module):
    """
    Mixture of Experts with:
    - Shared expert path (always active, preserves base capabilities)
    - Auxiliary load-balancing loss to prevent expert collapse
    """
    def __init__(self, n_dim: int, num_experts: int, top_k: int, num_shared_experts: int = 1) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_shared_experts = num_shared_experts

        # Hardware-friendly hidden dim (multiple of 256 for Tensor Cores)
        hidden_dim = int(n_dim * 2.5)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.gate = torch.nn.Linear(n_dim, num_experts, bias=False)
        self.experts = torch.nn.ModuleList([SwiGLU(n_dim, hidden_dim) for _ in range(num_experts)])
        self.shared_experts = SwiGLU(n_dim, hidden_dim * num_shared_experts) if num_shared_experts > 0 else None

    def forward(self, x):
        B, T, _ = x.shape
        x_flat = x.view(-1, x.size(-1))

        # Shared expert path (always executed)
        shared_out = self.shared_experts(x_flat) if self.shared_experts is not None else 0.0

        # Routed expert path
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat)

        for i in range(self.num_experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_x = x_flat[mask]
            expert_out = self.experts[i](expert_x)
            weight_mask = (topk_indices == i).float()
            expert_weights = (topk_weights * weight_mask).sum(dim=-1)[mask].unsqueeze(-1)
            final_output[mask] += expert_out * expert_weights

        final_output = final_output + shared_out

        # Auxiliary load balancing loss (encourages uniform expert utilization)
        expert_mask = torch.zeros_like(router_logits).scatter_(1, topk_indices, 1.0).mean(dim=0)
        expert_probs = routing_weights.mean(dim=0)
        aux_loss = self.num_experts * (expert_mask * expert_probs).sum()

        return final_output.view(B, T, self.n_dim), aux_loss


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = DEFAULT_NUM_LAYERS,
        n_head: int = DEFAULT_NUM_HEADS,
        n_dim: int = DEFAULT_EMBEDDING_DIM,
        n_seq: int = DEFAULT_SEQ_LEN,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        top_k: int = DEFAULT_TOP_K,
        rope_theta: float = 10000.0,
        num_shared_experts: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_seq = n_seq
        self.rope_theta = rope_theta

        self.l_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(dropout)

        self.attentions = torch.nn.ModuleList(
            [AttentionMLA(n_dim=n_dim, n_head=n_head, layer_idx=i, rope_theta=rope_theta) for i in range(n_layer)]
        )

        self.ffns = torch.nn.ModuleList(
            [MoE(n_dim=n_dim, num_experts=num_experts, top_k=top_k, num_shared_experts=num_shared_experts) for _ in range(n_layer)]
        )

        self.attn_norms = torch.nn.ModuleList([RMSNorm(n_dim) for _ in range(n_layer)])
        self.ffn_norms = torch.nn.ModuleList([RMSNorm(n_dim) for _ in range(n_layer)])
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.final_norm = RMSNorm(n_dim)

        self.logit_proj = torch.nn.Linear(n_dim, vocab_size, bias=False)
        self.logit_proj.weight = self.l_embeddings.weight  # Weight tying

        # Lazily-built RoPE table cache: (cos, sin, cached_len, device)
        self._rope_cos = None
        self._rope_sin = None
        self._rope_cached_len = 0
        self._rope_device = None

        self.apply(self._init_weights)

        # Scaled init on projections that write directly into the residual stream.
        # (q_up_proj / kv_up_proj produce Q/K, not residual writes, so they're excluded.)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, seq_len: int, device):
        if self._rope_cos is None or seq_len > self._rope_cached_len or self._rope_device != device:
            cos, sin = precompute_rope(self.attentions[0].qk_rope_head_dim, seq_len, device=device, base=self.rope_theta)
            self._rope_cos, self._rope_sin = cos, sin
            self._rope_cached_len = seq_len
            self._rope_device = device
        return self._rope_cos, self._rope_sin

    def forward(self, idx, start_pos=0, kv_cache=None):
        B, T = idx.shape
        x = self.emb_dropout(self.l_embeddings(idx))

        max_seq_len = self.n_seq if kv_cache is None else kv_cache.max_seq_len
        cos, sin = self._get_rope(max_seq_len, x.device)

        total_aux_loss = 0.0

        for i in range(self.n_layer):
            attn_out, _ = self.attentions[i](self.attn_norms[i](x), cos, sin, start_pos, kv_cache)
            x = x + self.resid_dropout(attn_out)

            ffn_out, aux_loss = self.ffns[i](self.ffn_norms[i](x))
            x = x + self.resid_dropout(ffn_out)
            total_aux_loss += aux_loss

        x = self.final_norm(x)
        logits = self.logit_proj(x)
        return logits, total_aux_loss

    @torch.no_grad()
    def generate(self, idx, max_count=128, temperature=1.0, top_p=0.9):
        B, T = idx.shape
        first_attn = self.attentions[0]

        # Initialize compressed KV cache
        kv_cache = KVCache(
            batch_size=B,
            max_seq_len=T + max_count,
            num_layers=self.n_layer,
            kv_lora_rank=first_attn.kv_lora_rank,
            qk_rope_head_dim=first_attn.qk_rope_head_dim,
            device=idx.device,
            dtype=self.l_embeddings.weight.dtype
        )

        # Prefill phase
        logits, _ = self.forward(idx, start_pos=0, kv_cache=kv_cache)
        pos = T

        for _ in range(max_count):
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)

            # Decode phase (processes only 1 token, leveraging KV cache)
            logits, _ = self.forward(next_token, start_pos=pos, kv_cache=kv_cache)
            pos += 1

        return idx

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def calculate_loss(self, xs, ys, aux_loss_weight=0.01):
        """Calculates Cross-Entropy loss + MoE Load Balancing auxiliary loss."""
        logits, aux_loss = self.forward(xs)
        ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), ys.view(-1))
        return ce_loss + (aux_loss_weight * aux_loss)