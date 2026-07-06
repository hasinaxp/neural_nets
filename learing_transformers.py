import math
import random

import torch
from tqdm import tqdm

DEFAULT_N_LAYER = 8
DEFAULT_N_SEQ_LEN = 256
DEFAULT_N_DIM = 128


class Tokenizer:
    def __init__(self, text_sample: str) -> None:
        vocab = list(sorted(set(text_sample.split())))
        self.vocab = ["<pad>"] + vocab
        self.inv_vocab = {word: i + 1 for i, word in enumerate(vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.inv_vocab.get(t, 0) for t in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join([self.vocab[i] for i in tokens])


class Attention(torch.nn.Module):
    def __init__(
        self, n_dim: int = DEFAULT_N_DIM, n_seq: int = DEFAULT_N_SEQ_LEN
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_seq = n_seq
        self.wq = torch.nn.Linear(n_dim, n_dim)
        self.wk = torch.nn.Linear(n_dim, n_dim)
        self.wv = torch.nn.Linear(n_dim, n_dim)

    def forward(self, idx):
        B, T, C = idx.shape
        Q = self.wq(idx)
        K = self.wk(idx)
        V = self.wv(idx)

        scores = Q @ K.transpose(-2, -1)

        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        scores.masked_fill_(mask, float("-inf"))

        scores /= math.sqrt(self.n_dim)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V


class FNN(torch.nn.Module):
    def __init__(
        self,
        n_dim=DEFAULT_N_DIM,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_hidden_dim = math.floor(n_dim * 1.5)
        self.fc1 = torch.nn.Linear(n_dim, self.n_hidden_dim)
        self.fc2 = torch.nn.Linear(self.n_hidden_dim, n_dim)

    def forward(self, idx):
        x = self.fc1(idx)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = DEFAULT_N_LAYER,
        n_head: int = 4,
        n_dim: int = DEFAULT_N_DIM,
        n_seq: int = DEFAULT_N_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.l_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_dim, padding_idx=0
        )
        self.positional_embeddings = torch.nn.Embedding(n_seq, n_dim)
        self.attentions = torch.nn.ModuleList(
            [Attention(n_dim=n_dim, n_seq=n_seq) for _ in range(n_layer)]
        )
        self.ffns = torch.nn.ModuleList([FNN(n_dim=n_dim) for _ in range(n_layer)])
        self.layer_norm = torch.nn.LayerNorm(n_dim)

        self.logit_proj = torch.nn.Linear(n_dim, vocab_size)

    def forward(self, idx):

        B, T = idx.shape
        x = self.l_embeddings(idx)

        positions = torch.arange(T, device=idx.device)
        positions = positions.unsqueeze(0).expand(B, T)
        x = x + self.positional_embeddings(positions)

        for i in range(self.n_layer):
            x = x + self.attentions[i](x)
            x = self.layer_norm(x)
            x = x + self.ffns[i](x)
            x = self.layer_norm(x)

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


if __name__ == "__main__":
    content = ""
    with open("sample.txt", "r") as file:
        content = file.read()

    seq_len = DEFAULT_N_SEQ_LEN
    tokenizer = Tokenizer(content)
    model = Transformer(vocab_size=tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Total parameters: {total_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    encoded_content = torch.tensor(tokenizer.encode(content))
    sample_size = (encoded_content.shape[0] // seq_len) * seq_len
    tokens = encoded_content[:sample_size]

    offset = 0
    sample = tokens[offset : offset + seq_len]
    target_sample = tokens[offset + 1 : offset + 1 + seq_len]

    model.train()

    xs = torch.stack([tokens[i : i + seq_len] for i in range(4)])
    ys = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in range(4)])

    loss = None
    steps = 2000
    pbar = tqdm(range(steps))
    for _ in pbar:
        optimizer.zero_grad()
        i = random.randint(0, steps) * seq_len
        xs = torch.stack([tokens[i : i + seq_len] for i in range(4)])
        ys = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in range(4)])

        logits = model.forward(xs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, tokenizer.vocab_size), ys.view(-1)
        )

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss}", perp=math.exp(loss))

    sample = model.generate(torch.tensor([[tokens[offset]]]))
    print(tokenizer.decode(sample[0].tolist()))
