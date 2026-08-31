"""Quick sanity check for simple_transformer.Transformer.

Character-level tokenization on dataset/sample.txt, train a small model for a
few hundred steps, and watch the samples turn from noise into Shakespeare-ish
text. Runs fine on CPU in a couple of minutes; much faster on a GPU.

    python transformer_sample_test.py
"""

import math
import time

import torch

from simple_transformer import Transformer

# --- config -----------------------------------------------------------------
DATA_FILE = "dataset/sample.txt"

N_DIM = 192
N_HEAD = 6          # head_dim = 32 -> hits the fast attention kernels
N_LAYER = 6
N_SEQ = 256         # context window (chars)

BATCH_SIZE = 48
MAX_STEPS = 3000
PEAK_LR = 3e-3
MIN_LR = 3e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_EVERY = 200
EVAL_BATCHES = 20
SAMPLE_EVERY = 200
SAMPLE_LEN = 400

SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# --- char-level tokenizer --------------------------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

print(f"device: {DEVICE}")
print(f"corpus: {len(data):,} chars | vocab: {vocab_size} | "
      f"train {len(train_data):,} / val {len(val_data):,}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - N_SEQ - 1, (BATCH_SIZE,))
    xs = torch.stack([d[i:i + N_SEQ] for i in ix])
    ys = torch.stack([d[i + 1:i + 1 + N_SEQ] for i in ix])
    return xs.to(DEVICE), ys.to(DEVICE)


# --- model ---------------------------------------------------------------
model = Transformer(
    vocab_size=vocab_size,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_dim=N_DIM,
    n_seq=N_SEQ,
    z_loss_weight=0.0,
    loss_chunk_size=N_SEQ,
).to(DEVICE)

print(f"params: {model.get_param_count():,}")

optimizer = torch.optim.AdamW(
    model.param_groups(WEIGHT_DECAY),
    lr=PEAK_LR,
    betas=(0.9, 0.95),
    eps=1e-8,
)


def lr_at(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    progress = min(1.0, progress)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    total = 0.0
    for _ in range(EVAL_BATCHES):
        xs, ys = get_batch("val")
        total += model.calculate_loss(xs, ys).item()
    model.train()
    return total / EVAL_BATCHES


@torch.no_grad()
def sample(prompt="\n", n=SAMPLE_LEN):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    n = min(n, N_SEQ - idx.size(1) - 1)
    out = model.generate(idx, max_count=n, temperature=0.8, top_k=40, top_p=0.95)
    model.train()
    return decode(out[0].cpu().tolist())


# --- train -------------------------------------------------------------------
print("\ntraining...\n")
model.train()
t0 = time.time()

for step in range(1, MAX_STEPS + 1):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    xs, ys = get_batch("train")
    loss = model.calculate_loss(xs, ys)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % 50 == 0:
        dt = time.time() - t0
        print(f"step {step:>4}/{MAX_STEPS} | loss {loss.item():.4f} "
              f"| lr {lr:.2e} | {dt:.1f}s")

    if step % EVAL_EVERY == 0:
        vl = estimate_val_loss()
        print(f"  >> val loss {vl:.4f} | val ppl {math.exp(vl):.2f}")

    if step % SAMPLE_EVERY == 0:
        print("  --- sample " + "-" * 50)
        print(sample())
        print("  " + "-" * 61 + "\n")


print(f"\ndone in {time.time() - t0:.1f}s")
print("\n=== final sample ===")
print(sample(prompt="ROMEO:", n=SAMPLE_LEN))
