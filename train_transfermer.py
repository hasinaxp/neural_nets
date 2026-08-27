

import torch
import torch.nn as nn
from tokenizer import Tokenizer
from simple_transformer import Transformer
from tqdm import tqdm
import time
from datetime import timedelta




SAMPLE_FILE = "dataset/sample.txt"
TOKENIZER_FILE = "artifacts/tokenizer.txt"
TRAINED_MODEL_FILE = "artifacts/sample_model.pt"



VOCAB_SIZE = 20000  # Target; actual model vocab size is taken from the loaded tokenizer.
SEQ_LEN = 128
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
GRAD_ACCUM_STEPS = 2
GRAD_CLIP_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

print("\n[1] Setting up sub-sentence tokenizer...")
tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
# tokenizer.train_from_file(SAMPLE_FILE)
# tokenizer.save(TOKENIZER_FILE)
tokenizer.load(TOKENIZER_FILE)


actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else tokenizer.vocab_size
print("vocab size", actual_vocab_size)

MODEL_VOCAB_SIZE = actual_vocab_size

print("\n[2] Preparing training data...")
with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
    text = f.read()
    text = text * 4

tokens = tokenizer.encode(text)
print(f"   Total tokens (including boundaries): {len(tokens)}")

sequences = []
for i in range(len(tokens) // SEQ_LEN - 1):
    seq = tokens[i * SEQ_LEN : (i + 1 ) * SEQ_LEN]
    sequences.append(seq)

if not sequences:
    print(f"   WARNING: Not enough tokens to create sequences of length {SEQ_LEN}")
    exit(1)

if len(sequences) < BATCH_SIZE:
    print(f"   WARNING: Only {len(sequences)} sequences available but BATCH_SIZE={BATCH_SIZE}. "
          f"No batches would be produced — reduce BATCH_SIZE or provide more data.")
    exit(1)

print(f"   Created {len(sequences)} sequences of length {SEQ_LEN}")

# ==========================================
# 3. MODEL SETUP
# ==========================================
print("\n[3] Creating transformer model...")
base_model = Transformer(
    vocab_size=MODEL_VOCAB_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
).to(DEVICE)

torch.compile(base_model)

param_count = base_model.get_param_count()
print(f"   Model parameters: {param_count:,}")


print("   Using eager execution mode for training...")
# torch.compile is unstable on some Windows CUDA builds for this custom
# transformer and causes device-side asserts during the RoPE setup.
train_model = base_model

optimizer = torch.optim.Adam(base_model.parameters(), lr=LEARNING_RATE)

print("\n[4] Training...")
train_model.train()
training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    total_loss = 0
    num_batches = 0
    accum_loss = 0
    accum_steps = 0

    # +1 so the final full batch (when it lands exactly on a multiple of BATCH_SIZE)
    # isn't silently dropped.
    pbar = tqdm(
        range(0, len(sequences) - BATCH_SIZE + 1, BATCH_SIZE),
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        unit="batch"
    )

    for batch_idx, batch_start in enumerate(pbar):
        batch_sequences = sequences[batch_start : batch_start + BATCH_SIZE]

        xs = torch.tensor([seq[:-1] for seq in batch_sequences], dtype=torch.long).to(DEVICE)
        ys = torch.tensor([seq[1:] for seq in batch_sequences], dtype=torch.long).to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
            loss = train_model.calculate_loss(xs, ys)
            loss = loss / GRAD_ACCUM_STEPS

        loss.backward()
        accum_loss += loss.item()
        accum_steps += 1

        if accum_steps % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += accum_loss
            accum_loss = 0
            num_batches += 1

            if num_batches > 0:
                current_avg_loss = total_loss / num_batches
                current_ppl = torch.exp(torch.tensor(current_avg_loss)).item()
                pbar.set_postfix({
                    "loss": f"{current_avg_loss:.4f}",
                    "ppl": f"{current_ppl:.2f}",
                    "lr": f"{LEARNING_RATE:.2e}"
                })

    pbar.close()

    if accum_steps % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += accum_loss
        num_batches += 1

    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    time.sleep(0.7)

    print(f"   Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | "
          f"Batches: {num_batches} | Time: {timedelta(seconds=int(epoch_time))}")

torch.save(base_model.state_dict(), TRAINED_MODEL_FILE)
print(f"   Model saved to {TRAINED_MODEL_FILE}")


print("\n[5] Generating predictions...")
base_model.eval()

seed_token = torch.tensor([[tokenizer.special_tokens["<|BOS|>"]]], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    generated = base_model.generate(seed_token, max_count=100)

generated_tokens = generated[0].cpu().tolist()
generated_text = tokenizer.decode(generated_tokens)

print(f"\n   Generated text ({len(generated_tokens)} tokens):")
print(f"   >>> {generated_text}\n")