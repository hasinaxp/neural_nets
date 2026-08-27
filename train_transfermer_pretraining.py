import json
import logging
import os
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import Tokenizer
from simple_transformer import Transformer
from config import CONFIG
from pretrain_dataset import PretrainTextDataset


VOCAB_SIZE = CONFIG.get('vocab_size', 20000)
SEQ_LEN = 2024
EMBEDDING_DIM = CONFIG.get('embedding_dim', 512)
NUM_HEADS = 16
NUM_LAYERS = 24
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
GRAD_ACCUM_STEPS = 2
GRAD_CLIP_NORM = 1.0
DATASET_BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = "artifacts"
LOG_DIR = "logs"
TOKENIZER_FILE = f"{ARTIFACTS_DIR}/tokenizer-{VOCAB_SIZE}.txt"
TRAINED_MODEL_FILE = f"{ARTIFACTS_DIR}/pretrain_model.pt"
CHECKPOINT_FILE = f"{ARTIFACTS_DIR}/pretrain_checkpoint_latest.pt"

os.makedirs("artifacts", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"train_{RUN_ID}.log")
METRICS_FILE = os.path.join(LOG_DIR, "metrics.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("pretrain")

logger.info(f"Using device: {DEVICE}")

logger.info("[1] Setting up sub-sentence tokenizer...")
tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
tokenizer.load(TOKENIZER_FILE)

actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else tokenizer.vocab_size
logger.info(f"vocab size {actual_vocab_size}")

MODEL_VOCAB_SIZE = actual_vocab_size

logger.info("[2] Preparing pretraining dataset...")
dataset = PretrainTextDataset(batch_size=DATASET_BATCH_SIZE, min_chunk_size=1024, max_chunk_size=3 * 1024)
logger.info(f"   Dataset has ~{len(dataset)} text batches")


def iter_training_batches(dataset, tokenizer, seq_len, batch_size, shuffle=True):
    """Streams tokenized (xs, ys) training batches by packing tokenized text
    chunks from `dataset` into a running buffer and slicing off fixed-length
    sequences once enough tokens have accumulated."""
    loader = DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=0)
    chunk_len = seq_len + 1
    buffer = []

    for text_batch in loader:
        for text in text_batch:
            buffer.extend(tokenizer.encode(text))

        while len(buffer) >= chunk_len * batch_size:
            block, buffer = buffer[:chunk_len * batch_size], buffer[chunk_len * batch_size:]
            seqs = [block[i * chunk_len:(i + 1) * chunk_len] for i in range(batch_size)]
            xs = torch.tensor([s[:-1] for s in seqs], dtype=torch.long)
            ys = torch.tensor([s[1:] for s in seqs], dtype=torch.long)
            yield xs, ys


logger.info("[3] Creating transformer model...")
base_model = Transformer(
    vocab_size=MODEL_VOCAB_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
).to(DEVICE)

param_count = base_model.get_param_count()
logger.info(f"   Model parameters: {param_count:,}")

logger.info("   Using eager execution mode for training...")
# torch.compile is unstable on some Windows CUDA builds for this custom
# transformer and causes device-side asserts during the RoPE setup.
train_model = base_model

optimizer = torch.optim.Adam(base_model.parameters(), lr=LEARNING_RATE)

step_losses, step_ppls = [], []
epoch_avg_losses, epoch_avg_ppls = [], []


def save_plots():
    fig, ax = plt.subplots()
    ax.plot(step_losses)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    fig.savefig(os.path.join(LOG_DIR, "loss_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(step_ppls)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("perplexity")
    ax.set_yscale("log")
    ax.set_title("Training perplexity")
    fig.savefig(os.path.join(LOG_DIR, "perplexity_curve.png"))
    plt.close(fig)

    if epoch_avg_losses:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(epoch_avg_losses) + 1), epoch_avg_losses, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("avg loss")
        ax.set_title("Epoch avg loss")
        fig.savefig(os.path.join(LOG_DIR, "epoch_loss_curve.png"))
        plt.close(fig)


def save_metrics():
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "run_id": RUN_ID,
            "step_losses": step_losses,
            "step_ppls": step_ppls,
            "epoch_avg_losses": epoch_avg_losses,
            "epoch_avg_ppls": epoch_avg_ppls,
        }, f)


logger.info("[4] Training...")
train_model.train()
training_start_time = time.time()
global_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    accum_loss = 0.0
    accum_steps = 0

    batch_iter = iter_training_batches(dataset, tokenizer, SEQ_LEN, BATCH_SIZE, shuffle=True)
    pbar = tqdm(batch_iter, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

    for xs, ys in pbar:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)

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
            num_batches += 1
            global_step += 1

            step_loss = accum_loss
            step_ppl = torch.exp(torch.tensor(step_loss)).item()
            step_losses.append(step_loss)
            step_ppls.append(step_ppl)
            accum_loss = 0.0

            current_avg_loss = total_loss / num_batches
            current_ppl = torch.exp(torch.tensor(current_avg_loss)).item()
            pbar.set_postfix({
                "loss": f"{current_avg_loss:.4f}",
                "ppl": f"{current_ppl:.2f}",
                "lr": f"{LEARNING_RATE:.2e}",
            })

            if global_step % 50 == 0:
                logger.info(f"   step {global_step} | loss {step_loss:.4f} | ppl {step_ppl:.2f}")
                save_plots()
                save_metrics()

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
    epoch_avg_losses.append(avg_loss)
    epoch_avg_ppls.append(perplexity)

    logger.info(f"   Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | "
                f"Batches: {num_batches} | Time: {timedelta(seconds=int(epoch_time))}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_loss": avg_loss,
    }, CHECKPOINT_FILE)
    save_plots()
    save_metrics()

total_training_time = time.time() - training_start_time
logger.info(f"   Total training time: {timedelta(seconds=int(total_training_time))}")

torch.save(base_model.state_dict(), TRAINED_MODEL_FILE)
logger.info(f"   Model saved to {TRAINED_MODEL_FILE}")
save_plots()
save_metrics()

logger.info("[5] Generating predictions...")
base_model.eval()

seed_token = torch.tensor([[tokenizer.special_tokens["<|BOS|>"]]], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    generated = base_model.generate(seed_token, max_count=100)

generated_tokens = generated[0].cpu().tolist()
generated_text = tokenizer.decode(generated_tokens)

logger.info(f"   Generated text ({len(generated_tokens)} tokens):")
logger.info(f"   >>> {generated_text}")