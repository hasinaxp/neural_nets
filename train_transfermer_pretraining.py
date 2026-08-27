import json
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from tokenizer import Tokenizer
from simple_transformer import Transformer
from config import CONFIG
from pretrain_dataset import PretrainTextDataset


SEED = 1337

VOCAB_SIZE = CONFIG.get("vocab_size", 20000)
EMBEDDING_DIM = CONFIG.get("embedding_dim", 512)
NUM_LAYERS = CONFIG.get("n_layers", 16)
NUM_HEADS = CONFIG.get("n_heads", 16)
SEQ_LEN = CONFIG.get("seq_len", 1024)

MICRO_BATCH_SIZE = 32          # sequences per forward pass
GRAD_ACCUM_STEPS = 4           # -> 32 * 1024 * 4 = 131,072 tokens per step
MAX_STEPS = CONFIG.get("max_steps", 17000)   # ~2.2B tokens ~= 20 tok/param here
PEAK_LR = 5e-4                 # 131k tokens/step supports a slightly higher peak
MIN_LR_RATIO = 0.1             # final LR = PEAK_LR * this
WARMUP_FRAC = 0.02
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP_NORM = 1.0

# Data
DATASET_BATCH_SIZE = 16
SHUFFLE_POOL = 4096            # sequences held in the shuffle buffer
VAL_BATCHES = 40               # micro-batches held out from the head of the stream

LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 1000            
CKPT_EVERY = 500
SNAPSHOT_EVERY = 5000          # keep a permanent copy this often
PLOT_EVERY = 200

APPLY_RESIDUAL_INIT_SCALING = True
USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = "artifacts"
LOG_DIR = "logs"
TOKENIZER_FILE = f"{ARTIFACTS_DIR}/tokenizer-{VOCAB_SIZE}.txt"
TRAINED_MODEL_FILE = f"{ARTIFACTS_DIR}/pretrain_model.pt"
CHECKPOINT_FILE = f"{ARTIFACTS_DIR}/pretrain_checkpoint_latest.pt"
VAL_CACHE_FILE = f"{ARTIFACTS_DIR}/val_batches.pt"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
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
for _lg in ("datasets", "pyarrow", "huggingface_hub", "urllib3", "fsspec",
            "tokenizers", "transformers", "matplotlib"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# TF32 matmuls: large speedup, negligible quality impact for LM pretraining.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

logger.info(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# bf16 needs no loss scaling and cannot silently overflow -> prefer it.
if DEVICE.type == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    AMP_DTYPE = torch.bfloat16
USE_SCALER = DEVICE.type == "cuda" and AMP_DTYPE == torch.float16
logger.info(f"AMP dtype: {AMP_DTYPE} (grad scaler: {USE_SCALER})")


logger.info("[1] Setting up sub-sentence tokenizer...")
tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
tokenizer.load(TOKENIZER_FILE)
actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else tokenizer.vocab_size

MODEL_VOCAB_SIZE = int(math.ceil(actual_vocab_size / 64) * 64)
logger.info(f"    tokenizer vocab {actual_vocab_size} -> model vocab {MODEL_VOCAB_SIZE} (padded)")

EOS_ID = tokenizer.special_tokens.get("<|EOS|>", tokenizer.special_tokens.get("<|BOS|>"))
if EOS_ID is None:
    raise RuntimeError("Tokenizer needs an <|EOS|> or <|BOS|> special token for document packing")



logger.info("[2] Preparing pretraining dataset...")
dataset = PretrainTextDataset(
    batch_size=DATASET_BATCH_SIZE,
    min_chunk_size=1024,
    max_chunk_size=2 * 1024,
)
logger.info(f"    Dataset has ~{len(dataset)} text batches")


_DATASET_SHARDS = bool(getattr(dataset, "shards_by_worker", False))
_NUM_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2)) if _DATASET_SHARDS else 0
if not _DATASET_SHARDS:
    logger.warning(
        "    PretrainTextDataset does not declare `shards_by_worker`; using num_workers=0 "
        "to avoid feeding each example N times. Add worker sharding in its __iter__ "
        "(see get_worker_info) and set self.shards_by_worker = True to re-enable workers."
    )


def shard_by_worker(iterable):
    """Helper to paste into PretrainTextDataset.__iter__ for worker sharding."""
    info = get_worker_info()
    if info is None:
        yield from iterable
        return
    for i, item in enumerate(iterable):
        if i % info.num_workers == info.id:
            yield item


class TokenStream:

    def __init__(self, dataset, tokenizer, seq_len, batch_size,
                 shuffle_pool=SHUFFLE_POOL, skip_texts=0, seed=SEED):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_len = seq_len + 1
        self.batch_size = batch_size
        self.shuffle_pool = shuffle_pool
        self.skip_texts = skip_texts
        self.rng = random.Random(seed)
        self.texts_consumed = 0

    def _texts(self):
        loader = DataLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=False,
            persistent_workers=bool(_NUM_WORKERS),
            prefetch_factor=4 if _NUM_WORKERS else None,
        )
        for text_batch in loader:
            texts = text_batch if isinstance(text_batch, (list, tuple)) else [text_batch]
            for text in texts:
                self.texts_consumed += 1
                # Fast-forward on resume: skip without tokenizing or touching the GPU.
                if self.texts_consumed <= self.skip_texts:
                    continue
                yield text

    def _sequences(self):
        buffer = []
        for text in self._texts():
            buffer.extend(self.tokenizer.encode(text))
            buffer.append(EOS_ID)
            while len(buffer) >= self.chunk_len:
                yield buffer[: self.chunk_len]
                buffer = buffer[self.chunk_len:]

    def __iter__(self):
        pool, out = [], []
        for seq in self._sequences():
            pool.append(seq)
            if len(pool) >= self.shuffle_pool:
                self.rng.shuffle(pool)
                out.extend(pool[self.shuffle_pool // 2:])
                del pool[self.shuffle_pool // 2:]
            while len(out) >= self.batch_size:
                batch, out = out[: self.batch_size], out[self.batch_size:]
                yield self._to_tensors(batch)
        # drain
        self.rng.shuffle(pool)
        out.extend(pool)
        while len(out) >= self.batch_size:
            batch, out = out[: self.batch_size], out[self.batch_size:]
            yield self._to_tensors(batch)

    @staticmethod
    def _to_tensors(seqs):
        block = torch.tensor(seqs, dtype=torch.long)
        return block[:, :-1].contiguous(), block[:, 1:].contiguous()


def build_validation_set():
    if os.path.exists(VAL_CACHE_FILE):
        blob = torch.load(VAL_CACHE_FILE, map_location="cpu")
        logger.info(f"    loaded {len(blob['batches'])} cached validation batches")
        return blob["batches"], int(blob["texts_consumed"])

    logger.info(f"    building {VAL_BATCHES} validation batches (held out from training)...")
    stream = TokenStream(dataset, tokenizer, SEQ_LEN, MICRO_BATCH_SIZE,
                         shuffle_pool=MICRO_BATCH_SIZE * 4)
    batches = []
    for xs, ys in stream:
        batches.append((xs, ys))
        if len(batches) >= VAL_BATCHES:
            break
    offset = stream.texts_consumed
    atomic_save({"batches": batches, "texts_consumed": offset}, VAL_CACHE_FILE)
    logger.info(f"    validation set uses the first {offset} texts of the stream")
    return batches, offset


logger.info("[3] Creating transformer model...")
base_model = Transformer(
    vocab_size=MODEL_VOCAB_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
).to(DEVICE)

param_count = base_model.get_param_count()
tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
logger.info(f"    Model parameters: {param_count:,}")
logger.info(f"    Tokens per optimizer step: {tokens_per_step:,}")
logger.info(f"    Planned budget: {MAX_STEPS * tokens_per_step / 1e9:.2f}B tokens "
            f"({MAX_STEPS * tokens_per_step / max(1, param_count):.1f} tok/param; "
            f"Chinchilla-optimal is ~20)")


def apply_residual_init_scaling(model, n_layer):
    """GPT-2 style: shrink each block's output projection by 1/sqrt(2*n_layer).

    Keeps residual-stream variance from growing with depth. Matters a lot for
    deep models at a real learning rate. Safe to run at init only.
    """
    scale = 1.0 / math.sqrt(2 * n_layer)
    targets = ("proj.weight", "o_proj.weight", "out_proj.weight",
               "fc2.weight", "w2.weight", "down_proj.weight", "wo.weight")
    touched = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and name.endswith(targets):
                p.mul_(scale)
                touched.append(name)
    if touched:
        logger.info(f"    residual init scaling applied to {len(touched)} tensors "
                    f"(e.g. {touched[0]})")
    else:
        logger.warning("    residual init scaling matched nothing -- check the parameter "
                       "names in simple_transformer.py and update `targets`")


if APPLY_RESIDUAL_INIT_SCALING:
    apply_residual_init_scaling(base_model, NUM_LAYERS)


train_model = base_model
if USE_COMPILE:
    try:
        train_model = torch.compile(base_model)
        logger.info("    torch.compile enabled")
    except Exception as e:
        logger.warning(f"    torch.compile failed, running eager: {e}")
else:
    logger.info("    eager mode (set USE_COMPILE=1 on Linux for ~1.3-1.8x)")


decay_params, nodecay_params = [], []
for name, p in base_model.named_parameters():
    if not p.requires_grad:
        continue
    (decay_params if p.dim() >= 2 else nodecay_params).append(p)
logger.info(f"    decay tensors: {len(decay_params)} | no-decay tensors: {len(nodecay_params)}")

_fused_ok = DEVICE.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=PEAK_LR,
    betas=BETAS,
    eps=1e-8,
    **({"fused": True} if _fused_ok else {}),
)

scaler = torch.amp.GradScaler(enabled=USE_SCALER) if USE_SCALER else None

WARMUP_STEPS = max(1, int(WARMUP_FRAC * MAX_STEPS))
MIN_LR = PEAK_LR * MIN_LR_RATIO


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


logger.info(f"    LR: warmup {WARMUP_STEPS} steps to {PEAK_LR:.2e}, cosine to {MIN_LR:.2e}")


history = {
    "run_id": RUN_ID,
    "steps": [],
    "train_loss": [],
    "lr": [],
    "grad_norm": [],
    "tokens_per_sec": [],
    "val_steps": [],
    "val_loss": [],
}


def atomic_save(obj, path):
    """Write to a temp file then rename, so a crash mid-write cannot corrupt
    the only checkpoint you have."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_metrics():
    tmp = METRICS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, METRICS_FILE)


def save_plots():
    if not history["steps"]:
        return
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 11))
    ax_loss, ax_lr, ax_gn, ax_tp = axes

    ax_loss.plot(history["steps"], history["train_loss"], label="train", lw=1)
    if history["val_steps"]:
        ax_loss.plot(history["val_steps"], history["val_loss"],
                     marker="o", ms=3, label="val", color="tab:red")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Pretraining")
    ax_loss.legend()

    ax_lr.plot(history["steps"], history["lr"], color="tab:green")
    ax_lr.set_ylabel("lr")

    ax_gn.plot(history["steps"], history["grad_norm"], color="tab:orange", lw=1)
    ax_gn.set_ylabel("grad norm")
    ax_gn.set_yscale("log")

    ax_tp.plot(history["steps"], history["tokens_per_sec"], color="tab:purple", lw=1)
    ax_tp.set_ylabel("tokens/sec")
    ax_tp.set_xlabel("optimizer step")

    fig.tight_layout()
    fig.savefig(os.path.join(LOG_DIR, "training_curves.png"), dpi=110)
    plt.close(fig)


@torch.no_grad()
def evaluate(val_batches):
    train_model.eval()
    total = torch.zeros((), device=DEVICE)
    for xs, ys in val_batches:
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            total += train_model.calculate_loss(xs, ys).detach()
    train_model.train()
    return (total / max(1, len(val_batches))).item()


@torch.inference_mode()
def log_sample_generation(step):
    base_model.eval()
    seed = torch.tensor([[tokenizer.special_tokens["<|BOS|>"]]],
                        dtype=torch.long, device=DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                        enabled=DEVICE.type == "cuda"):
        generated = base_model.generate(seed, max_count=80)
    text = tokenizer.decode(generated[0].cpu().tolist())
    logger.info(f"    sample @ step {step}: {text}")
    base_model.train()


def save_checkpoint(path, step, texts_consumed, last_val):
    ck = {
        "global_step": step,
        "texts_consumed": texts_consumed,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "val_loss": last_val,
        "config": {
            "vocab_size": MODEL_VOCAB_SIZE,
            "n_layer": NUM_LAYERS,
            "n_head": NUM_HEADS,
            "n_dim": EMBEDDING_DIM,
            "n_seq": SEQ_LEN,
        },
    }
    if scaler is not None:
        ck["scaler_state_dict"] = scaler.state_dict()
    atomic_save(ck, path)


global_step = 0
texts_consumed_at_ckpt = 0
last_val_loss = float("nan")

if os.path.exists(CHECKPOINT_FILE):
    try:
        ck = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        base_model.load_state_dict(ck["model_state_dict"])
        try:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        except Exception:
            logger.warning("Optimizer state did not load; Adam moments restart from zero")
        if scaler is not None and "scaler_state_dict" in ck:
            try:
                scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception:
                logger.warning("Scaler state did not load; continuing")
        global_step = int(ck.get("global_step", 0))
        texts_consumed_at_ckpt = int(ck.get("texts_consumed", 0))
        if isinstance(ck.get("history"), dict):
            history.update({k: v for k, v in ck["history"].items() if k in history})
            history["run_id"] = RUN_ID
        last_val_loss = float(ck.get("val_loss", float("nan")))
        logger.info(f"Resumed: step={global_step}, texts_consumed={texts_consumed_at_ckpt}")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {CHECKPOINT_FILE}: {e}")


val_batches, val_text_offset = build_validation_set()

logger.info("[4] Training...")
stream = TokenStream(
    dataset, tokenizer, SEQ_LEN, MICRO_BATCH_SIZE,
    skip_texts=max(val_text_offset, texts_consumed_at_ckpt),
    seed=SEED + global_step,
)
if texts_consumed_at_ckpt > val_text_offset:
    logger.info(f"    fast-forwarding data stream past {texts_consumed_at_ckpt} texts "
                f"(no forward/backward on skipped data)")

train_model.train()
optimizer.zero_grad(set_to_none=True)

training_start = time.time()
accum_loss = torch.zeros((), device=DEVICE)
accum_micro = 0
skipped_steps = 0
step_timer = time.time()

pbar = tqdm(total=MAX_STEPS, initial=global_step, desc="pretrain", unit="step")

for xs, ys in stream:
    if global_step >= MAX_STEPS:
        break

    xs = xs.to(DEVICE, non_blocking=True)
    ys = ys.to(DEVICE, non_blocking=True)

    with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                        enabled=DEVICE.type == "cuda"):
        loss = train_model.calculate_loss(xs, ys) / GRAD_ACCUM_STEPS

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Stay on-device: .item() here would force a host sync every micro-batch.
    accum_loss += loss.detach()
    accum_micro += 1

    if accum_micro < GRAD_ACCUM_STEPS:
        continue

    # ---- optimizer step ----
    lr = lr_at(global_step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP_NORM)

    if torch.isfinite(grad_norm):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    else:
        skipped_steps += 1
        if scaler is not None:
            scaler.update()
        logger.warning(f"    non-finite grad norm at step {global_step}; step skipped "
                       f"({skipped_steps} total)")

    optimizer.zero_grad(set_to_none=True)

    step_loss = accum_loss.item()          # one sync per optimizer step
    accum_loss.zero_()
    accum_micro = 0
    global_step += 1

    step_time = time.time() - step_timer
    step_timer = time.time()
    tok_per_sec = tokens_per_step / max(1e-6, step_time)

    history["steps"].append(global_step)
    history["train_loss"].append(step_loss)
    history["lr"].append(lr)
    history["grad_norm"].append(float(grad_norm))
    history["tokens_per_sec"].append(tok_per_sec)

    pbar.update(1)
    pbar.set_postfix({
        "loss": f"{step_loss:.4f}",
        "ppl": f"{math.exp(min(20, step_loss)):.1f}",
        "lr": f"{lr:.2e}",
        "gn": f"{float(grad_norm):.2f}",
        "tok/s": f"{tok_per_sec/1e3:.1f}k",
    })

    if global_step % LOG_EVERY == 0:
        recent = history["train_loss"][-LOG_EVERY:]
        logger.info(
            f"    step {global_step}/{MAX_STEPS} | loss {sum(recent)/len(recent):.4f} "
            f"| lr {lr:.2e} | gn {float(grad_norm):.2f} | {tok_per_sec/1e3:.1f}k tok/s"
        )

    if global_step % EVAL_EVERY == 0:
        last_val_loss = evaluate(val_batches)
        history["val_steps"].append(global_step)
        history["val_loss"].append(last_val_loss)
        logger.info(f"    >> val loss {last_val_loss:.4f} | "
                    f"val ppl {math.exp(min(20, last_val_loss)):.2f}")

    if global_step % SAMPLE_EVERY == 0:
        log_sample_generation(global_step)

    if global_step % PLOT_EVERY == 0:
        save_plots()
        save_metrics()

    if global_step % CKPT_EVERY == 0:
        save_checkpoint(CHECKPOINT_FILE, global_step, stream.texts_consumed, last_val_loss)
        logger.info(f"    checkpoint saved at step {global_step}")

    if global_step % SNAPSHOT_EVERY == 0:
        snap = f"{ARTIFACTS_DIR}/pretrain_step{global_step}.pt"
        save_checkpoint(snap, global_step, stream.texts_consumed, last_val_loss)
        logger.info(f"    snapshot -> {snap}")

pbar.close()

# Drop any partial accumulation: those grads are a fraction of a step and, with
# a scaler, were never unscaled. Not worth a wrong final update.
if accum_micro:
    optimizer.zero_grad(set_to_none=True)

last_val_loss = evaluate(val_batches)
history["val_steps"].append(global_step)
history["val_loss"].append(last_val_loss)
logger.info(f"Final val loss {last_val_loss:.4f} | ppl {math.exp(min(20, last_val_loss)):.2f}")

save_checkpoint(CHECKPOINT_FILE, global_step, stream.texts_consumed, last_val_loss)
atomic_save(base_model.state_dict(), TRAINED_MODEL_FILE)
save_plots()
save_metrics()

total_time = time.time() - training_start
logger.info(f"Total training time: {timedelta(seconds=int(total_time))} "
            f"| steps {global_step} | skipped {skipped_steps} "
            f"| tokens {global_step * tokens_per_step:,}")
logger.info(f"Model saved to {TRAINED_MODEL_FILE}")


logger.info("[5] Generating predictions...")
base_model.eval()
seed_token = torch.tensor([[tokenizer.special_tokens["<|BOS|>"]]],
                          dtype=torch.long, device=DEVICE)
with torch.no_grad():
    generated = base_model.generate(seed_token, max_count=100)
generated_tokens = generated[0].cpu().tolist()
logger.info(f"    Generated text ({len(generated_tokens)} tokens):")
logger.info(f"    >>> {tokenizer.decode(generated_tokens)}")