import json
import logging
import math
import os
import queue
import random
import threading
import time
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, IterableDataset, Subset,
                              get_worker_info)
from tqdm import tqdm

from tokenizer import Tokenizer
from simple_transformer import Transformer
from config import CONFIG
from pretrain_dataset import PretrainTextDataset


SEED = 1337

VOCAB_SIZE = CONFIG.get("vocab_size", 20000)
EMBEDDING_DIM = CONFIG.get("embedding_dim", 512)
NUM_LAYERS = CONFIG.get("n_layers", 16)
# head_dim must land on 64 (or 32/128): the flash-attention kernels are only
# compiled for those. 640/16 = 40 silently falls back to the math kernel and
# costs more than everything else on this list put together.
NUM_HEADS = CONFIG.get("n_heads") or max(1, EMBEDDING_DIM // 64)
SEQ_LEN = CONFIG.get("seq_len", CONFIG.get("sequence_length", 1024))

# Same 131,072 tokens/step, but in 2 fat passes instead of 4 thin ones: fewer
# kernel launches and Python-side steps per token, and the matmuls are shaped
# better. Override per-GPU with MICRO_BATCH / GRAD_ACCUM.
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH", "64"))   # sequences per forward pass
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM", "2"))
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
SNAPSHOT_EVERY = 5000          
PLOT_EVERY = 200

APPLY_RESIDUAL_INIT_SCALING = True
USE_COMPILE = os.environ.get("USE_COMPILE", "1") == "1"
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
PREFETCH = os.environ.get("PREFETCH", "1") == "1"
LOSS_CHUNK_SIZE = int(os.environ.get("LOSS_CHUNK_SIZE", "512"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PIN_MEMORY = DEVICE.type == "cuda"

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
if DEVICE.type == "cuda":
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

logger.info(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

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


# A map-style Dataset (this one) is safe with multiple workers: the DataLoader
# hands each worker a disjoint slice of indices and reassembles them in order,
# so the stream stays byte-for-byte deterministic. Only an *iterable* dataset
# has to shard itself.
if isinstance(dataset, IterableDataset) and not bool(getattr(dataset, "shards_by_worker", False)):
    _NUM_WORKERS = 0
    logger.warning(
        "    IterableDataset without `shards_by_worker`; forcing num_workers=0 so each "
        "example is not replayed once per worker. Shard its __iter__ on get_worker_info()."
    )
else:
    _NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))
logger.info(f"    dataloader workers: {_NUM_WORKERS} | background prefetch: {PREFETCH}")


def shard_by_worker(iterable):
    """Helper to paste into PretrainTextDataset.__iter__ for worker sharding."""
    info = get_worker_info()
    if info is None:
        yield from iterable
        return
    for i, item in enumerate(iterable):
        if i % info.num_workers == info.id:
            yield item


def _identity(item):
    """Keep DataLoader's default_convert from turning our int32 numpy arrays
    into torch tensors (and back again a moment later)."""
    return item


class TokenizedItems(Dataset):
    """Parquet read + BPE encode for one dataset item, run inside the worker.

    This is the whole point of the worker processes. `Tokenizer.encode` is pure
    Python -- a regex split plus a dict-cached merge loop per word -- and at
    131k tokens per optimizer step it does not keep up with an A100 on one core
    while also holding the GIL against the training loop. Doing it here puts it
    in N separate interpreters, ahead of time, and hands back int32 arrays that
    cost nothing to pack.
    """

    def __init__(self, base, tokenizer, eos_id):
        self.base = base
        self.tokenizer = tokenizer
        self.eos_id = eos_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        texts = self.base[i]
        if isinstance(texts, str):
            texts = [texts]
        parts = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            ids.append(self.eos_id)
            parts.append(np.asarray(ids, dtype=np.int32))
        tokens = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
        return len(texts), tokens


class TokenStream:

    def __init__(self, dataset, tokenizer, seq_len, batch_size,
                 shuffle_pool=SHUFFLE_POOL, skip_items=0, seed=SEED):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_len = seq_len + 1
        self.batch_size = batch_size
        self.shuffle_pool = shuffle_pool
        self.skip_items = skip_items
        self.rng = random.Random(seed)
        self.texts_consumed = 0
        self.items_consumed = skip_items

    def _token_arrays(self):
        """Yields int32 token arrays, one per dataset item, in order."""
        base = self.dataset
        if self.skip_items:
            # Resume the cheap way: never read, let alone tokenize, the items
            # this run has already trained on.
            base = Subset(base, range(self.skip_items, len(base)))
        loader = DataLoader(
            TokenizedItems(base, self.tokenizer, EOS_ID),
            batch_size=None,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=False,
            persistent_workers=bool(_NUM_WORKERS),
            prefetch_factor=6 if _NUM_WORKERS else None,
            collate_fn=_identity,
        )
        for n_texts, tokens in loader:
            self.items_consumed += 1
            self.texts_consumed += int(n_texts)
            if tokens.size:
                yield tokens

    def _sequences(self):
        """Yields chunk_len-long int32 numpy rows of packed tokens.

        The old version kept one Python list and did `buffer = buffer[chunk_len:]`
        after every chunk -- an O(len(buffer)) list copy per sequence, on the
        critical path. Here whole items arrive pre-encoded and are reshaped into
        rows at a stroke, so packing costs one memcpy per token instead of one
        per token per chunk.
        """
        chunk_len = self.chunk_len
        buf = np.empty(0, dtype=np.int32)
        for tokens in self._token_arrays():
            buf = np.concatenate((buf, tokens)) if buf.size else tokens
            n_full = buf.size // chunk_len
            if not n_full:
                continue
            block = buf[: n_full * chunk_len].reshape(n_full, chunk_len)
            for row in block:
                yield row
            buf = buf[n_full * chunk_len:].copy()

    def __iter__(self):
        pool, out = [], []
        half = self.shuffle_pool // 2
        bs = self.batch_size
        for seq in self._sequences():
            pool.append(seq)
            if len(pool) >= self.shuffle_pool:
                self.rng.shuffle(pool)
                out.extend(pool[half:])
                del pool[half:]
            # `out` is drained front-to-back; pop a window instead of rebuilding
            # the whole list on every batch.
            i = 0
            while len(out) - i >= bs:
                yield self._to_tensors(out[i:i + bs])
                i += bs
            if i:
                del out[:i]
        # drain
        self.rng.shuffle(pool)
        out.extend(pool)
        i = 0
        while len(out) - i >= bs:
            yield self._to_tensors(out[i:i + bs])
            i += bs

    @staticmethod
    def _to_tensors(seqs):
        """One numpy stack + one torch view, into pinned memory.

        `torch.tensor(list_of_lists)` walked ~66k Python ints per micro-batch on
        the critical path. Stacking numpy rows is a memcpy, and pinning lets the
        H2D copy actually be async (a pageable copy is synchronous no matter what
        non_blocking says).
        """
        block = torch.from_numpy(np.stack(seqs)).to(torch.long)
        xs = block[:, :-1].contiguous()
        ys = block[:, 1:].contiguous()
        if _PIN_MEMORY:
            xs, ys = xs.pin_memory(), ys.pin_memory()
        return xs, ys


class Prefetcher:
    """Runs the CPU-side stream (parquet -> BPE -> pack -> pin) on a background
    thread, and issues each H2D copy on a side CUDA stream from that same thread.

    Two separate overlaps, both of which the original loop gave up:

      * Tokenising is pure Python and holds the GIL, but every torch/CUDA call in
        the training loop drops it, so the producer really does run during the
        forward/backward rather than between steps.
      * The copy for batch N+k is issued as soon as its bytes exist, on a stream
        that is not the compute stream, and the compute stream only waits on the
        matching event when it actually needs that batch. The old
        `xs.to(DEVICE, non_blocking=True)` on the compute stream, from pageable
        memory, was a plain synchronous copy sitting between two steps.
    """

    def __init__(self, stream, depth=4):
        self.stream = stream
        self.depth = depth
        self.copy_stream = (torch.cuda.Stream(device=DEVICE)
                            if DEVICE.type == "cuda" else None)
        self._stop = threading.Event()
        self._thread = None

    def close(self):
        """Let the producer (and its DataLoader workers) shut down instead of
        sitting blocked on a full queue after the loop breaks at MAX_STEPS."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def _issue(self, xs, ys):
        """Start the H2D copy; return device tensors plus an event to wait on."""
        if self.copy_stream is None:
            return xs.to(DEVICE), ys.to(DEVICE), None
        with torch.cuda.stream(self.copy_stream):
            xs = xs.to(DEVICE, non_blocking=True)
            ys = ys.to(DEVICE, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return xs, ys, event

    @staticmethod
    def _consume(item):
        xs, ys, event = item
        if event is not None:
            compute = torch.cuda.current_stream()
            compute.wait_event(event)
            # Allocated on copy_stream, used on compute: tell the allocator not
            # to recycle the blocks until compute is done with them.
            xs.record_stream(compute)
            ys.record_stream(compute)
        return xs, ys

    def __iter__(self):
        if not PREFETCH:
            for xs, ys in self.stream:
                yield self._consume(self._issue(xs, ys))
            return

        q = queue.Queue(maxsize=self.depth)
        sentinel = object()
        self._stop.clear()

        def put(item):
            while not self._stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for xs, ys in self.stream:
                    if not put(self._issue(xs, ys)):
                        return
            except Exception as e:                     # surface, don't hang
                put(e)
                return
            put(sentinel)

        self._thread = threading.Thread(target=producer, daemon=True,
                                        name="tokenstream")
        self._thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield self._consume(item)


def build_validation_set():
    if os.path.exists(VAL_CACHE_FILE):
        blob = torch.load(VAL_CACHE_FILE, map_location="cpu")
        cached = blob["batches"]
        # A cache built at a different micro-batch size would make eval compile a
        # second graph and change what "40 batches" means; rebuild instead.
        if cached and tuple(cached[0][0].shape) != (MICRO_BATCH_SIZE, SEQ_LEN):
            logger.warning(
                f"    validation cache is {tuple(cached[0][0].shape)}, need "
                f"{(MICRO_BATCH_SIZE, SEQ_LEN)}; rebuilding")
        else:
            logger.info(f"    loaded {len(cached)} cached validation batches")
            return cached, int(blob.get("items_consumed", 0))

    logger.info(f"    building {VAL_BATCHES} validation batches (held out from training)...")
    stream = TokenStream(dataset, tokenizer, SEQ_LEN, MICRO_BATCH_SIZE,
                         shuffle_pool=MICRO_BATCH_SIZE * 4)
    batches = []
    for xs, ys in stream:
        batches.append((xs, ys))
        if len(batches) >= VAL_BATCHES:
            break
    offset = stream.items_consumed
    atomic_save({"batches": batches, "items_consumed": offset,
                 "texts_consumed": stream.texts_consumed}, VAL_CACHE_FILE)
    logger.info(f"    validation set uses the first {offset} dataset items "
                f"({stream.texts_consumed} texts) of the stream")
    return batches, offset


logger.info("[3] Creating transformer model...")
base_model = Transformer(
    vocab_size=MODEL_VOCAB_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
    activation_checkpointing=False,
    loss_chunk_size=LOSS_CHUNK_SIZE,
).to(DEVICE)
logger.info(f"    n_head={NUM_HEADS} -> head_dim={EMBEDDING_DIM // NUM_HEADS} "
            f"| kv heads={base_model.n_kv_head} | loss chunk={LOSS_CHUNK_SIZE}")
# channels_last is meaningless for a transformer, but making every parameter
# contiguous once avoids per-step relayout inside the fused kernels.
for _p in base_model.parameters():
    if not _p.is_contiguous():
        _p.data = _p.data.contiguous()

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
        # compile() is lazy, so a failure surfaces on the first step, long past
        # this try/except. suppress_errors turns that into a per-graph eager
        # fallback (logged) instead of killing a multi-hour run.
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 32
        # Compile calculate_loss, not the module: it is the whole training-step
        # graph (blocks + chunked logit projection + CE), so inductor gets to
        # fuse the norms, SwiGLU and the loss reduction instead of just the
        # blocks. Shapes are fixed by MICRO_BATCH_SIZE x SEQ_LEN, so one graph
        # is compiled and reused for the entire run.
        base_model.calculate_loss = torch.compile(
            base_model.calculate_loss, mode=COMPILE_MODE, dynamic=False)
        logger.info(f"    torch.compile enabled (mode={COMPILE_MODE}); "
                    f"first step pays a one-off ~1-3 min warmup")
    except Exception as e:
        logger.warning(f"    torch.compile failed, running eager: {e}")
else:
    logger.info("    eager mode (USE_COMPILE=1 for ~1.3-1.8x)")


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
    fig.savefig(os.path.join(LOG_DIR, "loss_history.png"), dpi=110)
    plt.close(fig)


@torch.no_grad()
def evaluate(val_batches):
    train_model.eval()
    total = torch.zeros((), device=DEVICE)
    for xs, ys in val_batches:
        # val_batches already live on the device -- 40 micro-batches is ~40 MB,
        # and re-uploading them every 500 steps was pure stall.
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


def save_checkpoint(path, step, stream, last_val):
    ck = {
        "global_step": step,
        # Item index, not text count: resuming can then skip straight past the
        # consumed part of the dataset instead of re-tokenizing it.
        "items_consumed": stream.items_consumed,
        "texts_consumed": stream.texts_consumed,
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
items_consumed_at_ckpt = 0
last_val_loss = float("nan")


class _FreshStart(Exception):
    """Not an error: the checkpoint is for a different architecture, so it is
    set aside and this run starts from scratch."""


if os.path.exists(CHECKPOINT_FILE):
    try:
        ck = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        old_cfg = ck.get("config") or {}
        new_cfg = {"vocab_size": MODEL_VOCAB_SIZE, "n_layer": NUM_LAYERS,
                   "n_head": NUM_HEADS, "n_dim": EMBEDDING_DIM, "n_seq": SEQ_LEN}
        mismatch = {k: (v, new_cfg[k]) for k, v in old_cfg.items()
                    if k in new_cfg and v != new_cfg[k]}

        # The config dict is only a label -- check the tensors themselves, so a
        # difference it does not record (vocab padding, kv-head count, a checkpoint
        # with no config at all) is caught too. load_state_dict copies as it goes
        # and reports shape errors at the end, which would otherwise leave a model
        # half-loaded and half-random.
        want = base_model.state_dict()
        have = ck["model_state_dict"]
        shape_diff = [k for k in want
                      if k not in have or tuple(have[k].shape) != tuple(want[k].shape)]
        if shape_diff and not mismatch:
            mismatch = {"tensors": (f"{len(shape_diff)} differ, e.g. {shape_diff[0]}",
                                    "new shapes")}

        if mismatch or shape_diff:
            # New architecture: the model built above already has it, so just
            # train it from step 0. The old checkpoint is moved aside rather than
            # deleted -- it is the only copy of that run, and the next CKPT_EVERY
            # would otherwise overwrite it a few minutes from now.
            stale = f"{CHECKPOINT_FILE}.{'-'.join(sorted(mismatch))}-{RUN_ID}.bak"
            os.replace(CHECKPOINT_FILE, stale)
            logger.warning(
                f"Checkpoint architecture differs from this config "
                f"(old -> new: {mismatch}); starting a fresh run at step 0 with "
                f"the new architecture. Previous checkpoint kept at {stale}")
            raise _FreshStart
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
        if "items_consumed" in ck:
            items_consumed_at_ckpt = int(ck["items_consumed"])
        elif ck.get("texts_consumed"):
            # Pre-item-index checkpoint. Dataset items hold DATASET_BATCH_SIZE
            # texts before chunking, so this is an estimate; it only shifts where
            # in the corpus the resumed run picks up.
            items_consumed_at_ckpt = int(ck["texts_consumed"]) // DATASET_BATCH_SIZE
            logger.warning("Old checkpoint format: data position estimated from "
                           f"texts_consumed -> item {items_consumed_at_ckpt}")
        if isinstance(ck.get("history"), dict):
            history.update({k: v for k, v in ck["history"].items() if k in history})
            history["run_id"] = RUN_ID
        last_val_loss = float(ck.get("val_loss", float("nan")))
        logger.info(f"Resumed: step={global_step}, items_consumed={items_consumed_at_ckpt}")
    except _FreshStart:
        pass                       # already logged; model is new-architecture
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {CHECKPOINT_FILE}: {e}")


val_batches, val_item_offset = build_validation_set()
val_batches = [(xs.to(DEVICE, non_blocking=True), ys.to(DEVICE, non_blocking=True))
               for xs, ys in val_batches]

logger.info("[4] Training...")
stream = TokenStream(
    dataset, tokenizer, SEQ_LEN, MICRO_BATCH_SIZE,
    skip_items=max(val_item_offset, items_consumed_at_ckpt),
    seed=SEED + global_step,
)
batches = Prefetcher(stream)
if items_consumed_at_ckpt > val_item_offset:
    logger.info(f"    resuming at dataset item {items_consumed_at_ckpt} "
                f"(skipped items are never read or tokenized)")

train_model.train()
optimizer.zero_grad(set_to_none=True)

# Cached once: rebuilding this generator inside clip_grad_norm_ every step walks
# the whole module tree in Python.
all_params = [p for p in base_model.parameters() if p.requires_grad]

training_start = time.time()
accum_loss = torch.zeros((), device=DEVICE)
accum_micro = 0
skipped_steps = 0
step_timer = time.time()

# Per-step metrics are staged on the GPU and read back in one transfer every
# LOG_EVERY steps. Reading loss/grad_norm every step drains the pipeline and
# kills CPU run-ahead -- the CPU can no longer queue step N+1's kernels while
# the GPU chews on step N, which on a fast card is most of the step time.
_pending = []          # list of (step, lr, tok_per_sec, device_tensor[loss, gn, finite])


def flush_metrics(pbar=None):
    global skipped_steps
    if not _pending:
        return
    packed = torch.stack([t for *_, t in _pending]).cpu()      # ONE sync
    last = None
    for (step, lr_v, tps, _), row in zip(_pending, packed):
        step_loss, gn, finite = float(row[0]), float(row[1]), float(row[2])
        history["steps"].append(step)
        history["train_loss"].append(step_loss)
        history["lr"].append(lr_v)
        history["grad_norm"].append(gn)
        history["tokens_per_sec"].append(tps)
        if finite < 0.5:
            skipped_steps += 1
            logger.warning(f"    non-finite grad norm at step {step}; grads zeroed "
                           f"({skipped_steps} total)")
        last = (step, step_loss, lr_v, gn, tps)
    _pending.clear()
    if pbar is not None and last is not None:
        step, step_loss, lr_v, gn, tps = last
        pbar.set_postfix({
            "loss": f"{step_loss:.4f}",
            "ppl": f"{math.exp(min(20, step_loss)):.1f}",
            "lr": f"{lr_v:.2e}",
            "gn": f"{gn:.2f}",
            "tok/s": f"{tps/1e3:.1f}k",
        })
    return last


pbar = tqdm(total=MAX_STEPS, initial=global_step, desc="pretrain", unit="step")

for xs, ys in batches:
    if global_step >= MAX_STEPS:
        break

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

    # Hand-rolled clip so the "skip non-finite steps" guard is free. The old
    # `if torch.isfinite(grad_norm):` was a blocking host sync on the critical
    # path of every step, and clip_grad_norm_ would then have made a second
    # full read/write pass over all 110M grads. Here the finite check folds
    # into the clip coefficient: non-finite -> scale 0 -> the update is a no-op,
    # and the CPU never waits to find out.
    grads = [p.grad for p in all_params if p.grad is not None]
    grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    finite = torch.isfinite(grad_norm)
    scale = torch.where(
        finite, (GRAD_CLIP_NORM / (grad_norm + 1e-6)).clamp(max=1.0),
        grad_norm.new_zeros(()))
    torch._foreach_mul_(grads, scale)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    step_metrics = torch.stack((
        accum_loss.detach().float(),
        grad_norm.detach().float(),
        finite.float(),
    ))
    accum_loss.zero_()          # torch.stack above already copied the value
    accum_micro = 0
    global_step += 1

    step_time = time.time() - step_timer
    step_timer = time.time()
    tok_per_sec = tokens_per_step / max(1e-6, step_time)
    _pending.append((global_step, lr, tok_per_sec, step_metrics))

    pbar.update(1)

    if global_step % LOG_EVERY == 0:
        last = flush_metrics(pbar)
        recent = history["train_loss"][-LOG_EVERY:]
        if last:
            logger.info(
                f"    step {global_step}/{MAX_STEPS} | loss {sum(recent)/len(recent):.4f} "
                f"| lr {lr:.2e} | gn {last[3]:.2f} | {tok_per_sec/1e3:.1f}k tok/s"
            )

    if global_step % EVAL_EVERY == 0:
        last_val_loss = evaluate(val_batches)
        history["val_steps"].append(global_step)
        history["val_loss"].append(last_val_loss)
        logger.info(f"    >> val loss {last_val_loss:.4f} | "
                    f"val ppl {math.exp(min(20, last_val_loss)):.2f}")

    if global_step % SAMPLE_EVERY == 0:
        log_sample_generation(global_step)

    # Plotting and checkpointing both read `history`, so drain the staged
    # metrics first; they are rare enough that the extra sync costs nothing.
    if global_step % PLOT_EVERY == 0:
        flush_metrics(pbar)
        save_plots()
        save_metrics()

    if global_step % CKPT_EVERY == 0:
        flush_metrics(pbar)
        save_checkpoint(CHECKPOINT_FILE, global_step, stream, last_val_loss)
        logger.info(f"    checkpoint saved at step {global_step}")

    if global_step % SNAPSHOT_EVERY == 0:
        snap = f"{ARTIFACTS_DIR}/pretrain_step{global_step}.pt"
        save_checkpoint(snap, global_step, stream, last_val_loss)
        logger.info(f"    snapshot -> {snap}")

pbar.close()
batches.close()
flush_metrics()

# Drop any partial accumulation: those grads are a fraction of a step and, with
# a scaler, were never unscaled. Not worth a wrong final update.
if accum_micro:
    optimizer.zero_grad(set_to_none=True)

last_val_loss = evaluate(val_batches)
history["val_steps"].append(global_step)
history["val_loss"].append(last_val_loss)
logger.info(f"Final val loss {last_val_loss:.4f} | ppl {math.exp(min(20, last_val_loss)):.2f}")

save_checkpoint(CHECKPOINT_FILE, global_step, stream, last_val_loss)
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