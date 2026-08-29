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
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import Tokenizer
from simple_transformer import Transformer, IGNORE_INDEX
from config import CONFIG
from sft_dataset import SFTDataset, build_sft_cache, render_conversation

SEED = 1337

VOCAB_SIZE = CONFIG.get("vocab_size", 20000)
EMBEDDING_DIM = CONFIG.get("embedding_dim", 640)
NUM_LAYERS = CONFIG.get("n_layers", 20)
NUM_HEADS = CONFIG.get("n_heads", 16)
SEQ_LEN = CONFIG.get("seq_len", 1024)

# Optimization
MICRO_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 3
PEAK_LR = 1e-5            # was 3e-5. At 3e-5 for three epochs the weights walk
                          # far enough off the pretrained solution that general
                          # language modelling degrades -- the model gets fluent
                          # at the four SFT formats and worse at everything else.
MIN_LR_RATIO = 0.1
WARMUP_FRAC = 0.05        # longer ramp: the first steps out of a converged
                          # checkpoint are the ones that do the damage.
WEIGHT_DECAY = 0.0        # SFT runs are short; decay mostly just fights the
                          # pretrained weights. Raise to 0.01 if you overfit.
BETAS = (0.9, 0.95)
GRAD_CLIP_NORM = 1.0

# Pretraining replay -- the other half of the forgetting fix.
#
# SFT's objective is narrow: loss only on assistant tokens, over four task
# formats. Nothing in it asks the model to keep modelling ordinary text, so it
# drifts. Mixing plain next-token batches from the pretraining corpus back in
# (rehearsal, in continual-learning terms) keeps the original objective pulling
# on the same weights. REPLAY_FRAC is the share of *micro-batches* drawn from
# pretraining rather than SFT.
REPLAY_FRAC = 0.25
REPLAY_SEQ_LEN = 512           # shorter than SEQ_LEN: replay is a regularizer,
                               # not a second pretraining run, and short
                               # sequences keep the step cost down
REPLAY_MICRO_BATCH_SIZE = MICRO_BATCH_SIZE
REPLAY_LOSS_WEIGHT = 1.0       # relative to the SFT loss
PRETRAIN_TEXT_BATCH = 16       # texts per pretrain-dataset __getitem__
PRETRAIN_VAL_BATCHES = 12      # held-out pretraining batches, for the
                               # forgetting curve

# Data
DATASET_BATCH_SIZE = 64        # conversations per dataset __getitem__
LENGTH_BUCKET_BATCHES = 32     # sort this many micro-batches by length together

# Intervals (optimizer steps)
LOG_EVERY = 25
EVAL_EVERY = 250
SAMPLE_EVERY = 250
CKPT_EVERY = 250
PLOT_EVERY = 100

USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = "artifacts"
LOG_DIR = "logs"
TOKENIZER_FILE = f"{ARTIFACTS_DIR}/tokenizer-{VOCAB_SIZE}.txt"
PRETRAINED_MODEL_FILE = f"{ARTIFACTS_DIR}/pretrain_checkpoint_latest.pt"
TRAINED_MODEL_FILE = f"{ARTIFACTS_DIR}/sft_model.pt"
CHECKPOINT_FILE = f"{ARTIFACTS_DIR}/sft_checkpoint_latest.pt"
PRETRAIN_VAL_CACHE = f"{ARTIFACTS_DIR}/val_batches.pt"   # written by pretraining

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"sft_{RUN_ID}.log")
METRICS_FILE = os.path.join(LOG_DIR, "sft_metrics.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("sft")
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

logger.info(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

if DEVICE.type == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    AMP_DTYPE = torch.bfloat16
USE_SCALER = DEVICE.type == "cuda" and AMP_DTYPE == torch.float16
logger.info(f"AMP dtype: {AMP_DTYPE} (grad scaler: {USE_SCALER})")


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# [1] Tokenizer
# ---------------------------------------------------------------------------

logger.info("[1] Loading tokenizer...")
tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
tokenizer.load(TOKENIZER_FILE)
actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else tokenizer.vocab_size
MODEL_VOCAB_SIZE = int(math.ceil(actual_vocab_size / 64) * 64)
logger.info(f"    tokenizer vocab {actual_vocab_size} -> model vocab {MODEL_VOCAB_SIZE}")

PAD_ID = tokenizer.special_tokens["<|PAD|>"]
EOS_ID = tokenizer.special_tokens["<|EOS|>"]
ASSISTANT_ID = tokenizer.special_tokens["<|ASSISTANT|>"]

# Sanity check on SQL fertility -- the tokenizer was trained on wiki + python,
# never on SQL, so keywords may fragment badly.
_sql = "SELECT COUNT(*) FROM employees WHERE department_id = 3 GROUP BY title;"
_fert = len(tokenizer.encode(_sql)) / len(_sql.split())
logger.info(f"    SQL fertility: {_fert:.2f} tokens/word "
            f"({'fine' if _fert < 2.5 else 'high -- SQL will be expensive'})")


# ---------------------------------------------------------------------------
# [2] Data
# ---------------------------------------------------------------------------

logger.info("[2] Preparing SFT dataset...")
build_sft_cache()

train_dataset = SFTDataset(batch_size=DATASET_BATCH_SIZE, split="train",
                           epochs=NUM_EPOCHS)
val_dataset = SFTDataset(batch_size=DATASET_BATCH_SIZE, split="val")
logger.info("    " + train_dataset.describe().replace("\n", "\n    "))
logger.info(f"    val: {len(val_dataset.index):,} examples")


def collate(examples):
    """Render conversations, right-pad, and build masked labels.

    Right-padding needs no attention mask: attention is causal, so real tokens
    never see the pads to their right, and the pad positions themselves are
    excluded from the loss via IGNORE_INDEX.
    """
    rendered = []
    for ex in examples:
        ids, mask = render_conversation(tokenizer, ex["messages"], SEQ_LEN)
        if ids is None or sum(mask) == 0:
            continue          # dropped: too long, or nothing to learn from
        rendered.append((ids, mask, ex["task"]))

    if not rendered:
        return None

    width = max(len(ids) for ids, _, _ in rendered)
    xs, ys, tasks = [], [], []
    for ids, mask, task in rendered:
        pad = width - len(ids)
        full_ids = ids + [PAD_ID] * pad
        full_mask = mask + [0] * pad
        # next-token targets, masked to assistant tokens only
        x = full_ids[:-1]
        y = [full_ids[i + 1] if full_mask[i + 1] else IGNORE_INDEX
             for i in range(len(full_ids) - 1)]
        xs.append(x)
        ys.append(y)
        tasks.append(task)

    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            tasks)


def iter_micro_batches(dataset, micro_batch_size, shuffle=True, bucket=True):
    """Yield padded micro-batches, grouping similar lengths together.

    Examples range from ~30 tokens (small talk) to ~1000 (a passage plus a
    question). Batching them randomly means most of every batch is padding.
    """
    loader = DataLoader(dataset, batch_size=None, shuffle=shuffle,
                        num_workers=0, collate_fn=None)

    pool = []
    pool_target = micro_batch_size * LENGTH_BUCKET_BATCHES if bucket else micro_batch_size

    def flush(pool):
        if bucket:
            pool.sort(key=lambda ex: sum(len(m["content"]) for m in ex["messages"]))
        batches = [pool[i:i + micro_batch_size]
                   for i in range(0, len(pool), micro_batch_size)]
        if shuffle:
            random.shuffle(batches)
        for b in batches:
            out = collate(b)
            if out is not None:
                yield out

    for example_batch in loader:
        pool.extend(example_batch)
        if len(pool) >= pool_target:
            yield from flush(pool)
            pool = []
    if pool:
        yield from flush(pool)


# ---------------------------------------------------------------------------
# [2b] Pretraining replay
# ---------------------------------------------------------------------------

class PretrainReplay:
    """Endless next-token batches packed from the pretraining corpus.

    Shards are walked sequentially and wrapped around at the end: the loader
    caches one parquet file at a time, so random access would re-read a shard
    per batch. Batches carry no IGNORE_INDEX masking -- every token is a
    target, exactly as during pretraining.
    """

    def __init__(self, dataset, tokenizer, start_batch=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cursor = int(start_batch)

    def _texts(self):
        n = max(1, len(self.dataset))
        empty_streak = 0
        while True:
            try:
                texts = self.dataset[self.cursor % n]
            except Exception as e:
                logger.warning(f"    replay: skipped corpus batch {self.cursor} ({e})")
                texts = []
            self.cursor += 1
            usable = [t for t in texts if isinstance(t, str) and t]
            # A whole pass over the corpus with nothing usable means there is no
            # corpus. Stop instead of spinning forever.
            empty_streak = 0 if usable else empty_streak + 1
            if empty_streak > n:
                logger.warning("    replay: corpus yielded no usable text; stopping")
                return
            yield from usable

    def batches(self, seq_len, batch_size):
        chunk_len = seq_len + 1
        buf, seqs = [], []
        for text in self._texts():
            buf.extend(self.tokenizer.encode(text))
            buf.append(EOS_ID)
            while len(buf) >= chunk_len:
                seqs.append(buf[:chunk_len])
                del buf[:chunk_len]
                if len(seqs) == batch_size:
                    block = torch.tensor(seqs, dtype=torch.long)
                    seqs = []
                    yield block[:, :-1].contiguous(), block[:, 1:].contiguous()


def build_replay():
    """(replay_batch_iterator, held_out_pretrain_batches).

    Returns (None, []) when replay is switched off or the pretraining corpus
    isn't on this machine -- SFT still runs, it just loses the rehearsal.
    """
    if REPLAY_FRAC <= 0:
        logger.info("    replay disabled (REPLAY_FRAC = 0)")
        return None, []

    try:
        # Imported lazily: the module scans dataset/raw at import time, and a
        # box with only the SFT cache on it should still be able to fine-tune.
        from pretrain_dataset import PretrainTextDataset
    except Exception as e:
        logger.warning(f"    pretraining corpus unavailable ({e}); replay disabled")
        return None, []

    try:
        corpus = PretrainTextDataset(
            batch_size=PRETRAIN_TEXT_BATCH,
            min_chunk_size=1024,
            max_chunk_size=2 * 1024,
            include_wikipedia=False,   # parquet shards only; skips the gz load
        )
    except Exception as e:
        logger.warning(f"    could not open pretraining corpus ({e}); replay disabled")
        return None, []

    val_batches = []
    start_batch = 0

    if os.path.exists(PRETRAIN_VAL_CACHE):
        # Reuse pretraining's own holdout, so the number here is comparable to
        # the val loss the pretraining run reported.
        try:
            blob = torch.load(PRETRAIN_VAL_CACHE, map_location="cpu")
            for xs, ys in blob["batches"][:PRETRAIN_VAL_BATCHES]:
                val_batches.append((xs[:, :REPLAY_SEQ_LEN].contiguous(),
                                    ys[:, :REPLAY_SEQ_LEN].contiguous()))
            # texts_consumed counts chunks, not corpus batches; divide and add
            # a margin so replay starts clear of the holdout.
            start_batch = int(blob.get("texts_consumed", 0)) // PRETRAIN_TEXT_BATCH + 8
            logger.info(f"    reusing {len(val_batches)} held-out batches from "
                        f"{PRETRAIN_VAL_CACHE}")
        except Exception as e:
            logger.warning(f"    could not read {PRETRAIN_VAL_CACHE}: {e}")
            val_batches = []

    if not val_batches:
        holdout = PretrainReplay(corpus, tokenizer)
        it = holdout.batches(REPLAY_SEQ_LEN, REPLAY_MICRO_BATCH_SIZE)
        try:
            for _ in range(PRETRAIN_VAL_BATCHES):
                val_batches.append(next(it))
        except StopIteration:
            pass
        start_batch = holdout.cursor + 1
        logger.info(f"    built {len(val_batches)} held-out pretraining batches "
                    f"from the head of the corpus")

    if not val_batches:
        logger.warning("    pretraining corpus yielded nothing; replay disabled")
        return None, []

    replay = PretrainReplay(corpus, tokenizer, start_batch=start_batch)
    logger.info(f"    replay: {REPLAY_FRAC:.0%} of micro-batches, "
                f"{REPLAY_MICRO_BATCH_SIZE}x{REPLAY_SEQ_LEN} tokens, "
                f"weight {REPLAY_LOSS_WEIGHT}, corpus cursor starts at {start_batch}")
    return replay.batches(REPLAY_SEQ_LEN, REPLAY_MICRO_BATCH_SIZE), val_batches


logger.info("[2b] Preparing pretraining replay...")
replay_iter, PRETRAIN_VAL = build_replay()


def mixed_micro_batches(dataset):
    """Interleave SFT micro-batches with replay batches.

    Yields (xs, ys, tasks, is_replay). An epoch still ends when the SFT data
    runs out -- replay is infinite and only fills in the chosen share.
    """
    rng = random.Random(SEED + 7)
    sft = iter_micro_batches(dataset, MICRO_BATCH_SIZE)
    while True:
        if replay_iter is not None and rng.random() < REPLAY_FRAC:
            try:
                rxs, rys = next(replay_iter)
            except StopIteration:
                pass
            else:
                yield rxs, rys, ["pretrain"] * rxs.size(0), True
                continue
        try:
            xs, ys, tasks = next(sft)
        except StopIteration:
            return
        yield xs, ys, tasks, False


# ---------------------------------------------------------------------------
# [3] Model
# ---------------------------------------------------------------------------

logger.info("[3] Loading pretrained model...")
model = Transformer(
    vocab_size=MODEL_VOCAB_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
).to(DEVICE)

if not os.path.exists(PRETRAINED_MODEL_FILE):
    raise FileNotFoundError(
        f"{PRETRAINED_MODEL_FILE} not found -- run pretraining first")

state = torch.load(PRETRAINED_MODEL_FILE, map_location=DEVICE)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    logger.warning(f"    missing keys: {missing}")
if unexpected:
    logger.warning(f"    unexpected keys: {unexpected}")
logger.info(f"    loaded {PRETRAINED_MODEL_FILE} "
            f"({model.get_param_count():,} params)")

train_model = model
if USE_COMPILE:
    try:
        train_model = torch.compile(model)
        logger.info("    torch.compile enabled")
    except Exception as e:
        logger.warning(f"    torch.compile failed, running eager: {e}")

# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------

_fused_ok = DEVICE.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
optimizer = torch.optim.AdamW(
    model.param_groups(weight_decay=WEIGHT_DECAY),
    lr=PEAK_LR, betas=BETAS, eps=1e-8,
    **({"fused": True} if _fused_ok else {}),
)
scaler = torch.amp.GradScaler(enabled=USE_SCALER) if USE_SCALER else None

# Steps are estimated: collate drops over-length examples, so the true count is
# slightly lower. The cosine schedule is clamped, so overshooting is harmless.
est_micro = len(train_dataset) * DATASET_BATCH_SIZE / MICRO_BATCH_SIZE
if replay_iter is not None:
    # SFT batches are only (1 - REPLAY_FRAC) of what the loop consumes.
    est_micro /= max(1e-6, 1.0 - REPLAY_FRAC)
MAX_STEPS = max(1, int(est_micro / GRAD_ACCUM_STEPS))
WARMUP_STEPS = max(1, int(WARMUP_FRAC * MAX_STEPS))
MIN_LR = PEAK_LR * MIN_LR_RATIO
logger.info(f"    ~{MAX_STEPS:,} optimizer steps over {NUM_EPOCHS} epochs "
            f"| warmup {WARMUP_STEPS} | LR {PEAK_LR:.2e} -> {MIN_LR:.2e}")


def lr_at(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    p = min(1.0, max(0.0, p))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Metrics / eval
# ---------------------------------------------------------------------------

history = {
    "run_id": RUN_ID,
    "steps": [], "train_loss": [], "replay_loss": [], "lr": [], "grad_norm": [],
    "tokens_per_sec": [],
    "val_steps": [], "val_loss": [], "val_by_task": [], "val_pretrain": [],
}


def save_metrics():
    tmp = METRICS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, METRICS_FILE)


def save_plots():
    if not history["steps"]:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

    ax1.plot(history["steps"], history["train_loss"], lw=1, label="train (sft)")
    if any(v == v for v in history["replay_loss"]):
        ax1.plot(history["steps"], history["replay_loss"], lw=1, alpha=0.6,
                 color="tab:green", label="train (replay)")
    if history["val_steps"]:
        ax1.plot(history["val_steps"], history["val_loss"], marker="o", ms=3,
                 color="tab:red", label="val (sft)")
    if any(v == v for v in history["val_pretrain"]):
        ax1.plot(history["val_steps"], history["val_pretrain"], marker="s", ms=3,
                 ls="--", color="tab:purple", label="val (pretrain)")
        if base_pretrain_val == base_pretrain_val:
            # The forgetting line: pretraining val loss should stay near it.
            ax1.axhline(base_pretrain_val, color="tab:purple", lw=0.8, ls=":",
                        alpha=0.7)
    ax1.set_ylabel("loss")
    ax1.set_title("SFT")
    ax1.legend()

    if history["val_by_task"]:
        tasks = sorted(history["val_by_task"][0].keys())
        for task in tasks:
            ys = [d.get(task) for d in history["val_by_task"]]
            ax2.plot(history["val_steps"], ys, marker="o", ms=3, label=task)
        ax2.set_ylabel("val loss by task")
        ax2.legend(fontsize=7)

    ax3.plot(history["steps"], history["grad_norm"], color="tab:orange", lw=1)
    ax3.set_ylabel("grad norm")
    ax3.set_yscale("log")
    ax3.set_xlabel("optimizer step")

    fig.tight_layout()
    fig.savefig(os.path.join(LOG_DIR, "sft_curves.png"), dpi=110)
    plt.close(fig)


VAL_BATCHES = None
base_pretrain_val = float("nan")


def build_val_batches():
    batches = []
    for xs, ys, tasks in iter_micro_batches(val_dataset, MICRO_BATCH_SIZE,
                                            shuffle=False, bucket=True):
        batches.append((xs, ys, tasks))
    logger.info(f"    {len(batches)} validation micro-batches")
    return batches


@torch.no_grad()
def evaluate():
    """Per-task loss. Token-weighted, so tasks with longer replies don't get
    silently downweighted."""
    train_model.eval()
    sums, counts = {}, {}
    for xs, ys, tasks in VAL_BATCHES:
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            logits, _ = train_model(xs)
        per_token = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, MODEL_VOCAB_SIZE), ys.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="none",
        ).view(ys.shape)
        valid = (ys != IGNORE_INDEX)
        row_sum = (per_token * valid).sum(dim=1).tolist()
        row_n = valid.sum(dim=1).tolist()
        for task, s, n in zip(tasks, row_sum, row_n):
            sums[task] = sums.get(task, 0.0) + s
            counts[task] = counts.get(task, 0) + n
    train_model.train()

    by_task = {t: sums[t] / max(1, counts[t]) for t in sums}
    overall = sum(sums.values()) / max(1, sum(counts.values()))
    return overall, by_task


@torch.no_grad()
def evaluate_pretrain():
    """Plain next-token loss on held-out pretraining text.

    This is the forgetting metric: SFT val loss can fall the whole run while
    this one climbs, which is exactly the failure mode being fixed here.
    """
    if not PRETRAIN_VAL:
        return float("nan")
    train_model.eval()
    total = torch.zeros((), device=DEVICE)
    for xs, ys in PRETRAIN_VAL:
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            total += train_model.calculate_loss(xs, ys).detach()
    train_model.train()
    return (total / len(PRETRAIN_VAL)).item()


SAMPLE_PROMPTS = [
    ("chat", "Hi! How are you doing today?"),
    ("summarization",
     "Summarize the conversation below.\n\n"
     "Amy: are we still on for dinner tomorrow?\n"
     "Ben: yes! 7pm at the usual place\n"
     "Amy: perfect, see you then"),
    ("extractive_qa",
     "Answer the question using only the passage below.\n\n"
     "Passage:\nThe Nile is a major river in northeastern Africa. It is about "
     "6,650 kilometres long and flows north into the Mediterranean Sea.\n\n"
     "Question: How long is the Nile?"),
    ("sql",
     "Write a SQL query that answers the question, using the schema provided.\n\n"
     "Schema:\nCREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)\n\n"
     "Question: What is the average salary in the engineering department?"),
    ("math",
     "Solve the problem below. Show your reasoning, then give the final answer.\n\n"
     "Natalia sold clips to 48 of her friends in April, and then she sold half "
     "as many clips in May. How many clips did Natalia sell altogether in April "
     "and May?"),
    ("reasoning",
     "Choose the option that best answers the question.\n\n"
     "What do people use to write on paper?\n\n"
     "A) pen\nB) car\nC) river\nD) cloud"),
]


@torch.inference_mode()
def log_samples(step):
    model.eval()
    for task, prompt in SAMPLE_PROMPTS:
        ids, _ = render_conversation(
            tokenizer, [{"role": "user", "content": prompt}], SEQ_LEN)
        # render_conversation closes the assistant turn; for generation we want
        # the prompt to end at the assistant marker instead.
        ids = ids[:ids.index(ASSISTANT_ID) + 1] if ASSISTANT_ID in ids else ids
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        out = model.generate(x, max_count=100, temperature=0.7, top_k=50,
                             top_p=0.9, eos_token_id=EOS_ID,
                             valid_vocab_size=actual_vocab_size)
        reply = tokenizer.decode(out[0, len(ids):].cpu().tolist())
        logger.info(f"    [{task}] {reply.replace(chr(10), ' ')[:220]}")
    model.train()


def save_checkpoint(path, step, epoch, last_val):
    ck = {
        "global_step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "val_loss": last_val,
        "config": {
            "vocab_size": MODEL_VOCAB_SIZE, "n_layer": NUM_LAYERS,
            "n_head": NUM_HEADS, "n_dim": EMBEDDING_DIM, "n_seq": SEQ_LEN,
        },
    }
    if scaler is not None:
        ck["scaler_state_dict"] = scaler.state_dict()
    atomic_save(ck, path)


# ---------------------------------------------------------------------------
# [4] Resume
# ---------------------------------------------------------------------------

global_step = 0
start_epoch = 0
last_val_loss = float("nan")

if os.path.exists(CHECKPOINT_FILE):
    try:
        ck = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        try:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        except Exception:
            logger.warning("Optimizer state did not load; Adam moments restart")
        if scaler is not None and "scaler_state_dict" in ck:
            scaler.load_state_dict(ck["scaler_state_dict"])
        global_step = int(ck.get("global_step", 0))
        start_epoch = int(ck.get("epoch", 0))
        if isinstance(ck.get("history"), dict):
            history.update({k: v for k, v in ck["history"].items() if k in history})
            history["run_id"] = RUN_ID
            # Checkpoints from before replay existed have no replay/pretrain
            # series; pad them with NaN so the plots stay aligned.
            for key, ref in (("replay_loss", "steps"), ("val_pretrain", "val_steps")):
                missing = len(history[ref]) - len(history[key])
                if missing > 0:
                    history[key] = [float("nan")] * missing + history[key]
        last_val_loss = float(ck.get("val_loss", float("nan")))
        logger.info(f"Resumed: epoch={start_epoch}, step={global_step}")
    except Exception as e:
        logger.warning(f"Failed to load {CHECKPOINT_FILE}: {e}")

# ---------------------------------------------------------------------------
# [4b] Train
# ---------------------------------------------------------------------------

logger.info("[4] Training...")
VAL_BATCHES = build_val_batches()

base_val, base_by_task = evaluate()
base_pretrain_val = evaluate_pretrain()
logger.info(f"    pretrained baseline val loss {base_val:.4f}")
for task, v in sorted(base_by_task.items()):
    logger.info(f"      {task:15s} {v:.4f}")
logger.info(f"    pretraining baseline val loss {base_pretrain_val:.4f} "
            f"(watch this one for forgetting)")

train_model.train()
optimizer.zero_grad(set_to_none=True)

training_start = time.time()
accum_loss = torch.zeros((), device=DEVICE)
accum_replay_loss = torch.zeros((), device=DEVICE)
accum_micro = 0
accum_sft_micro = 0
accum_replay_micro = 0
accum_tokens = 0
replay_micro_total = 0
skipped_steps = 0
step_timer = time.time()
stop = False

for epoch in range(start_epoch, NUM_EPOCHS):
    if stop:
        break
    logger.info(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    pbar = tqdm(desc=f"epoch {epoch + 1}", unit="step")

    for xs, ys, tasks, is_replay in mixed_micro_batches(train_dataset):
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            loss = train_model.calculate_loss(xs, ys)
            if is_replay:
                loss = loss * REPLAY_LOSS_WEIGHT
            loss = loss / GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # The two objectives sit at different scales, so they are tracked
        # apart -- one blended number would just wobble with the mix.
        if is_replay:
            accum_replay_loss += loss.detach()
            accum_replay_micro += 1
            replay_micro_total += 1
        else:
            accum_loss += loss.detach()
            accum_sft_micro += 1
        accum_micro += 1
        accum_tokens += int((ys != IGNORE_INDEX).sum())

        if accum_micro < GRAD_ACCUM_STEPS:
            continue

        lr = lr_at(global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

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
            logger.warning(f"    non-finite grad norm at step {global_step}; skipped")

        optimizer.zero_grad(set_to_none=True)

        # one host sync for both objectives; undo the /GRAD_ACCUM_STEPS so the
        # numbers are per-micro-batch means regardless of the step's mix
        sums = torch.stack([accum_loss, accum_replay_loss]).tolist()
        if accum_sft_micro:
            step_loss = sums[0] * GRAD_ACCUM_STEPS / accum_sft_micro
        else:
            step_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
        step_replay_loss = float("nan")
        if accum_replay_micro:
            step_replay_loss = (sums[1] * GRAD_ACCUM_STEPS
                                / (accum_replay_micro * max(1e-6, REPLAY_LOSS_WEIGHT)))
        accum_loss.zero_()
        accum_replay_loss.zero_()
        accum_micro = 0
        accum_sft_micro = 0
        accum_replay_micro = 0
        global_step += 1

        step_time = time.time() - step_timer
        step_timer = time.time()
        # Supervised tokens per second -- not total tokens; padding and prompt
        # tokens do work but carry no gradient signal.
        tok_per_sec = accum_tokens / max(1e-6, step_time)
        accum_tokens = 0

        history["steps"].append(global_step)
        history["train_loss"].append(step_loss)
        history["replay_loss"].append(step_replay_loss)
        history["lr"].append(lr)
        history["grad_norm"].append(float(grad_norm))
        history["tokens_per_sec"].append(tok_per_sec)

        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{step_loss:.4f}",
            "replay": f"{step_replay_loss:.3f}",
            "ppl": f"{math.exp(min(20, step_loss)):.1f}",
            "lr": f"{lr:.2e}",
            "gn": f"{float(grad_norm):.2f}",
        })

        if global_step % LOG_EVERY == 0:
            recent = [v for v in history["train_loss"][-LOG_EVERY:] if v == v]
            rec_replay = [v for v in history["replay_loss"][-LOG_EVERY:] if v == v]
            replay_txt = (f" | replay {sum(rec_replay)/len(rec_replay):.4f}"
                          if rec_replay else "")
            logger.info(f"    step {global_step}/{MAX_STEPS} | "
                        f"loss {sum(recent)/max(1, len(recent)):.4f}{replay_txt} | "
                        f"lr {lr:.2e} | gn {float(grad_norm):.2f} | "
                        f"{tok_per_sec:.0f} sup-tok/s")

        if global_step % EVAL_EVERY == 0:
            last_val_loss, by_task = evaluate()
            pre_val = evaluate_pretrain()
            history["val_steps"].append(global_step)
            history["val_loss"].append(last_val_loss)
            history["val_by_task"].append(by_task)
            history["val_pretrain"].append(pre_val)
            deltas = " ".join(
                f"{t}={by_task[t]:.3f}({by_task[t] - base_by_task.get(t, by_task[t]):+.3f})"
                for t in sorted(by_task))
            logger.info(f"    >> val {last_val_loss:.4f} | {deltas}")
            if pre_val == pre_val:
                drift = pre_val - base_pretrain_val
                warn = "  <-- forgetting" if drift > 0.15 else ""
                logger.info(f"    >> pretrain val {pre_val:.4f} "
                            f"({drift:+.4f} vs baseline){warn}")

        if global_step % SAMPLE_EVERY == 0:
            log_samples(global_step)

        if global_step % PLOT_EVERY == 0:
            save_plots()
            save_metrics()

        if global_step % CKPT_EVERY == 0:
            save_checkpoint(CHECKPOINT_FILE, global_step, epoch, last_val_loss)
            logger.info(f"    checkpoint saved at step {global_step}")

    pbar.close()

    if accum_micro:
        optimizer.zero_grad(set_to_none=True)
        accum_loss.zero_()
        accum_replay_loss.zero_()
        accum_micro = 0
        accum_sft_micro = 0
        accum_replay_micro = 0

    val, by_task = evaluate()
    pre_val = evaluate_pretrain()
    logger.info(f"    epoch {epoch + 1} val loss {val:.4f} | pretrain val "
                f"{pre_val:.4f} ({pre_val - base_pretrain_val:+.4f} vs baseline)")
    snap = f"{ARTIFACTS_DIR}/sft_model_epoch{epoch + 1}.pt"
    atomic_save(model.state_dict(), snap)
    logger.info(f"    epoch snapshot -> {snap}")
    save_checkpoint(CHECKPOINT_FILE, global_step, epoch + 1, val)

last_val_loss, by_task = evaluate()
final_pretrain_val = evaluate_pretrain()
history["val_steps"].append(global_step)
history["val_loss"].append(last_val_loss)
history["val_by_task"].append(by_task)
history["val_pretrain"].append(final_pretrain_val)

logger.info(f"Final val loss {last_val_loss:.4f} (baseline {base_val:.4f})")
for task in sorted(by_task):
    delta = by_task[task] - base_by_task.get(task, by_task[task])
    logger.info(f"    {task:15s} {by_task[task]:.4f} ({delta:+.4f} vs pretrained)")
logger.info(f"    {'pretrain':15s} {final_pretrain_val:.4f} "
            f"({final_pretrain_val - base_pretrain_val:+.4f} vs pretrained) "
            f"| {replay_micro_total:,} replay micro-batches mixed in")

atomic_save(model.state_dict(), TRAINED_MODEL_FILE)
save_checkpoint(CHECKPOINT_FILE, global_step, NUM_EPOCHS, last_val_loss)
save_plots()
save_metrics()

logger.info(f"Total time: {timedelta(seconds=int(time.time() - training_start))} "
            f"| steps {global_step} | skipped {skipped_steps}")
logger.info(f"Model saved to {TRAINED_MODEL_FILE}")

logger.info("[5] Final samples...")
log_samples(global_step)