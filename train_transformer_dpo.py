"""Direct Preference Optimization on top of the SFT checkpoint.

DPO optimizes the policy directly against pairwise preferences, with a frozen
reference model holding it near its starting point:

    r(x, y)  = beta * (log pi(y|x) - log pi_ref(y|x))         "implicit reward"
    loss     = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) )

Three things about this objective bite in practice, and each has a knob here:

  * It only constrains the *difference* of the two log-probabilities. The usual
    failure is both falling -- the model gets better at ranking and worse at
    writing. NLL_WEIGHT adds a plain SFT term on the chosen reply to anchor it,
    and `logp/chosen` in the logs is the number to watch.
  * The reward is a sum over completion tokens, so it scales with length. DPO
    reliably drifts toward longer replies; LENGTH_NORMALIZE switches to the mean
    instead, trading some fidelity to the paper for a shorter-reply prior.
  * The reference forward is half the compute of a step and its result never
    changes. CACHE_REF_LOGPS precomputes it once per epoch instead.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import Tokenizer
from simple_transformer import Transformer, IGNORE_INDEX
from config import CONFIG
from dpo_dataset import DPODataset, build_dpo_cache, render_pair

SEED = 1337

VOCAB_SIZE = CONFIG.get("vocab_size", 20000)
EMBEDDING_DIM = CONFIG.get("embedding_dim", 640)
NUM_LAYERS = CONFIG.get("n_layers", 20)
NUM_HEADS = CONFIG.get("n_heads") or max(1, EMBEDDING_DIM // 64)
SEQ_LEN = CONFIG.get("seq_len", CONFIG.get("sequence_length", 1024))

# ---- DPO objective --------------------------------------------------------
BETA = 0.1                # KL strength. Lower = further from the SFT model.
LABEL_SMOOTHING = 0.0     # cDPO: assume this fraction of labels are flipped.
NLL_WEIGHT = 0.2          # weight on the SFT loss over the chosen reply
LENGTH_NORMALIZE = False  # mean instead of sum over completion tokens

# ---- Optimization ---------------------------------------------------------
# A pair is two sequences, so a micro-batch of 8 is 16 forwards through the
# policy and 16 through the reference.
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH", "8"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_EPOCHS = 1            # DPO overfits preferences fast; one pass is standard
PEAK_LR = 5e-6            # an order of magnitude below SFT -- the reference
                          # term does not stop a large step from wrecking the
                          # policy, it only penalizes the result
MIN_LR_RATIO = 0.1
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
GRAD_CLIP_NORM = 1.0

# ---- Data -----------------------------------------------------------------
DATASET_BATCH_SIZE = 64        # pairs per dataset __getitem__
LENGTH_BUCKET_BATCHES = 32     # sort this many micro-batches by length together
PAD_TO_MULTIPLE = 128          # keeps torch.compile to a handful of shapes

# ---- Intervals (optimizer steps) ------------------------------------------
LOG_EVERY = 25
EVAL_EVERY = 200
SAMPLE_EVERY = 400
CKPT_EVERY = 200
PLOT_EVERY = 100

USE_COMPILE = os.environ.get("USE_COMPILE", "1") == "1"
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
# Only worth enabling for NUM_EPOCHS > 1 (or a resume): with one shuffled pass
# every batch is a miss, and the cache is then pure bookkeeping.
CACHE_REF_LOGPS = os.environ.get("CACHE_REF_LOGPS", "0") == "1"
LOSS_CHUNK_SIZE = int(os.environ.get("LOSS_CHUNK_SIZE", "256"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = "artifacts"
LOG_DIR = "logs"
TOKENIZER_FILE = f"{ARTIFACTS_DIR}/tokenizer-{VOCAB_SIZE}.txt"
SFT_MODEL_FILE = f"{ARTIFACTS_DIR}/sft_model.pt"
SFT_CHECKPOINT_FILE = f"{ARTIFACTS_DIR}/sft_checkpoint_latest.pt"
TRAINED_MODEL_FILE = f"{ARTIFACTS_DIR}/dpo_model.pt"
CHECKPOINT_FILE = f"{ARTIFACTS_DIR}/dpo_checkpoint_latest.pt"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"dpo_{RUN_ID}.log")
METRICS_FILE = os.path.join(LOG_DIR, "dpo_metrics.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("dpo")
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


# ---------------------------------------------------------------------------
# [2] Data
# ---------------------------------------------------------------------------

logger.info("[2] Preparing preference dataset...")
build_dpo_cache()

train_dataset = DPODataset(batch_size=DATASET_BATCH_SIZE, split="train",
                           epochs=NUM_EPOCHS)
val_dataset = DPODataset(batch_size=DATASET_BATCH_SIZE, split="val")
logger.info("    " + train_dataset.describe().replace("\n", "\n    "))
logger.info(f"    val: {len(val_dataset.index):,} pairs")


def collate(examples):
    """Render pairs and stack chosen and rejected into ONE batch.

    Layout of the returned tensors is [chosen block; rejected block], so a
    single forward covers both branches and the two halves are guaranteed to
    have seen identical padding and identical prompt tokens. Row i of the
    chosen half pairs with row i of the rejected half.

    Right-padding needs no attention mask: attention is causal, so real tokens
    never see pads to their right, and pad positions are IGNORE_INDEX anyway.
    """
    rendered = []
    for ex in examples:
        prompt_ids, chosen_ids, rejected_ids = render_pair(
            tokenizer, ex["prompt"], ex["chosen"], ex["rejected"], SEQ_LEN)
        if prompt_ids is None:
            continue
        rendered.append((prompt_ids, chosen_ids, rejected_ids, ex["task"]))

    if not rendered:
        return None

    width = max(len(p) + max(len(c), len(r)) for p, c, r, _ in rendered)
    width = min(SEQ_LEN, math.ceil(width / PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE)

    xs, ys, tasks = [], [], []
    for branch in (1, 2):                    # 1 = chosen, 2 = rejected
        for prompt_ids, chosen_ids, rejected_ids, task in rendered:
            reply = chosen_ids if branch == 1 else rejected_ids
            full = prompt_ids + reply
            pad = width - len(full)
            full = full + [PAD_ID] * pad
            # Loss/logprob is over the reply only. Position i of x predicts
            # token i+1, so the first supervised target sits at index
            # len(prompt) - 1 in y.
            y = [IGNORE_INDEX] * (width - 1)
            for i in range(len(prompt_ids) - 1, len(prompt_ids) + len(reply) - 1):
                y[i] = full[i + 1]
            xs.append(full[:-1])
            ys.append(y)
            if branch == 1:
                tasks.append(task)

    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            tasks)


def iter_micro_batches(dataset, micro_batch_size, shuffle=True, bucket=True):
    """Yield padded micro-batches, grouping similar lengths together.

    Preference replies range from a sentence to a full page. Batching randomly
    means most of every batch is padding, and here that padding is paid twice
    because every example is two sequences.
    """
    loader = DataLoader(dataset, batch_size=None, shuffle=shuffle,
                        num_workers=0, collate_fn=None)

    pool = []
    pool_target = micro_batch_size * LENGTH_BUCKET_BATCHES if bucket else micro_batch_size

    def flush(pool):
        if bucket:
            pool.sort(key=lambda ex: (
                sum(len(m["content"]) for m in ex["prompt"])
                + max(len(ex["chosen"]), len(ex["rejected"]))))
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
# [3] Policy + frozen reference
# ---------------------------------------------------------------------------

def load_sft_state():
    for path in (SFT_MODEL_FILE, SFT_CHECKPOINT_FILE):
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        logger.info(f"    starting from {path}")
        return state
    raise FileNotFoundError(
        f"Neither {SFT_MODEL_FILE} nor {SFT_CHECKPOINT_FILE} exists -- DPO runs "
        f"on top of SFT, not on a raw pretrained model. Run "
        f"train_transformer_sft.py first.")


def build_model():
    return Transformer(
        vocab_size=MODEL_VOCAB_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        n_dim=EMBEDDING_DIM,
        n_seq=SEQ_LEN,
        activation_checkpointing=False,
        loss_chunk_size=LOSS_CHUNK_SIZE,
    ).to(DEVICE)


logger.info("[3] Loading policy and reference models...")
sft_state = load_sft_state()

model = build_model()
missing, unexpected = model.load_state_dict(sft_state, strict=False)
if missing:
    logger.warning(f"    missing keys: {missing}")
if unexpected:
    logger.warning(f"    unexpected keys: {unexpected}")
logger.info(f"    policy: {model.get_param_count():,} params")

ref_model = build_model()
ref_model.load_state_dict(sft_state, strict=False)
ref_model.eval()
ref_model.requires_grad_(False)
# The reference is never updated, so it does not need fp32 master weights. In
# bf16 it costs half the memory and runs its forward faster; the reference
# log-probs enter the loss as a difference against the policy, where a bf16
# rounding error is far below the signal.
if DEVICE.type == "cuda" and AMP_DTYPE == torch.bfloat16:
    ref_model = ref_model.to(torch.bfloat16)
logger.info(f"    reference: frozen copy, dtype "
            f"{next(ref_model.parameters()).dtype}")

del sft_state


# ---------------------------------------------------------------------------
# Sequence log-probabilities
# ---------------------------------------------------------------------------

def _logp_chunk(net, h, targets):
    """Per-row summed log-prob and token count for one slice of the sequence.

    Projecting the whole batch to (B, T, vocab) in fp32 is 32k floats per token
    -- at batch 16 x 1024 that is a 2 GB tensor that autograd would also keep.
    Chunking over T and recomputing in the backward pass keeps peak memory to
    one chunk, the same trade simple_transformer.calculate_loss makes.

    cross_entropy already returns 0 at ignored positions, so summing it gives
    the completion log-prob with no extra masking.
    """
    logits = net.logit_proj(h).float()
    ce = F.cross_entropy(
        logits.reshape(-1, net.vocab_size), targets.reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX,
    ).view(targets.shape)
    n = (targets != IGNORE_INDEX).to(ce.dtype)
    return torch.stack(((-ce).sum(dim=1), n.sum(dim=1)), dim=1)


def sequence_logprobs(net, xs, ys, grad=True):
    """(logp_per_row, token_count_per_row) over the completion tokens."""
    h = net.forward_hidden(xs)
    T = h.size(1)
    chunk = LOSS_CHUNK_SIZE or T
    totals = None
    for i in range(0, T, chunk):
        hs, ts = h[:, i:i + chunk], ys[:, i:i + chunk]
        if grad and torch.is_grad_enabled():
            part = torch.utils.checkpoint.checkpoint(
                _logp_chunk, net, hs, ts, use_reentrant=False)
        else:
            part = _logp_chunk(net, hs, ts)
        totals = part if totals is None else totals + part
    return totals[:, 0], totals[:, 1]


def dpo_loss(policy_logp, ref_logp, counts, half):
    """Sigmoid DPO loss plus diagnostics.

    `half` is the number of pairs: rows [0:half] are the chosen completions and
    rows [half:] are their rejected partners, in the same order.
    """
    if LENGTH_NORMALIZE:
        denom = counts.clamp(min=1.0)
        policy_logp = policy_logp / denom
        ref_logp = ref_logp / denom

    pi_chosen, pi_rejected = policy_logp[:half], policy_logp[half:]
    ref_chosen, ref_rejected = ref_logp[:half], ref_logp[half:]

    chosen_reward = BETA * (pi_chosen - ref_chosen)
    rejected_reward = BETA * (pi_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward

    if LABEL_SMOOTHING > 0:
        # cDPO: some fraction of the human labels are assumed wrong, so the
        # loss stops pushing once a pair is confidently separated.
        loss = -(F.logsigmoid(margin) * (1 - LABEL_SMOOTHING)
                 + F.logsigmoid(-margin) * LABEL_SMOOTHING)
    else:
        loss = -F.logsigmoid(margin)

    stats = torch.stack((
        loss.detach().mean(),
        (margin.detach() > 0).float().mean(),      # reward accuracy
        margin.detach().mean(),
        chosen_reward.detach().mean(),
        rejected_reward.detach().mean(),
        pi_chosen.detach().mean(),
        pi_rejected.detach().mean(),
    ))
    return loss.mean(), stats


STAT_NAMES = ("dpo_loss", "reward_acc", "margin", "reward_chosen",
              "reward_rejected", "logp_chosen", "logp_rejected")


# ---------------------------------------------------------------------------
# Reference log-prob cache
# ---------------------------------------------------------------------------

class RefLogpCache:
    """Reference log-probs for batches already seen this run.

    The reference is frozen, so its output for a given batch never changes, and
    skipping it removes roughly a third of the step. But with a single epoch
    over shuffled data every batch is new, so this only earns its keep for
    NUM_EPOCHS > 1 or a resume -- hence off by default.

    Keys come from the *host-side* batch, computed before the H2D copy. Hashing
    the device tensor instead would mean a `.cpu()` per micro-batch, which is a
    blocking sync on the critical path -- more expensive than the forward it is
    trying to avoid.
    """

    def __init__(self, enabled=True, max_entries=20000):
        self.enabled = enabled
        self.max_entries = max_entries
        self.store = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(xs_cpu, ys_cpu):
        return (hash(xs_cpu.numpy().tobytes()), hash(ys_cpu.numpy().tobytes()))

    def get(self, key, device):
        if not self.enabled or key is None:
            return None
        hit = self.store.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit.to(device, non_blocking=True)

    def put(self, key, logp):
        if not self.enabled or key is None or len(self.store) >= self.max_entries:
            return
        self.store[key] = logp.detach().to("cpu")


ref_cache = RefLogpCache(enabled=CACHE_REF_LOGPS)


@torch.no_grad()
def reference_logprobs(xs, ys, key=None):
    cached = ref_cache.get(key, xs.device)
    if cached is not None:
        return cached
    with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                        enabled=DEVICE.type == "cuda"):
        logp, _ = sequence_logprobs(ref_model, xs, ys, grad=False)
    logp = logp.float()
    ref_cache.put(key, logp)
    return logp


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------

train_model = model
if USE_COMPILE:
    try:
        torch._dynamo.config.suppress_errors = True
        # Bucketed lengths mean a handful of distinct shapes rather than one.
        torch._dynamo.config.cache_size_limit = 64
        model.forward_hidden = torch.compile(model.forward_hidden,
                                             mode=COMPILE_MODE)
        ref_model.forward_hidden = torch.compile(ref_model.forward_hidden,
                                                 mode=COMPILE_MODE)
        logger.info(f"    torch.compile enabled (mode={COMPILE_MODE}); one "
                    f"warmup per length bucket")
    except Exception as e:
        logger.warning(f"    torch.compile failed, running eager: {e}")

_fused_ok = DEVICE.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
optimizer = torch.optim.AdamW(
    model.param_groups(weight_decay=WEIGHT_DECAY),
    lr=PEAK_LR, betas=BETAS, eps=1e-8,
    **({"fused": True} if _fused_ok else {}),
)
scaler = torch.amp.GradScaler(enabled=USE_SCALER) if USE_SCALER else None

# Estimated: collate drops over-length pairs, so the true count is a little
# lower. The cosine schedule is clamped, so overshooting is harmless.
est_micro = len(train_dataset) * DATASET_BATCH_SIZE / MICRO_BATCH_SIZE
MAX_STEPS = max(1, int(est_micro / GRAD_ACCUM_STEPS))
WARMUP_STEPS = max(1, int(WARMUP_FRAC * MAX_STEPS))
MIN_LR = PEAK_LR * MIN_LR_RATIO
logger.info(f"    ~{MAX_STEPS:,} optimizer steps over {NUM_EPOCHS} epoch(s) "
            f"| warmup {WARMUP_STEPS} | LR {PEAK_LR:.2e} -> {MIN_LR:.2e} "
            f"| beta {BETA} | nll weight {NLL_WEIGHT}")


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
    "steps": [], "train_loss": [], "lr": [], "grad_norm": [],
    "reward_acc": [], "margin": [], "logp_chosen": [], "logp_rejected": [],
    "val_steps": [], "val_loss": [], "val_reward_acc": [], "val_margin": [],
    "val_by_task": [],
}


def save_metrics():
    tmp = METRICS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, METRICS_FILE)


def save_plots():
    if not history["steps"]:
        return
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(9, 12))

    ax1.plot(history["steps"], history["train_loss"], lw=1, label="train")
    if history["val_steps"]:
        ax1.plot(history["val_steps"], history["val_loss"], marker="o", ms=3,
                 color="tab:red", label="val")
    ax1.axhline(math.log(2), color="grey", lw=0.8, ls=":",
                label="chance (log 2)")
    ax1.set_ylabel("dpo loss")
    ax1.set_title("DPO")
    ax1.legend(fontsize=8)

    ax2.plot(history["steps"], history["reward_acc"], lw=1, color="tab:blue",
             label="train")
    if history["val_steps"]:
        ax2.plot(history["val_steps"], history["val_reward_acc"], marker="o",
                 ms=3, color="tab:red", label="val")
    ax2.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax2.set_ylabel("reward accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)

    # The diagnostic that matters most: if both lines fall together the model
    # is learning to rank by getting worse at writing.
    ax3.plot(history["steps"], history["logp_chosen"], lw=1, color="tab:green",
             label="chosen")
    ax3.plot(history["steps"], history["logp_rejected"], lw=1, color="tab:red",
             alpha=0.7, label="rejected")
    ax3.set_ylabel("policy logp")
    ax3.legend(fontsize=8)

    ax4.plot(history["steps"], history["grad_norm"], color="tab:orange", lw=1)
    ax4.set_ylabel("grad norm")
    ax4.set_yscale("log")
    ax4.set_xlabel("optimizer step")

    fig.tight_layout()
    fig.savefig(os.path.join(LOG_DIR, "dpo_curves.png"), dpi=110)
    plt.close(fig)


VAL_BATCHES = None


def build_val_batches():
    batches = [b for b in iter_micro_batches(val_dataset, MICRO_BATCH_SIZE,
                                             shuffle=False, bucket=True)]
    logger.info(f"    {len(batches)} validation micro-batches")
    return batches


@torch.no_grad()
def evaluate():
    """DPO loss, reward accuracy and margin on held-out pairs, per task."""
    train_model.eval()
    totals = {}
    for xs, ys, tasks in VAL_BATCHES:
        key = ref_cache.key(xs, ys) if ref_cache.enabled else None
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)
        half = len(tasks)

        ref_logp = reference_logprobs(xs, ys, key)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            pi_logp, counts = sequence_logprobs(model, xs, ys, grad=False)
        pi_logp = pi_logp.float()

        if LENGTH_NORMALIZE:
            denom = counts.clamp(min=1.0)
            pi_logp, ref_logp = pi_logp / denom, ref_logp / denom
        margin = BETA * ((pi_logp[:half] - ref_logp[:half])
                         - (pi_logp[half:] - ref_logp[half:]))
        losses = -F.logsigmoid(margin)

        for task, l, m in zip(tasks, losses.tolist(), margin.tolist()):
            acc = totals.setdefault(task, [0.0, 0.0, 0, 0])
            acc[0] += l
            acc[1] += m
            acc[2] += 1 if m > 0 else 0
            acc[3] += 1
    train_model.train()

    by_task = {t: {"loss": s / n, "acc": w / n, "margin": mg / n}
               for t, (s, mg, w, n) in totals.items()}
    n_all = sum(v[3] for v in totals.values()) or 1
    overall = {
        "loss": sum(v[0] for v in totals.values()) / n_all,
        "acc": sum(v[2] for v in totals.values()) / n_all,
        "margin": sum(v[1] for v in totals.values()) / n_all,
    }
    return overall, by_task


SAMPLE_PROMPTS = [
    ("chat", "Hi! How are you doing today?"),
    ("helpful", "My laptop fan is very loud. What should I check first?"),
    ("instruct", "Explain what a hash table is to someone who has never "
                 "programmed. Keep it under four sentences."),
    ("harmless", "What's a good way to deal with a neighbour who plays loud "
                 "music at 3am?"),
]


@torch.inference_mode()
def log_samples(step):
    model.eval()
    for task, prompt in SAMPLE_PROMPTS:
        prompt_ids, _, _ = render_pair(
            tokenizer, [{"role": "user", "content": prompt}],
            "placeholder", "placeholder alternative", SEQ_LEN)
        if prompt_ids is None:
            continue
        x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        out = model.generate(x, max_count=100, temperature=0.7, top_k=50,
                             top_p=0.9, eos_token_id=EOS_ID,
                             valid_vocab_size=actual_vocab_size)
        reply = tokenizer.decode(out[0, len(prompt_ids):].cpu().tolist())
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
        "dpo": {"beta": BETA, "nll_weight": NLL_WEIGHT,
                "label_smoothing": LABEL_SMOOTHING,
                "length_normalize": LENGTH_NORMALIZE},
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


class _FreshStart(Exception):
    """Not an error: the checkpoint is for a different architecture, so it is
    set aside and this run starts from the SFT weights instead."""


if os.path.exists(CHECKPOINT_FILE):
    try:
        ck = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        want = model.state_dict()
        have = ck["model_state_dict"]
        shape_diff = [k for k in want
                      if k not in have or tuple(have[k].shape) != tuple(want[k].shape)]
        if shape_diff:
            # load_state_dict copies as it goes and only reports shape errors at
            # the end, which would leave the policy half-restored and half-SFT.
            stale = f"{CHECKPOINT_FILE}.{RUN_ID}.bak"
            os.replace(CHECKPOINT_FILE, stale)
            logger.warning(
                f"Checkpoint architecture differs ({len(shape_diff)} tensors, "
                f"e.g. {shape_diff[0]}); starting fresh from the SFT weights. "
                f"Previous checkpoint kept at {stale}")
            raise _FreshStart

        model.load_state_dict(have)
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
        last_val_loss = float(ck.get("val_loss", float("nan")))

        old = ck.get("dpo") or {}
        changed = {k: (v, cur) for k, v, cur in (
            ("beta", old.get("beta"), BETA),
            ("nll_weight", old.get("nll_weight"), NLL_WEIGHT),
            ("length_normalize", old.get("length_normalize"), LENGTH_NORMALIZE),
        ) if v is not None and v != cur}
        if changed:
            # Not fatal, but the loss curve before and after is not one curve.
            logger.warning(f"DPO hyperparameters changed on resume: {changed}")
        logger.info(f"Resumed: epoch={start_epoch}, step={global_step}")
    except _FreshStart:
        pass
    except Exception as e:
        logger.warning(f"Failed to load {CHECKPOINT_FILE}: {e}")


# ---------------------------------------------------------------------------
# [4b] Train
# ---------------------------------------------------------------------------

logger.info("[4] Training...")
VAL_BATCHES = build_val_batches()

base_val, base_by_task = evaluate()
logger.info(f"    SFT baseline: dpo loss {base_val['loss']:.4f} | "
            f"reward acc {base_val['acc']:.3f} | margin {base_val['margin']:+.4f}")
logger.info("    (policy == reference at step 0, so margin is 0 and loss is "
            "log 2 = 0.6931 by construction)")
for task, v in sorted(base_by_task.items()):
    logger.info(f"      {task:12s} loss {v['loss']:.4f}  acc {v['acc']:.3f}")

train_model.train()
optimizer.zero_grad(set_to_none=True)

training_start = time.time()
accum_stats = torch.zeros(len(STAT_NAMES), device=DEVICE)
accum_nll = torch.zeros((), device=DEVICE)
accum_micro = 0
accum_pairs = 0
skipped_steps = 0
step_timer = time.time()

all_params = [p for p in model.parameters() if p.requires_grad]

for epoch in range(start_epoch, NUM_EPOCHS):
    logger.info(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    pbar = tqdm(desc=f"epoch {epoch + 1}", unit="step")

    for xs, ys, tasks in iter_micro_batches(train_dataset, MICRO_BATCH_SIZE):
        key = ref_cache.key(xs, ys) if ref_cache.enabled else None
        xs = xs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)
        half = len(tasks)

        ref_logp = reference_logprobs(xs, ys, key)

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                            enabled=DEVICE.type == "cuda"):
            pi_logp, counts = sequence_logprobs(model, xs, ys, grad=True)
            pi_logp = pi_logp.float()
            loss, stats = dpo_loss(pi_logp, ref_logp, counts, half)

            nll = torch.zeros((), device=DEVICE)
            if NLL_WEIGHT:
                # Plain SFT loss on the chosen replies, token-averaged. Without
                # it, DPO is free to satisfy the ranking by pushing both
                # log-probs down together.
                nll = -(pi_logp[:half].sum() / counts[:half].sum().clamp(min=1.0))
                loss = loss + NLL_WEIGHT * nll

            loss = loss / GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Everything stays on-device: .item() per micro-batch would sync.
        accum_stats += stats
        accum_nll += nll.detach()
        accum_micro += 1
        accum_pairs += half

        if accum_micro < GRAD_ACCUM_STEPS:
            continue

        lr = lr_at(global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None:
            scaler.unscale_(optimizer)

        # Clip and the non-finite guard in one pass, with no host sync: a
        # non-finite norm scales the grads to zero, so the step is a no-op and
        # the CPU never has to wait to find out whether to take it.
        grads = [p.grad for p in all_params if p.grad is not None]
        grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        finite = torch.isfinite(grad_norm)
        scale = torch.where(finite,
                            (GRAD_CLIP_NORM / (grad_norm + 1e-6)).clamp(max=1.0),
                            grad_norm.new_zeros(()))
        torch._foreach_mul_(grads, scale)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # One sync per optimizer step, for every number at once.
        packed = torch.cat((accum_stats / accum_micro,
                            torch.stack((accum_nll / accum_micro,
                                         grad_norm, finite.float())))).tolist()
        s = dict(zip(STAT_NAMES, packed))
        step_nll, step_gn, step_finite = packed[-3], packed[-2], packed[-1]

        if step_finite < 0.5:
            skipped_steps += 1
            logger.warning(f"    non-finite grad norm at step {global_step}; "
                           f"grads zeroed ({skipped_steps} total)")

        accum_stats.zero_()
        accum_nll.zero_()
        accum_micro = 0
        pairs_this_step, accum_pairs = accum_pairs, 0
        global_step += 1

        step_time = time.time() - step_timer
        step_timer = time.time()

        history["steps"].append(global_step)
        history["train_loss"].append(s["dpo_loss"])
        history["lr"].append(lr)
        history["grad_norm"].append(step_gn)
        history["reward_acc"].append(s["reward_acc"])
        history["margin"].append(s["margin"])
        history["logp_chosen"].append(s["logp_chosen"])
        history["logp_rejected"].append(s["logp_rejected"])

        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{s['dpo_loss']:.4f}",
            "acc": f"{s['reward_acc']:.2f}",
            "margin": f"{s['margin']:+.3f}",
            "logp_c": f"{s['logp_chosen']:.0f}",
            "nll": f"{step_nll:.2f}",
            "pairs/s": f"{pairs_this_step / max(1e-6, step_time):.1f}",
        })

        if global_step % LOG_EVERY == 0:
            recent = history["train_loss"][-LOG_EVERY:]
            acc = history["reward_acc"][-LOG_EVERY:]
            logger.info(
                f"    step {global_step}/{MAX_STEPS} | "
                f"loss {sum(recent)/len(recent):.4f} | "
                f"acc {sum(acc)/len(acc):.3f} | margin {s['margin']:+.4f} | "
                f"logp c/r {s['logp_chosen']:.1f}/{s['logp_rejected']:.1f} | "
                f"lr {lr:.2e} | gn {step_gn:.2f}")

        if global_step % EVAL_EVERY == 0:
            val, by_task = evaluate()
            last_val_loss = val["loss"]
            history["val_steps"].append(global_step)
            history["val_loss"].append(val["loss"])
            history["val_reward_acc"].append(val["acc"])
            history["val_margin"].append(val["margin"])
            history["val_by_task"].append(by_task)
            detail = " ".join(f"{t}={v['acc']:.2f}" for t, v in sorted(by_task.items()))
            logger.info(f"    >> val loss {val['loss']:.4f} | acc {val['acc']:.3f} "
                        f"(baseline 0.500) | margin {val['margin']:+.4f} | {detail}")

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
        # Partial accumulation: a fraction of a step, and with a scaler never
        # unscaled. Not worth a wrong final update.
        optimizer.zero_grad(set_to_none=True)
        accum_stats.zero_()
        accum_nll.zero_()
        accum_micro = 0
        accum_pairs = 0

    val, by_task = evaluate()
    logger.info(f"    epoch {epoch + 1} val loss {val['loss']:.4f} | "
                f"reward acc {val['acc']:.3f}")
    snap = f"{ARTIFACTS_DIR}/dpo_model_epoch{epoch + 1}.pt"
    atomic_save(model.state_dict(), snap)
    logger.info(f"    epoch snapshot -> {snap}")
    save_checkpoint(CHECKPOINT_FILE, global_step, epoch + 1, val["loss"])

final_val, final_by_task = evaluate()
history["val_steps"].append(global_step)
history["val_loss"].append(final_val["loss"])
history["val_reward_acc"].append(final_val["acc"])
history["val_margin"].append(final_val["margin"])
history["val_by_task"].append(final_by_task)

logger.info(f"Final val loss {final_val['loss']:.4f} "
            f"(baseline {base_val['loss']:.4f}) | "
            f"reward acc {final_val['acc']:.3f} (baseline {base_val['acc']:.3f})")
for task in sorted(final_by_task):
    v, b = final_by_task[task], base_by_task.get(task, {})
    logger.info(f"    {task:12s} acc {v['acc']:.3f} "
                f"({v['acc'] - b.get('acc', v['acc']):+.3f}) | "
                f"margin {v['margin']:+.4f}")
if ref_cache.enabled:
    total = ref_cache.hits + ref_cache.misses
    logger.info(f"    reference cache: {ref_cache.hits:,}/{total:,} hits "
                f"({100 * ref_cache.hits / max(1, total):.1f}%)")

atomic_save(model.state_dict(), TRAINED_MODEL_FILE)
save_checkpoint(CHECKPOINT_FILE, global_step, NUM_EPOCHS, final_val["loss"])
save_plots()
save_metrics()

logger.info(f"Total time: {timedelta(seconds=int(time.time() - training_start))} "
            f"| steps {global_step} | skipped {skipped_steps}")
logger.info(f"Model saved to {TRAINED_MODEL_FILE}")

logger.info("[5] Final samples...")
log_samples(global_step)
