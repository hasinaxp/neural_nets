import json
import math
import os
import random

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

SFT_DATASET_FOLDER = "dataset/sft"
MANIFEST_FILE = f"{SFT_DATASET_FOLDER}/manifest.json"
CACHE_FOLDER = f"{SFT_DATASET_FOLDER}/cache"
TRAIN_CACHE = f"{CACHE_FOLDER}/sft_train.parquet"
VAL_CACHE = f"{CACHE_FOLDER}/sft_val.parquet"

IGNORE_INDEX = -100
SEED = 1337

# Share of the training mixture per task. Raw row counts are wildly unbalanced
# (SQL alone is ~185k rows), so the index is resampled to hit these instead.
TASK_WEIGHTS = {
    "chat": 0.35,
    "extractive_qa": 0.20,
    "summarization": 0.15,
    "sql": 0.20,
    "instruct": 0.10,
}

VAL_PER_TASK = 300          # held-out examples per task
MAX_TASK_REPEATS = 3        # a task may be oversampled at most this much per epoch
MAX_UNANSWERABLE_FRAC = 0.25   # cap on SQuAD-v2 "no answer" examples

# Instruction paraphrases. One fixed phrasing per task teaches the model to key
# off that exact string; a handful teaches the task.
SUMMARY_PROMPTS = [
    "Summarize the conversation below.",
    "Give a short summary of this dialogue.",
    "What happened in this conversation? Answer in a sentence or two.",
    "Write a brief summary.",
]
QA_PROMPTS = [
    "Answer the question using only the passage below.",
    "Read the passage and answer the question. If the passage does not say, reply that you don't know.",
    "Use the context to answer the question.",
]
SQL_PROMPTS = [
    "Write a SQL query that answers the question, using the schema provided.",
    "Given the schema below, write a SQL query for the question.",
    "Translate the question into a SQL query against this schema.",
]
NO_ANSWER_REPLIES = [
    "The passage doesn't say.",
    "I can't answer that from the given passage.",
    "That information isn't in the context.",
]


# ---------------------------------------------------------------------------
# Per-source normalizers: dataframe row -> [{"role": ..., "content": ...}, ...]
# Return None to drop the row.
# ---------------------------------------------------------------------------

def _clean(text):
    if not isinstance(text, str):
        return ""
    return text.encode("ascii", errors="ignore").decode("ascii").strip()


def _from_messages(row, rng):
    """smoltalk / no_robots: already a list of {role, content}."""
    msgs = row.get("messages")
    if msgs is None or len(msgs) == 0:
        return None
    out = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = _clean(m.get("content") or m.get("value"))
        if role == "system":
            continue          # no system-role token in the tokenizer
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    if len(out) < 2 or out[0]["role"] != "user":
        return None
    return out


def _from_soda(row, rng):
    """soda: `dialogue` is a list of utterances alternating between speakers."""
    turns = row.get("dialogue")
    if turns is None or len(turns) < 2:
        return None
    out = []
    for i, utt in enumerate(turns):
        utt = _clean(utt)
        if not utt:
            continue
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": utt})
    if len(out) < 2 or out[0]["role"] != "user":
        return None
    if out[-1]["role"] != "assistant":
        out = out[:-1]
    return out or None


def _from_squad(row, rng):
    question = _clean(row.get("question"))
    context = _clean(row.get("context"))
    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    if not question or not context:
        return None

    if texts is not None and len(texts) > 0:
        answer = _clean(texts[0])
        if not answer:
            return None
    else:
        # Unanswerable. Teaching abstention is valuable, but too many of these
        # and the model learns to refuse everything.
        answer = rng.choice(NO_ANSWER_REPLIES)

    prompt = f"{rng.choice(QA_PROMPTS)}\n\nPassage:\n{context}\n\nQuestion: {question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}]


def _from_sciq(row, rng):
    question = _clean(row.get("question"))
    answer = _clean(row.get("correct_answer"))
    support = _clean(row.get("support"))
    if not question or not answer or not support:
        return None      # skip closed-book rows: no passage, no grounding
    prompt = f"{rng.choice(QA_PROMPTS)}\n\nPassage:\n{support}\n\nQuestion: {question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}]


def _from_dialogue_summary(row, rng):
    dialogue = _clean(row.get("dialogue"))
    summary = _clean(row.get("summary"))
    if not dialogue or not summary:
        return None
    prompt = f"{rng.choice(SUMMARY_PROMPTS)}\n\n{dialogue}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": summary}]


def _from_xsum(row, rng):
    document = _clean(row.get("document"))
    summary = _clean(row.get("summary"))
    if not document or not summary:
        return None
    prompt = f"Summarize the article below in one sentence.\n\n{document}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": summary}]


def _from_sql_create_context(row, rng):
    question = _clean(row.get("question"))
    schema = _clean(row.get("context"))
    query = _clean(row.get("answer"))
    if not question or not schema or not query:
        return None
    prompt = f"{rng.choice(SQL_PROMPTS)}\n\nSchema:\n{schema}\n\nQuestion: {question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": query}]


# Complexity tiers a ~110M model can plausibly learn. Window functions and
# multi-join queries are dropped for the first SFT pass.
SQL_COMPLEXITY_KEEP = (
    "basic sql", "aggregation", "single join", "basic joins",
    "subqueries", "window functions", "set operations", "multiple_joins",
)
SQL_COMPLEXITY_DROP = ("window functions", "set operations", "multiple_joins",
                       "multiple joins", "cte", "cte with joins")


def _from_gretel_sql(row, rng):
    question = _clean(row.get("sql_prompt"))
    schema = _clean(row.get("sql_context"))
    query = _clean(row.get("sql"))
    if not question or not schema or not query:
        return None
    complexity = str(row.get("sql_complexity") or "").lower()
    if any(d in complexity for d in SQL_COMPLEXITY_DROP):
        return None
    prompt = f"{rng.choice(SQL_PROMPTS)}\n\nSchema:\n{schema}\n\nQuestion: {question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": query}]


DOLLY_DROP_CATEGORIES = ("closed_qa", "open_qa", "general_qa")


def _from_dolly(row, rng):
    instruction = _clean(row.get("instruction"))
    context = _clean(row.get("context"))
    response = _clean(row.get("response"))
    if not instruction or not response:
        return None
    category = str(row.get("category") or "")
    # Closed-book QA teaches confident fabrication at this scale. Keep the
    # category only when a context passage is actually supplied.
    if category in DOLLY_DROP_CATEGORIES and not context:
        return None
    prompt = instruction if not context else f"{instruction}\n\n{context}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]


NORMALIZERS = {
    "smoltalk": _from_messages,
    "no-robots": _from_messages,
    "soda": _from_soda,
    "squad-v2": _from_squad,
    "sciq": _from_sciq,
    "samsum": _from_dialogue_summary,
    "dialogsum": _from_dialogue_summary,
    "xsum": _from_xsum,
    "sql-create-context": _from_sql_create_context,
    "synthetic-text-to-sql": _from_gretel_sql,
    "dolly": _from_dolly,
}


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

def render_conversation(tokenizer, messages, seq_len=1024, truncate=True):
    """Tokenize a conversation into ids plus a per-token loss mask.

    Layout: <|BOS|> <|USER|> ...prompt... <|ASSISTANT|> ...reply... <|EOS|>

    Loss is on assistant content and its closing EOS only. Everything else is
    masked -- otherwise the model spends most of its capacity learning to
    reproduce prompts, which is the most common SFT bug.
    """
    bos = tokenizer.special_tokens["<|BOS|>"]
    eos = tokenizer.special_tokens["<|EOS|>"]
    user = tokenizer.special_tokens["<|USER|>"]
    assistant = tokenizer.special_tokens["<|ASSISTANT|>"]

    ids = [bos]
    mask = [0]

    for msg in messages:
        content = tokenizer.encode(msg["content"])
        if msg["role"] == "user":
            ids.append(user)
            mask.append(0)
            ids.extend(content)
            mask.extend([0] * len(content))
        else:
            ids.append(assistant)
            mask.append(0)          # the role marker itself is part of the prompt
            ids.extend(content)
            mask.extend([1] * len(content))
            ids.append(eos)
            mask.append(1)          # learning to stop is the point

    if len(ids) <= seq_len:
        return ids, mask

    if not truncate:
        return None, None

    # Too long: trim the first user message, which holds the passage/schema.
    # Never trim the assistant reply -- a truncated target teaches the model to
    # produce truncated answers.
    overflow = len(ids) - seq_len
    first_user_start = 2                     # after <|BOS|> <|USER|>
    first_user_end = first_user_start
    while first_user_end < len(ids) and mask[first_user_end] == 0 \
            and ids[first_user_end] not in (assistant,):
        first_user_end += 1

    trimmable = first_user_end - first_user_start
    if trimmable <= overflow + 16:
        return None, None                    # nothing left to say; drop it

    cut_from = first_user_end - overflow
    ids = ids[:cut_from] + ids[first_user_end:]
    mask = mask[:cut_from] + mask[first_user_end:]
    return ids, mask


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def _read_source(key, entry):
    """Load every parquet for one source, splitting train vs validation files."""
    folder = entry["folder"]
    train_frames, val_frames = [], []
    for name in sorted(entry["files"]):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  could not read {path}: {e}")
            continue
        lowered = name.lower()
        if any(s in lowered for s in ("validation", "valid", "dev")):
            val_frames.append(df)
        else:
            train_frames.append(df)

    train = pd.concat(train_frames, ignore_index=True) if train_frames else None
    val = pd.concat(val_frames, ignore_index=True) if val_frames else None
    return train, val


def _normalize_frame(key, task, df, rng):
    fn = NORMALIZERS.get(key)
    if fn is None:
        print(f"  no normalizer for {key}, skipping")
        return []

    rows = []
    unanswerable = 0
    kept = 0
    for record in df.to_dict("records"):
        messages = fn(record, rng)
        if not messages:
            continue
        if key == "squad-v2":
            is_no_answer = messages[-1]["content"] in NO_ANSWER_REPLIES
            if is_no_answer:
                if unanswerable > MAX_UNANSWERABLE_FRAC * max(1, kept):
                    continue
                unanswerable += 1
        kept += 1
        rows.append({
            "source": key,
            "task": task,
            "messages": json.dumps(messages),
        })
    return rows


def build_sft_cache(force=False, val_per_task=VAL_PER_TASK):
    """Normalize every downloaded source into two parquet files.

    Mirrors the pretrain side's "parquet on disk, read lazily" approach rather
    than re-parsing eleven different schemas on every training run.
    """
    if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE) and not force:
        return TRAIN_CACHE, VAL_CACHE

    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found -- run dataset_sft_download.py first")

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    rng = random.Random(SEED)

    train_rows, val_rows = [], []

    for key, entry in manifest.items():
        task = entry.get("task", "instruct")
        print(f"Normalizing {key} ({task})...")
        train_df, val_df = _read_source(key, entry)

        if train_df is not None:
            rows = _normalize_frame(key, task, train_df, rng)
            print(f"  train: {len(train_df)} rows -> {len(rows)} conversations")
            train_rows.extend(rows)

        if val_df is not None:
            rows = _normalize_frame(key, task, val_df, rng)
            print(f"  val:   {len(val_df)} rows -> {len(rows)} conversations")
            val_rows.extend(rows)

    # Any source without its own validation files donates from train.
    have_val_tasks = {r["task"] for r in val_rows}
    rng.shuffle(train_rows)
    if train_rows:
        by_task = {}
        for r in train_rows:
            by_task.setdefault(r["task"], []).append(r)
        train_rows = []
        for task, rows in by_task.items():
            if task in have_val_tasks:
                need = 0
            else:
                # Never donate more than 10% of a task -- otherwise a small
                # source gets swallowed by the holdout entirely.
                need = min(val_per_task, int(0.1 * len(rows)))
            val_rows.extend(rows[:need])
            train_rows.extend(rows[need:])

    # Trim oversized validation splits; we only need enough to track a curve.
    trimmed = []
    counts = {}
    rng.shuffle(val_rows)
    for r in val_rows:
        n = counts.get(r["task"], 0)
        if n >= val_per_task:
            train_rows.append(r)
            continue
        counts[r["task"]] = n + 1
        trimmed.append(r)
    val_rows = trimmed

    rng.shuffle(train_rows)
    pd.DataFrame(train_rows).to_parquet(TRAIN_CACHE)
    pd.DataFrame(val_rows).to_parquet(VAL_CACHE)

    print(f"\nWrote {len(train_rows):,} train / {len(val_rows):,} val conversations")
    for task in sorted({r["task"] for r in train_rows}):
        n = sum(1 for r in train_rows if r["task"] == task)
        print(f"  {task:15s} {n:>8,} available")

    return TRAIN_CACHE, VAL_CACHE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Map-style dataset over normalized conversations, returning batches.

    Same contract as PretrainTextDataset: __getitem__(i) returns a *batch*
    (a list), so the DataLoader is used with batch_size=None.
    """

    def __init__(self, batch_size=8, split="train", task_weights=None,
                 total_examples=None, seed=SEED, epochs=1):
        self.batch_size = batch_size
        self.split = split

        path = TRAIN_CACHE if split == "train" else VAL_CACHE
        if not os.path.exists(path):
            build_sft_cache()
        self.df = pd.read_parquet(path)
        if len(self.df) == 0:
            raise RuntimeError(f"{path} is empty")

        self.tasks = list(self.df["task"].unique())
        self._by_task = {t: self.df.index[self.df["task"] == t].to_numpy()
                         for t in self.tasks}

        if split == "train":
            weights = task_weights or TASK_WEIGHTS
            self.index = self._build_mixture(weights, total_examples, seed, epochs)
        else:
            self.index = self.df.index.to_numpy()

        self._num_batches = math.ceil(len(self.index) / batch_size)

    def _build_mixture(self, weights, total_examples, seed, epochs):
        """Resample per-task so the mixture matches TASK_WEIGHTS.

        Raw counts are unbalanced by an order of magnitude, so without this the
        model sees mostly SQL. Tasks with too little data are oversampled
        (repeated within an epoch); tasks with too much are subsampled.
        """
        rng = random.Random(seed)
        present = {t: w for t, w in weights.items() if t in self._by_task}
        if not present:
            raise RuntimeError(f"No overlap between TASK_WEIGHTS and data tasks {self.tasks}")

        self.missing_tasks = [t for t in weights if t not in self._by_task]
        if self.missing_tasks:
            print(f"WARNING: no data for tasks {self.missing_tasks} -- their weight "
                  f"is redistributed across {list(present)}")

        norm = sum(present.values())

        if total_examples is None:
            # Size the epoch as large as possible without repeating any task
            # more than MAX_TASK_REPEATS times within one epoch. Sizing by the
            # scarcest task instead would leave most of the data unused.
            total_examples = int(min(
                len(self._by_task[t]) * MAX_TASK_REPEATS / (w / norm)
                for t, w in present.items()
            ))

        index = []
        self.mixture = {}
        for task, weight in present.items():
            want = int(total_examples * weight / norm)
            pool = list(self._by_task[task])
            if want <= len(pool):
                picked = rng.sample(pool, want)
            else:
                reps = want // len(pool)
                picked = pool * reps + rng.sample(pool, want - reps * len(pool))
            self.mixture[task] = {
                "available": len(pool),
                "used": want,
                "repeats": round(want / max(1, len(pool)), 2),
            }
            index.extend(picked)

        index = index * max(1, epochs)
        rng.shuffle(index)
        return index

    def describe(self):
        lines = [f"{self.split}: {len(self.index):,} examples in "
                 f"{self._num_batches:,} batches of {self.batch_size}"]
        for task in getattr(self, "missing_tasks", []):
            lines.append(f"  {task:15s} NO DATA -- weight redistributed")
        for task, info in getattr(self, "mixture", {}).items():
            lines.append(f"  {task:15s} used {info['used']:>7,} of "
                         f"{info['available']:>7,} ({info['repeats']}x)")
        return "\n".join(lines)

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx * self.batch_size
        rows = self.index[start:start + self.batch_size]
        out = []
        for row_id in rows:
            record = self.df.loc[row_id]
            out.append({
                "messages": json.loads(record["messages"]),
                "task": record["task"],
                "source": record["source"],
            })
        return out


if __name__ == "__main__":
    build_sft_cache(force=True)
    ds = SFTDataset(batch_size=8)
    print(ds.describe())
    batch = ds[0]
    print(f"\nfirst batch: {len(batch)} examples")
    for ex in batch[:2]:
        print(f"  [{ex['task']}/{ex['source']}] "
              f"{ex['messages'][0]['content'][:90]!r} -> "
              f"{ex['messages'][-1]['content'][:60]!r}")