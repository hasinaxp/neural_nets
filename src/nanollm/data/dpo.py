"""Preference pairs for DPO: normalization, caching, and the chat rendering.

Mirrors sft_dataset.py. Every source is reduced to one shape:

    {"prompt": [{"role": "user"|"assistant", "content": ...}, ...],
     "chosen": "<assistant reply>",
     "rejected": "<assistant reply>"}

`prompt` is the full conversation *up to but not including* the final assistant
turn, so multi-turn sources (hh-rlhf) survive intact. The two replies share that
prompt exactly -- DPO compares two completions of one context, and any
difference in the prompt tokens between the two branches silently poisons the
implicit reward.
"""

import json
import math
import os
import random
import re

import pandas as pd
from torch.utils.data import Dataset

from .sources import strip_foreign_scripts

DPO_DATASET_FOLDER = "dataset/dpo"
MANIFEST_FILE = f"{DPO_DATASET_FOLDER}/manifest.json"
CACHE_FOLDER = f"{DPO_DATASET_FOLDER}/cache"
TRAIN_CACHE = f"{CACHE_FOLDER}/dpo_train.parquet"
VAL_CACHE = f"{CACHE_FOLDER}/dpo_val.parquet"

SEED = 1337

# Share of the training mixture per task.
TASK_WEIGHTS = {
    "helpful": 0.50,
    "instruct": 0.30,
    "harmless": 0.20,
}

VAL_PER_TASK = 400
MAX_TASK_REPEATS = 2

# Pairs whose two replies are near-identical teach nothing and dominate the
# gradient with noise; pairs where the "rejected" reply is empty teach the model
# that anything beats nothing, which it already knows.
MIN_REPLY_CHARS = 8
MAX_REPLY_CHARS = 4000
MIN_PAIR_EDIT_RATIO = 0.02      # replies must differ by at least this fraction


# ---------------------------------------------------------------------------
# Per-schema normalizers: dataframe row -> dict or None
# ---------------------------------------------------------------------------

def _clean(text):
    if not isinstance(text, str):
        return ""
    # Keep punctuation, symbols and accents; drop only non-Latin letters. The
    # old encode("ascii") pass silently deleted curly quotes, em-dashes, etc.
    return strip_foreign_scripts(text).strip()


_HH_TURN = re.compile(r"\n\n(Human|Assistant):\s*")


def _parse_hh_transcript(text):
    """'\\n\\nHuman: ...\\n\\nAssistant: ...' -> [{role, content}, ...].

    Anthropic's format is a single flat string per branch. Splitting on the
    speaker markers is the only way to recover turn structure, and it has to be
    done for chosen and rejected separately because they diverge partway.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    parts = _HH_TURN.split("\n\n" + text.strip().lstrip("\n"))
    # split() gives ['', 'Human', body, 'Assistant', body, ...]
    msgs = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        content = _clean(parts[i + 1])
        if content:
            msgs.append({"role": role, "content": content})
    return msgs or None


def _split_final_reply(msgs):
    """(prompt_turns, final_assistant_text) or None."""
    if not msgs or msgs[-1]["role"] != "assistant":
        return None
    prompt = msgs[:-1]
    if not prompt or prompt[0]["role"] != "user":
        return None
    return prompt, msgs[-1]["content"]


def _from_hh_transcript(row, rng):
    chosen = _parse_hh_transcript(row.get("chosen"))
    rejected = _parse_hh_transcript(row.get("rejected"))
    if not chosen or not rejected:
        return None
    c = _split_final_reply(chosen)
    r = _split_final_reply(rejected)
    if not c or not r:
        return None
    prompt_c, chosen_text = c
    prompt_r, rejected_text = r
    # The two branches must agree on the context. They usually do; when the
    # divergence happens earlier than the last turn, the pair is not a
    # same-prompt comparison and has to go.
    if [m["content"] for m in prompt_c] != [m["content"] for m in prompt_r]:
        return None
    return {"prompt": prompt_c, "chosen": chosen_text, "rejected": rejected_text}


def _from_hh_prompt_pair(row, rng):
    """Dahoas/rm-static: `prompt` is an hh-style transcript, replies are bare."""
    prompt = _parse_hh_transcript(row.get("prompt"))
    chosen = _clean(row.get("chosen"))
    rejected = _clean(row.get("rejected"))
    if not prompt or not chosen or not rejected:
        return None
    if prompt[-1]["role"] == "assistant":
        prompt = prompt[:-1]
    if not prompt or prompt[0]["role"] != "user":
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _from_messages_pair(row, rng):
    """ultrafeedback_binarized: `chosen`/`rejected` are full message lists."""
    def last_reply(msgs):
        if msgs is None or len(msgs) == 0:
            return None
        out = []
        for m in msgs:
            role = m.get("role")
            content = _clean(m.get("content"))
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return _split_final_reply(out)

    c = last_reply(row.get("chosen"))
    r = last_reply(row.get("rejected"))
    if not c or not r:
        return None
    prompt, chosen_text = c
    _, rejected_text = r
    return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


def _from_system_input_pair(row, rng):
    """Intel / argilla orca pairs: `system` + `input`, bare reply strings."""
    system = _clean(row.get("system"))
    question = _clean(row.get("input") or row.get("question"))
    chosen = _clean(row.get("chosen"))
    rejected = _clean(row.get("rejected"))
    if not question or not chosen or not rejected:
        return None
    # argilla's version flags ties and rows where the "chosen" reply was rated
    # no better than the rejected one.
    if str(row.get("status") or "") == "tie":
        return None
    rating = row.get("rating")
    if rating is not None and len(rating) == 2 and rating[0] <= rating[1]:
        return None
    # There is no system role in the tokenizer, so it is folded into the user
    # turn rather than dropped -- several of these prompts are meaningless
    # without their instruction preamble.
    content = f"{system}\n\n{question}" if system else question
    return {"prompt": [{"role": "user", "content": content}],
            "chosen": chosen, "rejected": rejected}


def _from_shp(row, rng):
    """stanfordnlp/SHP: `labels` == 1 means human_ref_A was preferred."""
    history = _clean(row.get("history"))
    ref_a = _clean(row.get("human_ref_A"))
    ref_b = _clean(row.get("human_ref_B"))
    label = row.get("labels")
    if not history or not ref_a or not ref_b or label is None:
        return None
    chosen, rejected = (ref_a, ref_b) if int(label) == 1 else (ref_b, ref_a)
    return {"prompt": [{"role": "user", "content": history}],
            "chosen": chosen, "rejected": rejected}


NORMALIZERS = {
    "hh_transcript": _from_hh_transcript,
    "hh_prompt_pair": _from_hh_prompt_pair,
    "messages_pair": _from_messages_pair,
    "system_input_pair": _from_system_input_pair,
    "shp": _from_shp,
}


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

def render_pair(tokenizer, prompt_messages, chosen, rejected, seq_len=1024):
    """Tokenize one preference pair against the SFT chat template.

    Returns (prompt_ids, chosen_ids, rejected_ids) or (None, None, None).
    Layout matches sft_dataset.render_conversation exactly:

        <|BOS|> <|USER|> ...prompt... <|ASSISTANT|> ...reply... <|EOS|>

    prompt_ids ends at <|ASSISTANT|>; the reply ids carry the trailing <|EOS|>.
    Both branches reuse the *same* prompt_ids object, so there is no way for the
    two to drift apart.
    """
    bos = tokenizer.special_tokens["<|BOS|>"]
    eos = tokenizer.special_tokens["<|EOS|>"]
    user = tokenizer.special_tokens["<|USER|>"]
    assistant = tokenizer.special_tokens["<|ASSISTANT|>"]

    prompt_ids = [bos]
    for msg in prompt_messages:
        prompt_ids.append(user if msg["role"] == "user" else assistant)
        prompt_ids.extend(tokenizer.encode(msg["content"]))
    prompt_ids.append(assistant)

    chosen_ids = tokenizer.encode(chosen) + [eos]
    rejected_ids = tokenizer.encode(rejected) + [eos]

    longest = max(len(chosen_ids), len(rejected_ids))
    if longest + 8 >= seq_len:
        # A reply that fills the window on its own leaves no room for context,
        # and truncating a reply changes which of the two is actually better.
        return None, None, None

    overflow = len(prompt_ids) + longest - seq_len
    if overflow > 0:
        # Trim the prompt from the front, keeping <|BOS|> <|USER|> and the tail
        # (the question). The head is passage/system text; the tail is the ask.
        keep = len(prompt_ids) - overflow - 2
        if keep < 32:
            return None, None, None
        prompt_ids = prompt_ids[:2] + prompt_ids[len(prompt_ids) - keep:]

    return prompt_ids, chosen_ids, rejected_ids


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def _read_source(entry):
    train_frames, val_frames = [], []
    for name in sorted(entry["files"]):
        path = os.path.join(entry["folder"], name)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  could not read {path}: {e}")
            continue
        lowered = name.lower()
        if any(s in lowered for s in ("validation", "valid", "dev", "test")):
            val_frames.append(df)
        else:
            train_frames.append(df)

    train = pd.concat(train_frames, ignore_index=True) if train_frames else None
    val = pd.concat(val_frames, ignore_index=True) if val_frames else None
    return train, val


def _too_similar(a, b):
    """Cheap near-duplicate check.

    A real edit distance over 100k pairs is not worth the minutes. Comparing
    length and a shared prefix/suffix catches the common case: two replies that
    differ by a word or a trailing sentence, which give the loss almost no
    signal but full weight.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    longest = max(la, lb)
    if longest == 0:
        return True
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        common += 1
    # Prefix and suffix runs can overlap -- "abc" vs "abcabc" matches 3 of each
    # for 6 "common" characters out of a 3-character string. Without the clamp
    # that reads as a perfect match and drops a genuinely different pair.
    common = min(common, la, lb)
    return (longest - common) / longest < MIN_PAIR_EDIT_RATIO


def _normalize_frame(key, task, schema, df, rng):
    fn = NORMALIZERS.get(schema)
    if fn is None:
        print(f"  no normalizer for schema {schema!r}, skipping {key}")
        return [], {}

    rows = []
    dropped = {"malformed": 0, "length": 0, "similar": 0}
    for record in df.to_dict("records"):
        pair = fn(record, rng)
        if not pair:
            dropped["malformed"] += 1
            continue

        chosen, rejected = pair["chosen"], pair["rejected"]
        if not (MIN_REPLY_CHARS <= len(chosen) <= MAX_REPLY_CHARS) or \
           not (MIN_REPLY_CHARS <= len(rejected) <= MAX_REPLY_CHARS):
            dropped["length"] += 1
            continue
        if _too_similar(chosen, rejected):
            dropped["similar"] += 1
            continue

        rows.append({
            "source": key,
            "task": task,
            "prompt": json.dumps(pair["prompt"]),
            "chosen": chosen,
            "rejected": rejected,
        })
    return rows, dropped


def build_dpo_cache(force=False, val_per_task=VAL_PER_TASK):
    """Normalize every downloaded source into two parquet files."""
    if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE) and not force:
        return TRAIN_CACHE, VAL_CACHE

    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found -- run dataset_dpo_download.py first")

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    rng = random.Random(SEED)

    train_rows, val_rows = [], []

    for key, entry in manifest.items():
        task = entry.get("task", "helpful")
        schema = entry.get("schema")
        print(f"Normalizing {key} ({task}, schema={schema})...")
        train_df, val_df = _read_source(entry)

        if train_df is not None:
            rows, dropped = _normalize_frame(key, task, schema, train_df, rng)
            print(f"  train: {len(train_df):,} rows -> {len(rows):,} pairs "
                  f"(dropped {dropped})")
            train_rows.extend(rows)

        if val_df is not None:
            rows, dropped = _normalize_frame(key, task, schema, val_df, rng)
            print(f"  val:   {len(val_df):,} rows -> {len(rows):,} pairs")
            val_rows.extend(rows)

    if not train_rows:
        raise RuntimeError(
            "No usable preference pairs. Check that dataset_dpo_download.py "
            "actually wrote files and that each manifest entry has a `schema`.")

    # Sources without their own held-out files donate from train.
    have_val_tasks = {r["task"] for r in val_rows}
    rng.shuffle(train_rows)
    by_task = {}
    for r in train_rows:
        by_task.setdefault(r["task"], []).append(r)
    train_rows = []
    for task, rows in by_task.items():
        need = 0 if task in have_val_tasks else min(val_per_task, int(0.1 * len(rows)))
        val_rows.extend(rows[:need])
        train_rows.extend(rows[need:])

    # Trim oversized validation splits.
    trimmed, counts = [], {}
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

    print(f"\nWrote {len(train_rows):,} train / {len(val_rows):,} val pairs")
    for task in sorted({r["task"] for r in train_rows}):
        n = sum(1 for r in train_rows if r["task"] == task)
        print(f"  {task:12s} {n:>8,} available")

    return TRAIN_CACHE, VAL_CACHE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """Map-style dataset over normalized pairs, returning batches.

    Same contract as SFTDataset and PretrainTextDataset: __getitem__(i) returns
    a *list* of examples, so the DataLoader runs with batch_size=None.
    """

    def __init__(self, batch_size=8, split="train", task_weights=None,
                 total_examples=None, seed=SEED, epochs=1):
        self.batch_size = batch_size
        self.split = split

        path = TRAIN_CACHE if split == "train" else VAL_CACHE
        if not os.path.exists(path):
            build_dpo_cache()
        self.df = pd.read_parquet(path)
        if len(self.df) == 0:
            raise RuntimeError(f"{path} is empty")

        self.tasks = list(self.df["task"].unique())
        self._by_task = {t: self.df.index[self.df["task"] == t].to_numpy()
                         for t in self.tasks}

        if split == "train":
            self.index = self._build_mixture(task_weights or TASK_WEIGHTS,
                                             total_examples, seed, epochs)
        else:
            self.index = self.df.index.to_numpy()

        self._num_batches = math.ceil(len(self.index) / batch_size)

    def _build_mixture(self, weights, total_examples, seed, epochs):
        rng = random.Random(seed)
        present = {t: w for t, w in weights.items() if t in self._by_task}
        if not present:
            raise RuntimeError(
                f"No overlap between TASK_WEIGHTS and data tasks {self.tasks}")

        self.missing_tasks = [t for t in weights if t not in self._by_task]
        if self.missing_tasks:
            print(f"WARNING: no data for tasks {self.missing_tasks} -- their "
                  f"weight is redistributed across {list(present)}")

        norm = sum(present.values())
        if total_examples is None:
            total_examples = int(min(
                len(self._by_task[t]) * MAX_TASK_REPEATS / (w / norm)
                for t, w in present.items()))

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
        lines = [f"{self.split}: {len(self.index):,} pairs in "
                 f"{self._num_batches:,} batches of {self.batch_size}"]
        for task in getattr(self, "missing_tasks", []):
            lines.append(f"  {task:12s} NO DATA -- weight redistributed")
        for task, info in getattr(self, "mixture", {}).items():
            lines.append(f"  {task:12s} used {info['used']:>7,} of "
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
        out = []
        for row_id in self.index[start:start + self.batch_size]:
            record = self.df.loc[row_id]
            out.append({
                "prompt": json.loads(record["prompt"]),
                "chosen": record["chosen"],
                "rejected": record["rejected"],
                "task": record["task"],
                "source": record["source"],
            })
        return out


if __name__ == "__main__":
    build_dpo_cache(force=True)
    ds = DPODataset(batch_size=8)
    print(ds.describe())
    batch = ds[0]
    print(f"\nfirst batch: {len(batch)} pairs")
    for ex in batch[:2]:
        print(f"  [{ex['task']}/{ex['source']}] "
              f"{ex['prompt'][-1]['content'][:80]!r}")
        print(f"      chosen:   {ex['chosen'][:70]!r}")
        print(f"      rejected: {ex['rejected'][:70]!r}")
