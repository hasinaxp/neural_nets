import json
import math
import os
import random
import re

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .sources import strip_foreign_scripts

SFT_DATASET_FOLDER = "dataset/sft"
MANIFEST_FILE = f"{SFT_DATASET_FOLDER}/manifest.json"
CACHE_FOLDER = f"{SFT_DATASET_FOLDER}/cache"
TRAIN_CACHE = f"{CACHE_FOLDER}/sft_train.parquet"
VAL_CACHE = f"{CACHE_FOLDER}/sft_val.parquet"

IGNORE_INDEX = -100
SEED = 1337

# Share of the training mixture per task. Raw row counts are wildly unbalanced
# (SQL alone is ~185k rows), so the index is resampled to hit these instead.
#
# SQL is down from 0.16 to 0.10: it is the narrowest, most formulaic task here
# and the least useful to a general-purpose assistant per token spent. That
# 0.06 goes to chat, which is what a general model is actually judged on and
# which has by far the deepest pool.
TASK_WEIGHTS = {
    "chat": 0.32,
    "extractive_qa": 0.16,
    "summarization": 0.12,
    "math": 0.12,
    "sql": 0.10,
    "reasoning": 0.10,
    "instruct": 0.08,
}

# Examples per SFT epoch. Set explicitly rather than derived, because every
# derivation is wrong in one direction or the other: min(pool * cap / weight)
# over tasks hands the size of the run to the scarcest source (gsm8k's 7.5k
# rows capped it at 224k and starved chat), while sizing by a larger quantile
# runs for 50k steps and squeezes the small tasks down to ~2% share.
#
# 200k examples is ~6.2k optimizer steps at the default 8x4 batch, and at this
# size every task in TASK_WEIGHTS fits under its repeat cap, so the delivered
# mixture matches the target shares exactly. Raise it and the small tasks cap
# out first; describe() prints the drift when that happens.
DEFAULT_EPOCH_EXAMPLES = 200_000

VAL_PER_TASK = 300          # held-out examples per (task, source)
MAX_TASK_REPEATS = 3        # default cap on oversampling within one epoch
# Per-task overrides. Small pools that would otherwise be repeated hard are
# held down; deep, diverse pools are allowed a little more headroom.
TASK_REPEAT_CAPS = {
    "reasoning": 2,         # ~13k rows of short MCQ; memorised fast
    "instruct": 2,          # dolly is 15k rows of human prose
}
MAX_UNANSWERABLE_FRAC = 0.25   # cap on SQuAD-v2 "no answer" examples

# Examples whose text is largely non-Latin are dropped rather than run through
# strip_foreign_scripts(), which would silently hand the model a mangled
# sentence with the content characters removed. English-only corpus, so this
# should fire rarely -- it is a guard, not a filter.
MIN_LATIN_FRAC = 0.85

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
# Three phrasings put ~4% of the whole mixture on three literal strings, which
# the model then emits verbatim in unrelated contexts. More phrasings spread
# that mass over the *behaviour* (abstaining) instead of the exact sentence.
NO_ANSWER_REPLIES = [
    "The passage doesn't say.",
    "I can't answer that from the given passage.",
    "That information isn't in the context.",
    "The passage doesn't mention that.",
    "There's nothing in the text about that.",
    "I don't know -- the passage doesn't cover it.",
    "That isn't stated in the passage.",
    "The text given doesn't answer that.",
    "I can't tell from what's provided here.",
    "Not something the passage says.",
]
MATH_PROMPTS = [
    "Solve the problem below. Show your reasoning, then give the final answer.",
    "Work through this math problem step by step.",
    "Solve this word problem, explaining your steps as you go.",
]
REASONING_PROMPTS = [
    "Answer the question by choosing the correct option below.",
    "Choose the option that best answers the question.",
    "Pick the correct choice for the question below.",
]


# ---------------------------------------------------------------------------
# Per-source normalizers: dataframe row -> [{"role": ..., "content": ...}, ...]
# Return None to drop the row.
# ---------------------------------------------------------------------------

def _clean(text):
    if not isinstance(text, str):
        return ""
    # Drop letters from non-Latin scripts but keep punctuation, symbols and
    # accents -- curly quotes, em-dashes and the like carry meaning and were
    # being silently deleted by the old encode("ascii") pass.
    return strip_foreign_scripts(text).strip()


def _mostly_latin(text):
    """False when stripping non-Latin letters would gut the string.

    _clean() removes those characters in place, which turns a Chinese sentence
    into its punctuation. Dropping the example is the honest outcome; a mangled
    target is worse than no target.
    """
    if not isinstance(text, str) or not text:
        return True
    if text.isascii():
        return True
    kept = len(strip_foreign_scripts(text))
    return kept >= MIN_LATIN_FRAC * len(text)


def _as_list(seq):
    """List-ify a sequence that may be a numpy array (no truthiness) or None."""
    if seq is None:
        return []
    return list(seq)


def _from_messages(row, rng):
    """smoltalk / no_robots: already a list of {role, content}."""
    msgs = row.get("messages")
    if msgs is None or len(msgs) == 0:
        return None
    out = []
    system = ""
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = _clean(m.get("content") or m.get("value"))
        if role == "system":
            # There is no system-role token in the tokenizer, but dropping the
            # message leaves the reply conditioned on an instruction the model
            # can no longer see -- which is how you train confident
            # non-sequiturs. Fold it into the first user turn instead, the same
            # way the DPO orca normalizer already does.
            if content:
                system = content
            continue
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    if len(out) < 2 or out[0]["role"] != "user":
        return None
    if system:
        out[0] = {"role": "user",
                  "content": system + "\n\n" + out[0]["content"]}
    if not all(_mostly_latin(m["content"]) for m in out):
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


# Complexity tiers a model this small cannot plausibly learn from a few tens
# of thousands of examples. There used to be a SQL_COMPLEXITY_KEEP tuple beside
# this one that listed three of these same strings as *keep*; it was never
# referenced, so behaviour was always what DROP says, but it described the
# opposite of the truth to anyone reading.
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


_GSM8K_CALC = re.compile(r"<<[^>]*>>")   # calculator annotations, e.g. <<48/2=24>>


def _from_gsm8k(row, rng):
    """gsm8k: `answer` is a worked solution ending in '#### <final number>'."""
    question = _clean(row.get("question"))
    raw = row.get("answer")
    if not question or not isinstance(raw, str) or "####" not in raw:
        return None
    steps, _, final = raw.partition("####")
    steps = _clean(_GSM8K_CALC.sub("", steps))
    final = _clean(final)
    if not steps or not final:
        return None
    prompt = f"{rng.choice(MATH_PROMPTS)}\n\n{question}"
    reply = f"{steps}\nThe answer is {final}."
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _from_metamath(row, rng):
    """meta-math/MetaMathQA: `query` + `response`, response ends with
    'The answer is: X'. Kept as-is -- it is already a worked CoT solution in
    the same shape gsm8k is normalised into."""
    question = _clean(row.get("query"))
    reply = _clean(row.get("response"))
    if not question or not reply:
        return None
    # A handful of rows carry the MATH-style \boxed{} wrapper; unwrap it so the
    # model is not taught to emit LaTeX control sequences it never saw in
    # pretraining.
    reply = _BOXED.sub(r"\1", reply)
    if len(reply) < 16:
        return None
    prompt = f"{rng.choice(MATH_PROMPTS)}\n\n{question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


def _from_orca_math(row, rng):
    """microsoft/orca-math-word-problems-200k: `question` + `answer`, where the
    answer is already a step-by-step solution."""
    question = _clean(row.get("question"))
    reply = _clean(row.get("answer"))
    if not question or len(reply) < 16:
        return None
    reply = _BOXED.sub(r"\1", reply)
    prompt = f"{rng.choice(MATH_PROMPTS)}\n\n{question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


def _from_mcq_reasoning(row, rng):
    """ai2_arc / commonsense_qa: multiple-choice, `choices` is {text, label},
    `answerKey` names the correct label. No rationale is provided upstream, so
    the reply is just the chosen option -- the prompt only asks for that."""
    question = _clean(row.get("question"))
    choices = row.get("choices")
    if not isinstance(choices, dict):
        return None
    # Parquet -> pandas hands these back as numpy arrays, which have no usable
    # truth value, so length-check instead of leaning on `or []`.
    texts = _as_list(choices.get("text"))
    labels = _as_list(choices.get("label"))
    answer_key = row.get("answerKey")
    if not question or not answer_key or len(texts) != len(labels) or len(texts) < 2:
        return None
    try:
        idx = labels.index(answer_key)
    except ValueError:
        return None
    answer_text = _clean(texts[idx])
    if not answer_text:
        return None
    options = "\n".join(f"{lab}) {_clean(t)}" for lab, t in zip(labels, texts) if _clean(t))
    if not options:
        return None
    prompt = f"{rng.choice(REASONING_PROMPTS)}\n\n{question}\n\n{options}"
    reply = f"{answer_key}) {answer_text}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


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
    "gsm8k": _from_gsm8k,
    "metamath": _from_metamath,
    "orca-math": _from_orca_math,
    "arc-challenge": _from_mcq_reasoning,
    "arc-easy": _from_mcq_reasoning,
    "commonsense-qa": _from_mcq_reasoning,
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

    Overlong conversations are salvaged in two stages before being dropped:
    oldest turns go first, then the middle of the remaining user turn. The
    previous version could only trim the *first* user message, so any long
    multi-turn chat -- where the first turn is short -- was discarded outright,
    which quietly took a bite out of the chat and summarization tasks that the
    task weights had no way to see.
    """
    bos = tokenizer.special_tokens["<|BOS|>"]
    eos = tokenizer.special_tokens["<|EOS|>"]
    user = tokenizer.special_tokens["<|USER|>"]
    assistant = tokenizer.special_tokens["<|ASSISTANT|>"]

    # Encode each message once; the salvage paths below reassemble from these
    # rather than re-tokenising the whole conversation per attempt.
    segments = [(m["role"], tokenizer.encode(m["content"])) for m in messages]

    def assemble(segs):
        ids, mask = [bos], [0]
        for role, content in segs:
            if role == "user":
                ids.append(user)
                mask.append(0)
                ids.extend(content)
                mask.extend([0] * len(content))
            else:
                ids.append(assistant)
                mask.append(0)          # the role marker is part of the prompt
                ids.extend(content)
                mask.extend([1] * len(content))
                ids.append(eos)
                mask.append(1)          # learning to stop is the point
        return ids, mask

    ids, mask = assemble(segments)
    if len(ids) <= seq_len:
        return ids, mask
    if not truncate:
        return None, None

    # Stage 1: drop whole leading turns, oldest first, down to the final
    # exchange. The last turn is what the reply actually depends on.
    while len(segments) > 2 and len(ids) > seq_len:
        segments = segments[1:]
        while segments and segments[0][0] != "user":
            segments = segments[1:]
        if len(segments) < 2:
            return None, None
        ids, mask = assemble(segments)

    if len(ids) <= seq_len:
        return ids, mask

    # Stage 2: one exchange, still too long. Trim the user turn -- never the
    # reply, since a truncated target teaches truncated answers.
    #
    # Cut from the MIDDLE, keeping both ends. These prompts are laid out as
    # "<instruction>\n\nPassage:\n<body>\n\nQuestion: <q>", so the old
    # trim-from-the-end removed the question and left the model an instruction
    # and a passage with nothing to answer.
    if len(segments) != 2 or segments[0][0] != "user":
        return None, None
    u_role, u_ids = segments[0]
    overflow = len(ids) - seq_len
    budget = len(u_ids) - overflow
    if budget < 48:
        return None, None               # nothing meaningful left to ask
    head = min(64, budget // 4)         # the instruction paraphrase
    tail = budget - head                # the passage tail and the question
    segments[0] = (u_role, u_ids[:head] + u_ids[len(u_ids) - tail:])

    ids, mask = assemble(segments)
    if len(ids) > seq_len:
        return None, None
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

    # SQuAD-v2 is stored in article order, so a running "have I seen too many
    # unanswerables yet" check decides which ones survive by position in the
    # file rather than at random. Shuffling the row order first makes the
    # surviving quarter a random quarter.
    records = df.to_dict("records")
    if key == "squad-v2":
        rng.shuffle(records)

    rows = []
    unanswerable = 0
    kept = 0
    for record in records:
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
    #
    # Keyed on (task, source), not task. Keyed on task alone, a single source
    # shipping a published validation split suppressed the donation for every
    # other source in that task: no_robots has one, so the entire `chat`
    # holdout was no_robots and smoltalk -- 98% of the chat pool -- never
    # appeared in the curve the run is steered on.
    have_val = {(r["task"], r["source"]) for r in val_rows}
    rng.shuffle(train_rows)
    if train_rows:
        by_source = {}
        for r in train_rows:
            by_source.setdefault((r["task"], r["source"]), []).append(r)
        train_rows = []
        per_source_target = max(1, val_per_task // max(1, len(by_source)))
        for pair, rows in by_source.items():
            if pair in have_val:
                need = 0
            else:
                # Never donate more than 10% of a source -- otherwise a small
                # source gets swallowed by the holdout entirely.
                need = min(per_source_target, int(0.1 * len(rows)))
            val_rows.extend(rows[:need])
            train_rows.extend(rows[need:])

    # Trim oversized validation splits; we only need enough to track a curve.
    # Capped per source as well, so one large published validation split
    # cannot crowd its task-mates out of the holdout.
    val_sources = {(r["task"], r["source"]) for r in val_rows}
    per_task_source = max(1, val_per_task // max(
        1, len({t for t, _ in val_sources})))
    trimmed = []
    counts, source_counts = {}, {}
    rng.shuffle(val_rows)
    for r in val_rows:
        pair = (r["task"], r["source"])
        n_task = counts.get(r["task"], 0)
        n_source = source_counts.get(pair, 0)
        n_sources_in_task = max(1, sum(1 for t, _ in val_sources if t == r["task"]))
        source_cap = max(1, val_per_task // n_sources_in_task)
        if n_task >= val_per_task or n_source >= source_cap:
            train_rows.append(r)
            continue
        counts[r["task"]] = n_task + 1
        source_counts[pair] = n_source + 1
        trimmed.append(r)
    val_rows = trimmed

    rng.shuffle(train_rows)
    pd.DataFrame(train_rows).to_parquet(TRAIN_CACHE)
    pd.DataFrame(val_rows).to_parquet(VAL_CACHE)

    print(f"\nWrote {len(train_rows):,} train / {len(val_rows):,} val conversations")
    for task in sorted({r["task"] for r in train_rows}):
        n = sum(1 for r in train_rows if r["task"] == task)
        v = sum(1 for r in val_rows if r["task"] == task)
        sources = sorted({r["source"] for r in val_rows if r["task"] == task})
        print(f"  {task:15s} {n:>8,} train | {v:>4,} val "
              f"from {', '.join(sources) or '-'}")

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

        Sizing used to be ``min(pool * repeats / weight)`` over tasks, which
        hands the size of the whole run to the single scarcest source: gsm8k's
        7.5k rows at weight 0.10 capped the epoch at ~224k examples and left
        88% of the chat pool unused so gsm8k could be repeated three times.
        The epoch size is now set directly (DEFAULT_EPOCH_EXAMPLES) and each
        task is clipped at its own repeat cap, with any shortfall redistributed
        across the tasks that still have data. One small source can no longer
        shrink the run, no source is epoched harder than its cap allows, and
        describe() reports delivered share against target so the cases where
        the two disagree are visible rather than silent.
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
        caps = {t: len(self._by_task[t]) * TASK_REPEAT_CAPS.get(t, MAX_TASK_REPEATS)
                for t in present}

        if total_examples is None:
            total_examples = DEFAULT_EPOCH_EXAMPLES

        want = self._allocate(present, caps, total_examples, norm)

        index = []
        self.mixture = {}
        for task in present:
            n = want[task]
            pool = list(self._by_task[task])
            if n <= len(pool):
                picked = rng.sample(pool, n)
            else:
                reps = n // len(pool)
                picked = pool * reps + rng.sample(pool, n - reps * len(pool))
            self.mixture[task] = {
                "available": len(pool),
                "used": n,
                "repeats": round(n / max(1, len(pool)), 2),
                "target_share": round(present[task] / norm, 3),
                "actual_share": 0.0,      # filled in once the realized total is known
                "capped": n >= caps[task],
            }
            index.extend(picked)

        realized = max(1, sum(want.values()))
        for task in self.mixture:
            self.mixture[task]["actual_share"] = round(want[task] / realized, 3)

        # `epochs` here multiplies the index in place. The trainers pass 1 and
        # loop epochs in the batch stream instead, so this is a no-op on the
        # real path -- kept only so a caller building a dataset directly can
        # still ask for a multi-epoch index.
        index = index * max(1, epochs)
        rng.shuffle(index)
        return index

    @staticmethod
    def _allocate(weights, caps, total, norm):
        """Hand out ``total`` draws by weight, clipping each task at its cap.

        Water-filling: any task whose weighted share exceeds what its pool can
        supply is pinned at its cap and removed, and the remainder is shared
        out among the rest by weight. Repeats until nothing else overflows, so
        the weights are honoured exactly for every task that has the data.
        """
        alloc = {}
        live = dict(weights)
        remaining = total
        while live:
            live_norm = sum(live.values()) or 1.0
            overflowed = [t for t, w in live.items()
                          if remaining * w / live_norm > caps[t]]
            if not overflowed:
                for t, w in live.items():
                    alloc[t] = int(remaining * w / live_norm)
                break
            for t in overflowed:
                alloc[t] = int(caps[t])
                remaining -= alloc[t]
                del live[t]
            if remaining <= 0:
                for t in live:
                    alloc[t] = 0
                break
        return alloc

    def describe(self):
        lines = [f"{self.split}: {len(self.index):,} examples in "
                 f"{self._num_batches:,} batches of {self.batch_size}"]
        for task in getattr(self, "missing_tasks", []):
            lines.append(f"  {task:15s} NO DATA -- weight redistributed")
        for task, info in getattr(self, "mixture", {}).items():
            flag = "  CAPPED" if info.get("capped") else ""
            lines.append(f"  {task:15s} used {info['used']:>7,} of "
                         f"{info['available']:>7,} ({info['repeats']}x) "
                         f"| share {info.get('actual_share', 0):.3f} "
                         f"target {info.get('target_share', 0):.3f}{flag}")
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