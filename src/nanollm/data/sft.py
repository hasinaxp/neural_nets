import copy
import json
import math
import os
import random
import re
from collections import Counter

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
# Rebalanced for "most useful general assistant per token". Three tasks are new
# -- writing, rewrite, shell -- and the room for them comes from the two that
# were paying the least rent:
#
#   sql        0.10 -> 0.04   the narrowest, most formulaic task here. A model
#                             this size is not going to be anyone's text-to-SQL
#                             engine, and every point of weight spent on it was
#                             a point not spent on things people actually ask a
#                             small assistant to do.
#   chat       0.32 -> 0.24   still the largest single task, but a chunk of what
#                             made it large was no_robots Generation rows that
#                             are now correctly labelled `writing`. This is a
#                             relabelling as much as a cut.
#   instruct   0.08 -> 0.04   dolly's best rows (creative_writing, brainstorming)
#                             moved to `writing`; what is left is the residual.
#
# The three additions are the common asks this model could not previously do:
#   writing    drafting -- emails, posts, blurbs, brainstorms
#   rewrite    rephrase, simplify, fix grammar, change tone
#   shell      natural language -> a Linux command
TASK_WEIGHTS = {
    # A bare "hi" is the first thing anyone types and the mixture could not
    # answer it -- see the Small talk section for why 1,872 real greeting rows
    # amounted to ~0.04% of an epoch's gradient. 0.03 is deliberately small:
    # this is a handful of short behaviours that need to exist at all, not a
    # capability that rewards more weight, and the replies are short enough
    # that it costs almost nothing in tokens.
    "smalltalk": 0.03,
    "chat": 0.21,
    "extractive_qa": 0.12,
    "writing": 0.12,
    "rewrite": 0.10,
    "math": 0.10,
    "summarization": 0.09,
    "shell": 0.08,
    "reasoning": 0.07,
    "instruct": 0.04,
    "sql": 0.04,
}

# Examples per SFT epoch. Set explicitly rather than derived, because every
# derivation is wrong in one direction or the other: min(pool * cap / weight)
# over tasks hands the size of the run to the scarcest source (gsm8k's 7.5k
# rows capped it at 224k and starved chat), while sizing by a larger quantile
# runs for 50k steps and squeezes the small tasks down to ~2% share.
#
# 200k examples is ~6.2k optimizer steps at the default 8x4 batch, and at this
# size every task in TASK_WEIGHTS is delivered at exactly its target share.
# The binding task is now `writing` (~13k rows at a 2x repeat cap, so 26.6k
# available against 24k drawn); past ~220k it caps and its share starts to
# slip. describe() prints delivered vs target so that is visible rather than
# silent.
DEFAULT_EPOCH_EXAMPLES = 200_000

VAL_PER_TASK = 300          # held-out examples per (task, source)
MAX_TASK_REPEATS = 3        # default cap on oversampling within one epoch
# Per-task overrides. Small pools that would otherwise be repeated hard are
# held down; deep, diverse pools are allowed a little more headroom.
TASK_REPEAT_CAPS = {
    "smalltalk": 4,         # a few thousand short rows; the behaviour is tiny
                            # and repeating it is cheap, but past this the
                            # model starts greeting people who did not say hi
    "reasoning": 2,         # ~21k rows of short MCQ; memorised fast
    "instruct": 2,          # dolly is 15k rows of human prose
    "writing": 2,           # ~13k rows, and long free-form targets overfit
                            # faster than short ones -- the model starts
                            # reciting whole no_robots stories verbatim
    "shell": 2,             # ~45k rows, but the answers are one line each;
                            # repeating them mostly teaches memorised commands
}
MAX_UNANSWERABLE_FRAC = 0.25   # cap on SQuAD-v2 "no answer" examples

# Cap on how much of one task's replies may open with the same command.
#
# nl2bash and nl2sh are `find` corpora more than they are shell corpora: 43.1%
# of the 45,285 shell replies start with `find`, against 0.9% for `ls`, and only
# 0.46% contain an `ls -a`-style listing at all. A model trained on that answers
# "what lists all files including hidden ones?" with a find pipeline -- and
# sometimes ends it in `rm -f`, because that is what the neighbouring rows do.
# The task weight is then buying `find` flag trivia rather than the everyday
# commands the weight was allocated for.
#
# Down-sampling to this cap keeps every distinct `find` row eligible while
# giving the rest of the distribution room; the surplus is dropped rather than
# reweighted, since the alternative is oversampling ~600 `ls` rows to balance
# ~19.5k `find` ones.
TASK_HEAD_COMMAND_CAP = {"shell": 0.15}

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
# Thinking
# ---------------------------------------------------------------------------
# There is no <|THINK|> special token and there cannot be one: Tokenizer.
# SPECIAL_TOKENS is a fixed list of six, and the merge ids are laid out at
# `256 + len(SPECIAL_TOKENS)`. Adding a seventh shifts merge_id_offset and
# invalidates every merge id in artifacts/tokenizer-32768.txt -- i.e. it would
# invalidate the pretrained checkpoint. So the scratchpad is plain text, in a
# fixed two-line shape the model can learn and a caller can split on:
#
#     Thinking: <a few sentences of working>
#     Answer: <the final answer>
#
# Two rules keep this from becoming a tic:
#
# 1. It is only ever attached to rows that carry a REAL rationale upstream --
#    ecqa's human explanations and the worked solutions in gsm8k / metamath /
#    orca-math. Nothing is invented. A 169M model taught to emit plausible
#    reasoning it did not do is strictly worse than one that answers directly,
#    because the reasoning then reads as evidence and is not.
#
# 2. The same sources also produce direct-answer rows (THINK_FRACTION below),
#    paired with DIRECT_PROMPTS. Without that contrast the marker is not a
#    behaviour the user can ask for, it is just the house style for anything
#    that smells like a maths question -- and it leaks into everything else.
THINK_PREFIX = "Thinking:"
ANSWER_PREFIX = "Answer:"

# Share of think-capable rows rendered with the scratchpad. The rest are the
# same question answered flat, so both modes are addressable.
THINK_FRACTION = 0.75

THINK_PROMPTS = [
    "Think it through step by step, then give the answer.",
    "Work through this carefully before answering.",
    "Reason it out first, then answer.",
    "Explain your thinking, then state the answer.",
    "Take it step by step and finish with the answer.",
]
DIRECT_PROMPTS = [
    "Answer directly, without explanation.",
    "Give just the answer.",
    "Answer in one line, no working.",
    "Just the final answer, please.",
]


def _think_reply(rationale, answer):
    """Render the two-line scratchpad. Falls back to a bare answer when the
    upstream rationale is missing or degenerate -- see rule 1 above."""
    rationale = " ".join(str(rationale or "").split())
    answer = str(answer or "").strip()
    if not answer:
        return None
    if len(rationale) < 16 or rationale.lower() == answer.lower():
        return answer
    return f"{THINK_PREFIX} {rationale}\n{ANSWER_PREFIX} {answer}"


def split_thinking(text):
    """Inverse of _think_reply: (thinking, answer). `thinking` is "" when the
    reply has no scratchpad. Used by chat.py to fold the working away, and by
    eval code that wants to score only the answer."""
    if not isinstance(text, str) or THINK_PREFIX not in text:
        return "", (text or "").strip()
    _, _, rest = text.partition(THINK_PREFIX)
    thinking, sep, answer = rest.partition(ANSWER_PREFIX)
    if not sep:
        return thinking.strip(), ""
    return thinking.strip(), answer.strip()


# ---------------------------------------------------------------------------
# Rewriting / rephrasing
# ---------------------------------------------------------------------------
# CoEdIT ships its instruction glued to the front of `src` ("Fix grammar in
# this sentence: ..."), one fixed phrasing per edit type. Training on that
# verbatim teaches the string, not the edit, so the prefix is split off and
# replaced with one of ours, chosen by the row's declared edit type.
REWRITE_PROMPTS = {
    "gec": [
        "Fix the grammar in this text.",
        "Correct any grammatical errors below.",
        "Rewrite this with the grammar mistakes fixed.",
    ],
    "paraphrase": [
        "Rephrase the text below.",
        "Say this another way.",
        "Rewrite this in different words, keeping the meaning.",
        "Give me a paraphrase of this.",
    ],
    "simplification": [
        "Rewrite this in simpler language.",
        "Make this easier to read.",
        "Simplify the text below.",
    ],
    "coherence": [
        "Rewrite this so it reads more coherently.",
        "Make this flow better.",
        "Improve the coherence of the text below.",
    ],
    "neutralize": [
        "Rewrite this in a neutral tone.",
        "Remove the bias from this text.",
        "Make this sound more impartial.",
    ],
    "clarity": [
        "Rewrite this more clearly.",
        "Make this clearer and more direct.",
        "Improve the clarity of the text below.",
    ],
}
REWRITE_FALLBACK_PROMPTS = [
    "Rewrite the text below.",
    "Improve this text.",
]

# ---------------------------------------------------------------------------
# Linux shell
# ---------------------------------------------------------------------------
# Replies are the bare command and nothing else, because that is all the
# upstream data contains. Writing an explanation around it would mean
# generating the explanation here, and a synthetic gloss on a command is
# exactly the kind of confident-sounding filler this model should not be
# taught to produce.
SHELL_PROMPTS = [
    "Write a Linux command for this.",
    "What shell command does this?",
    "Give me the bash command for the following.",
    "How do I do this from the Linux terminal?",
    "Write a bash one-liner for this task.",
]

# Interactive shells are not sandboxes, and a one-line answer with no caveat is
# how a small model gets someone to paste `rm -rf /` into a terminal. Rows whose
# command matches any of these are dropped rather than taught.
#
# Matched with \b rather than anchored to a command separator, so a `sudo`,
# `doas` or `xargs` prefix cannot walk a destructive command past the filter.
# The cost is the occasional false positive on a command that merely mentions
# one of these strings, which is a trade worth making in this direction.
SHELL_DROP = re.compile("|".join([
    # rm -rf against / itself, a root-level glob, or a system directory.
    # `rm -rf ./build` and `rm -f notes.txt` are ordinary and must survive.
    r"\brm\s+(?:-[-\w]+\s+)*/(?:\s|$|\*|(?:bin|boot|dev|etc|home|lib|proc"
    r"|root|sbin|srv|sys|usr|var)\b)",
    r"\bmkfs(?:\.\w+)?\b",
    # Only block devices. `dd if=image.iso of=/dev/null` is a throughput test.
    r"\bdd\s+[^;&|]*of=/dev/(?:sd|nvme|hd|vd|disk|mmcblk|loop)",
    r":\(\)\s*\{",                       # fork bomb
    # `halt` only in command position: \bhalt\b also matches the -halt-on-error
    # flag that every LaTeX invocation in these corpora carries.
    r"\b(?:shutdown|reboot|poweroff)\b",
    r"(?:^|[;&|]\s*|\bsudo\s+)halt\b",
    r"\bchmod\s+(?:-[-\w]+\s+)*777\s+/(?:\s|$)",
    r">\s*/dev/(?:sd|nvme|hd|vd|mmcblk)",
    r"\bmv\s+[^;&|]*\s/dev/null\b",
]), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Writing / drafting
# ---------------------------------------------------------------------------
# no_robots Generation and Brainstorm rows already read as natural instructions
# ("Write me a short poem about..."), so these are only used for dolly, whose
# creative_writing / brainstorming instructions are sometimes bare topics.
WRITING_PROMPTS = [
    "Write a draft for the following.",
    "Draft this for me.",
    "Write something for this.",
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


# no_robots is one repo feeding three tasks. The category column is the whole
# reason to bother: 8,692 of its 19,000 rows are `Generation` -- human-written
# drafting examples -- and lumping them into `chat` meant the amount of
# drafting data in the run was set by how much small talk we wanted.
NO_ROBOTS_TASKS = {
    "Generation": "writing",
    "Brainstorm": "writing",
    "Rewrite": "rewrite",
    "Summarize": "summarization",
}


def _from_no_robots(row, rng):
    """no_robots: same message shape as smoltalk, but routed by category."""
    messages = _from_messages(row, rng)
    if not messages:
        return None
    task = NO_ROBOTS_TASKS.get(str(row.get("category") or ""))
    return (messages, task) if task else messages


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
# Just the marker and its value -- NOT to end of string. metamath often
# writes "... #### 140 The answer is: 140", so a greedy strip here removes
# the very sentence _split_worked_solution() keys on and silently drops the
# think rate from ~70% to ~25%.
_GSM8K_FINAL = re.compile(r"####\s*\S+")   # "#### 140" answer marker

# "The answer is: 42", "the answer is 42." -- how metamath and orca-math close
# a worked solution. Anchored to the end so a mid-solution restatement is not
# mistaken for the conclusion.
_FINAL_ANSWER = re.compile(
    r"(?:^|\n|\.\s+)(?:so\s+|therefore,?\s+)?the\s+answer\s+is:?\s*"
    r"(.+?)\s*\.?\s*$", re.IGNORECASE | re.DOTALL)


def _split_worked_solution(text):
    """(steps, final answer) from a solution that states its answer at the end.

    Returns (None, None) when the closing statement is not found, which is the
    signal to leave the row in its original free-form shape rather than force
    it into the scratchpad format around a guessed answer.
    """
    text = (text or "").strip()
    if not text:
        return None, None
    match = _FINAL_ANSWER.search(text)
    if not match:
        return None, None
    final = " ".join(match.group(1).split())
    steps = text[:match.start()].strip()
    # A "solution" that is only its own conclusion has no working to show.
    if not final or len(final) > 120 or len(steps) < 16:
        return None, None
    return steps, final


def _math_example(question, steps, final, fallback_reply, rng):
    """Render a maths row, with or without the scratchpad.

    `fallback_reply` is used when the solution could not be split -- the row is
    still worth training on, it just cannot carry an Answer: line.
    """
    question = _clean(question)
    if not question:
        return None

    if steps and final and rng.random() < THINK_FRACTION:
        reply = _think_reply(steps, final)
        prompt_pool = THINK_PROMPTS
    elif steps and final:
        reply = final                      # the direct-answer contrast
        prompt_pool = DIRECT_PROMPTS
    else:
        reply = _clean(fallback_reply)
        prompt_pool = MATH_PROMPTS
    if not reply:
        return None

    prompt = f"{rng.choice(prompt_pool)}\n\n{question}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


def _from_gsm8k(row, rng):
    """gsm8k: `answer` is a worked solution ending in '#### <final number>'."""
    question = row.get("question")
    raw = row.get("answer")
    if not isinstance(raw, str) or "####" not in raw:
        return None
    steps, _, final = raw.partition("####")
    steps = _clean(_GSM8K_CALC.sub("", steps))
    final = _clean(final)
    if not steps or not final:
        return None
    return _math_example(question, steps, final, None, rng)


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


# MetaMathQA is two corpora under one name, and its `type` column separates
# them cleanly:
#
#   GSM_*   54,869 rows, grade-school word problems, 0.0% LaTeX
#   MATH_*  35,131 rows, competition algebra/geometry, 28-65% LaTeX
#
# Only the GSM half is kept. Two reasons, and they point the same way. A 169M
# model is not going to solve competition geometry no matter how many examples
# it sees, so those rows buy nothing but a confident wrong answer in a format
# that looks authoritative. And they are where essentially all of metamath's
# LaTeX lives -- \frac, \sqrt, \cdot -- which pretraining barely covered and
# which _from_metamath already goes out of its way to unwrap from \boxed{}.
# Filtering here rather than at download keeps the choice next to its reason.
METAMATH_KEEP_PREFIX = "GSM"


def _from_metamath(row, rng):
    """meta-math/MetaMathQA: `query` + `response`, response ends with
    'The answer is: X' -- already a worked CoT solution in the same shape
    gsm8k is normalised into, so _math_example can split it."""
    if not str(row.get("type") or "").startswith(METAMATH_KEEP_PREFIX):
        return None
    question = _clean(row.get("query"))
    reply = _clean(row.get("response"))
    if not question or not reply:
        return None
    # MetaMathQA is built on GSM8K and carries its markup through: calculator
    # annotations and a trailing "#### <answer>". Left in, both end up inside
    # the Thinking: block, teaching the model to emit a format marker it is
    # never asked for and that nothing downstream parses.
    reply = _GSM8K_CALC.sub("", reply)
    reply = _GSM8K_FINAL.sub("", reply).strip()
    if not reply:
        return None
    # A handful of rows carry the MATH-style \boxed{} wrapper; unwrap it so the
    # model is not taught to emit LaTeX control sequences it never saw in
    # pretraining.
    reply = _BOXED.sub(r"\1", reply)
    if len(reply) < 16:
        return None
    steps, final = _split_worked_solution(reply)
    return _math_example(question, steps, final, reply, rng)


def _from_orca_math(row, rng):
    """microsoft/orca-math-word-problems-200k: `question` + `answer`, where the
    answer is already a step-by-step solution."""
    question = _clean(row.get("question"))
    reply = _clean(row.get("answer"))
    if not question or len(reply) < 16:
        return None
    reply = _BOXED.sub(r"\1", reply)
    steps, final = _split_worked_solution(reply)
    return _math_example(question, steps, final, reply, rng)


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

# Same idea as NO_ROBOTS_TASKS: dolly's categories already say what each row
# teaches, so route on them instead of filing all 15k rows under `instruct`.
DOLLY_TASKS = {
    "creative_writing": "writing",
    "brainstorming": "writing",
    "summarization": "summarization",
}


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
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    task = DOLLY_TASKS.get(category)
    return (messages, task) if task else messages


def _from_coedit(row, rng):
    """grammarly/coedit: `src` is "<instruction>: <text>", `tgt` is the edit."""
    src = _clean(row.get("src"))
    tgt = _clean(row.get("tgt"))
    if not src or not tgt:
        return None

    # Split the glued-on instruction off the front. Only the first ": " counts,
    # and only when what precedes it is short enough to be an instruction
    # rather than a colon inside the text itself.
    head, sep, body = src.partition(": ")
    if sep and len(head) <= 80:
        text = body.strip()
    else:
        text = src
    if not text:
        return None

    edit = str(row.get("task") or "").strip().lower()
    prompts = REWRITE_PROMPTS.get(edit, REWRITE_FALLBACK_PROMPTS)
    prompt = f"{rng.choice(prompts)}\n\n{text}"
    if not _mostly_latin(text) or not _mostly_latin(tgt):
        return None
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": tgt}]


def _shell_example(instruction, command, rng):
    """Shared renderer for the two natural-language -> bash sources."""
    instruction = _clean(instruction)
    command = _clean(command)
    if not instruction or not command:
        return None
    # Multi-line scripts are out of scope; these sources are one-liners and the
    # few multi-line rows are usually a parse artifact.
    command = command.strip()
    if "\n" in command or len(command) > 400:
        return None
    if SHELL_DROP.search(command):
        return None
    prompt = f"{rng.choice(SHELL_PROMPTS)}\n\n{instruction}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": command}]


def _from_nl2bash(row, rng):
    return _shell_example(row.get("nl_command"), row.get("bash_code"), rng)


def _from_nl2sh(row, rng):
    return _shell_example(row.get("nl"), row.get("bash"), rng)


_ECQA_OPTIONS = ("q_op1", "q_op2", "q_op3", "q_op4", "q_op5")


def _from_ecqa(row, rng):
    """yangdong/ecqa: CommonsenseQA plus a human-written explanation.

    This is the only reasoning source here with a real rationale, which is what
    makes a Thinking: block honest on this task. `taskA_pos` explains why the
    correct option is correct; `taskB` also argues against the distractors and
    runs long, so the shorter one is used.
    """
    question = _clean(row.get("q_text"))
    answer = _clean(row.get("q_ans"))
    if not question or not answer:
        return None

    options = [_clean(row.get(k)) for k in _ECQA_OPTIONS]
    options = [o for o in options if o]
    if len(options) < 2 or answer not in options:
        return None

    labels = [chr(ord("A") + i) for i in range(len(options))]
    rendered = "\n".join(f"{lab}) {opt}" for lab, opt in zip(labels, options))
    label = labels[options.index(answer)]
    final = f"{label}) {answer}"

    if rng.random() < THINK_FRACTION:
        reply = _think_reply(row.get("taskA_pos"), final)
        prompt_pool = THINK_PROMPTS
    else:
        reply = final
        prompt_pool = DIRECT_PROMPTS
    if not reply:
        return None

    prompt = f"{rng.choice(prompt_pool)}\n\n{question}\n\n{rendered}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": reply}]


NORMALIZERS = {
    "smoltalk": _from_messages,
    "no-robots": _from_no_robots,
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
    "ecqa": _from_ecqa,
    "coedit": _from_coedit,
    "nl2bash": _from_nl2bash,
    "nl2sh": _from_nl2sh,
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
    routed = {}
    for record in records:
        result = fn(record, rng)
        if not result:
            continue
        # A normalizer may return (messages, task) to override the repo-level
        # task for that row -- no_robots and dolly both carry a category column
        # that says what the row actually teaches. Anything else is the plain
        # message list and keeps the manifest's task.
        if isinstance(result, tuple):
            messages, row_task = result
            row_task = row_task or task
        else:
            messages, row_task = result, task
        if not messages:
            continue
        if key == "squad-v2":
            is_no_answer = messages[-1]["content"] in NO_ANSWER_REPLIES
            if is_no_answer:
                if unanswerable > MAX_UNANSWERABLE_FRAC * max(1, kept):
                    continue
                unanswerable += 1
        kept += 1
        routed[row_task] = routed.get(row_task, 0) + 1
        rows.append({
            "source": key,
            "task": row_task,
            "messages": json.dumps(messages),
        })
    if len(routed) > 1:
        breakdown = ", ".join(f"{t}={n:,}" for t, n in sorted(routed.items()))
        print(f"  routed by category: {breakdown}")
    return rows


# ---------------------------------------------------------------------------
# Small talk
# ---------------------------------------------------------------------------
# A bare "hi" is the first thing anyone types at a chat model and the mixture
# had no answer for it. The greeting rows that exist are real but invisible:
# 1,872 smoltalk conversations open with one, every one of them answered with
# the same sentence, and each is the first exchange of a 6-8 turn conversation
# rendered as a single training example -- so the eight tokens of "Hello! How
# can I help you today?" are ~1% of that example's supervised loss, and the
# whole behaviour is ~0.04% of an epoch's gradient. A model trained on that
# answers "hi" with an essay about whatever the rest of the conversation was
# about, which is exactly what it did.
#
# So: harvest the real openers (see harvest_smalltalk) and pair them with the
# turns the corpus has none of -- thanks, goodbye, how-are-you, and the
# identity and capability questions that otherwise get answered from whatever
# first-person narration the corpus happens to contain ("I am Elena, a young
# woman from a small town in the Czech Republic").
#
# Three rules, the same ones the rest of this file follows:
#
# 1. Replies are SHORT and every one ends the turn. This is the one task where
#    the correct answer is a sentence, which makes it the cheapest possible
#    training signal for stopping.
# 2. Many phrasings on both sides, paired at random. One fixed reply teaches
#    the string; a pool teaches the behaviour.
# 3. The identity and capability answers claim nothing the model cannot back
#    up -- no name, no feelings, no memory between chats, no web access, and a
#    capability list that is literally the task list it was trained on. These
#    are the lines to edit if this model is deployed as something with a name.
SMALLTALK_OPENERS = [
    "hi", "Hi", "hi!", "hello", "Hello", "hello!", "Hello!", "hey", "Hey",
    "hey!", "hey there", "Hi there", "hiya", "howdy", "yo", "greetings",
    "good morning", "Good morning", "good afternoon", "good evening",
    "morning", "hi, how are you?", "hello, are you there?", "anyone there?",
    "hey, got a minute?", "hi again", "hello?", "Hi!",
]
SMALLTALK_OPENER_REPLIES = [
    "Hello! How can I help you today?",
    "Hi! What can I do for you?",
    "Hello! What would you like help with?",
    "Hi there! What are you working on?",
    "Hey! What can I help you with?",
    "Hello! Ask me anything you like.",
    "Hi! How can I help?",
    "Hello! What's on your mind?",
    "Hi there! What do you need?",
    "Hello! Happy to help -- what do you need?",
]
SMALLTALK_HOW_ARE_YOU = [
    "how are you?", "How are you?", "how are you doing?", "how's it going?",
    "how have you been?", "you doing ok?", "how are things?", "what's up?",
    "what's new?", "how's your day going?",
]
SMALLTALK_HOW_ARE_YOU_REPLIES = [
    "I'm a program, so I don't have good days or bad ones -- but I'm ready to help. What do you need?",
    "I don't have feelings to report, but everything's working. What can I do for you?",
    "Doing fine, as much as software can. What are you working on?",
    "I'm here and ready to go. What would you like to do?",
    "No news on my end -- I don't carry anything over between chats. What can I help with?",
    "All systems working. What can I help you with?",
]
SMALLTALK_THANKS = [
    "thanks", "thanks!", "Thanks", "thank you", "thank you!", "thanks a lot",
    "thanks so much", "cheers", "ty", "appreciate it", "that helps, thanks",
    "perfect, thanks", "great, thank you",
]
SMALLTALK_THANKS_REPLIES = [
    "You're welcome! Anything else?",
    "Happy to help. Let me know if you need anything else.",
    "Any time. Anything else you'd like to look at?",
    "You're welcome.",
    "Glad it helped!",
    "No problem -- just ask if something else comes up.",
]
SMALLTALK_BYE = [
    "bye", "goodbye", "bye!", "see ya", "see you later", "that's all",
    "that's all for now", "nothing else", "I'm done", "no thanks, that's it",
    "gotta go", "later!",
]
SMALLTALK_BYE_REPLIES = [
    "Goodbye! Come back any time.",
    "See you later!",
    "Bye -- good luck with it.",
    "Take care!",
    "Sounds good. See you next time.",
    "Bye for now.",
]
SMALLTALK_IDENTITY = [
    "who are you?", "what are you?", "what's your name?", "do you have a name?",
    "are you a human?", "are you a robot?", "are you an AI?", "are you real?",
    "am I talking to a person?", "tell me about yourself",
    "what kind of model are you?", "are you ChatGPT?",
]
SMALLTALK_IDENTITY_REPLIES = [
    "I'm a small language model -- a program that answers questions in text. I don't have a name.",
    "I'm an AI assistant, not a person. I answer in text and that's all I can do.",
    "Not a human -- I'm a language model. I generate text, one word at a time.",
    "I'm a small AI text assistant. No name, no body, and no memory of past conversations.",
    "I'm a computer program trained to answer questions and write text. I'm not a person.",
    "I'm a language model. I'm quite a small one, so I get things wrong sometimes -- worth checking anything important.",
]
SMALLTALK_CAPABILITY = [
    "what can you do?", "what are you good at?", "how can you help me?",
    "what can I ask you?", "can you help me?", "what do you do?",
    "what should I ask you?", "are you any good?",
]
SMALLTALK_CAPABILITY_REPLIES = [
    "I can answer questions, summarize text, rewrite or fix writing, draft short "
    "pieces, work through maths problems, and write shell commands or SQL queries. "
    "What do you need?",
    "Ask me to summarize something, answer a question about a passage, tidy up some "
    "writing, draft an email, solve a maths problem, or write a shell command.",
    "Summarizing, answering questions, rewriting text, drafting, simple maths, and "
    "shell or SQL one-liners. What are you working on?",
    "Mostly text: questions and answers, summaries, rewriting, drafting, maths, and "
    "commands. I'm small, so I'm better at short, concrete tasks than long ones.",
    "I can help with writing and rewriting, summaries, questions about a passage, "
    "maths problems, and shell or SQL commands. What would you like to start with?",
]

# (prompts, replies) pools making up the task.
SMALLTALK_GROUPS = [
    (SMALLTALK_OPENERS, SMALLTALK_OPENER_REPLIES),
    (SMALLTALK_HOW_ARE_YOU, SMALLTALK_HOW_ARE_YOU_REPLIES),
    (SMALLTALK_THANKS, SMALLTALK_THANKS_REPLIES),
    (SMALLTALK_BYE, SMALLTALK_BYE_REPLIES),
    (SMALLTALK_IDENTITY, SMALLTALK_IDENTITY_REPLIES),
    (SMALLTALK_CAPABILITY, SMALLTALK_CAPABILITY_REPLIES),
]

# Openers the corpus already answers correctly, lifted out of the multi-turn
# conversations that bury them. Matched against the FIRST user turn only.
SMALLTALK_HARVEST = re.compile(
    r"^(hi|hello|hey|yo|howdy|greetings|good (morning|afternoon|evening|day)|"
    r"how are you( doing)?|how'?s it going)\b[\s!.,?]*$", re.IGNORECASE)
SMALLTALK_HARVEST_MAX_WORDS = 45     # a greeting answer is a sentence, not an essay


def harvest_smalltalk(rows):
    """Pull (greeting, short reply) first exchanges out of multi-turn rows.

    Real data, not authored: the corpus answers these correctly, it just does
    it inside conversations long enough that the behaviour never gets any
    gradient. Returned as standalone two-message conversations.
    """
    out = []
    for row in rows:
        try:
            messages = json.loads(row["messages"])
        except (TypeError, ValueError):
            continue
        if len(messages) < 2 or messages[0]["role"] != "user":
            continue
        opener = messages[0]["content"].strip()
        reply = messages[1]["content"].strip()
        if not SMALLTALK_HARVEST.match(opener):
            continue
        if not reply or len(reply.split()) > SMALLTALK_HARVEST_MAX_WORDS:
            continue
        out.append({"source": "smoltalk-greetings", "task": "smalltalk",
                    "messages": json.dumps(
                        [{"role": "user", "content": opener},
                         {"role": "assistant", "content": reply}])})
    return out


def build_smalltalk_rows(rng, harvested=()):
    """Every prompt x reply pairing in SMALLTALK_GROUPS, plus the harvested ones.

    Enumerated rather than sampled: the pools are small and the whole point is
    that every phrasing is covered, so the model keys on the behaviour instead
    of on the three strings smoltalk happens to contain.
    """
    rows = list(harvested)
    for prompts, replies in SMALLTALK_GROUPS:
        for prompt in prompts:
            for reply in replies:
                rows.append({
                    "source": "smalltalk", "task": "smalltalk",
                    "messages": json.dumps(
                        [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": reply}]),
                })
    rng.shuffle(rows)
    return rows


def _head_command(messages_json):
    """First word of the assistant reply -- the command a shell answer runs."""
    try:
        messages = json.loads(messages_json)
    except (TypeError, ValueError):
        return ""
    if not messages:
        return ""
    reply = str(messages[-1].get("content", "")).strip()
    if not reply:
        return ""
    head = re.split(r"[\s|;&]+", reply)[0]
    # A leading `sudo`/`env` says nothing about what the row teaches; the word
    # after it does.
    if head in ("sudo", "env", "time", "nohup") and len(reply.split()) > 1:
        head = re.split(r"[\s|;&]+", reply)[1]
    return head


def _cap_head_command(rows, rng, caps=None):
    """Drop rows so no single opening command exceeds its task's share.

    See TASK_HEAD_COMMAND_CAP. Only tasks listed there are touched; everything
    else is returned untouched and in its original order.
    """
    caps = TASK_HEAD_COMMAND_CAP if caps is None else caps
    if not caps:
        return rows
    totals = Counter(r["task"] for r in rows if r["task"] in caps)
    if not totals:
        return rows

    order = list(range(len(rows)))
    rng.shuffle(order)          # which copies survive is random, not positional
    budgets, kept_flag = {}, [True] * len(rows)
    dropped = Counter()
    for i in order:
        row = rows[i]
        task = row["task"]
        if task not in caps:
            continue
        head = _head_command(row["messages"])
        if not head:
            continue
        key = (task, head)
        budget = budgets.get(key)
        if budget is None:
            budget = max(1, int(caps[task] * totals[task]))
            budgets[key] = budget
        if budget <= 0:
            kept_flag[i] = False
            dropped[key] += 1
            continue
        budgets[key] = budget - 1

    if dropped:
        for (task, head), n in dropped.most_common(5):
            print(f"  {task}: dropped {n:,} rows opening with `{head}` "
                  f"(cap {caps[task]:.0%} of {totals[task]:,})")
    return [r for r, keep in zip(rows, kept_flag) if keep]


def cache_paths(cache_dir=None):
    """(train, val) parquet paths for a cache directory."""
    folder = cache_dir or CACHE_FOLDER
    return (os.path.join(folder, "sft_train.parquet"),
            os.path.join(folder, "sft_val.parquet"))


def build_sft_cache(force=False, val_per_task=VAL_PER_TASK, cache_dir=None):
    """Normalize every downloaded source into two parquet files.

    Mirrors the pretrain side's "parquet on disk, read lazily" approach rather
    than re-parsing eleven different schemas on every training run.
    """
    train_cache, val_cache = cache_paths(cache_dir)
    if os.path.exists(train_cache) and os.path.exists(val_cache) and not force:
        return train_cache, val_cache

    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found -- run dataset_sft_download.py first")

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    os.makedirs(os.path.dirname(train_cache) or ".", exist_ok=True)
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

    # Small talk: harvest the real greeting openers buried in multi-turn rows,
    # then add the turns the corpus has none of. Built here rather than from a
    # manifest entry because there is no repo to download -- see the Small talk
    # section above.
    harvested = harvest_smalltalk(train_rows)
    smalltalk = build_smalltalk_rows(rng, harvested)
    print(f"Small talk: {len(harvested):,} harvested + "
          f"{len(smalltalk) - len(harvested):,} authored = {len(smalltalk):,} rows")
    train_rows.extend(smalltalk)

    # Flatten command monocultures before the split, so the holdout reflects
    # the same distribution the model is actually trained on.
    train_rows = _cap_head_command(train_rows, rng)
    val_rows = _cap_head_command(val_rows, rng)

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
        # val_per_task is per TASK, so what each source owes depends on how many
        # sources ITS task has -- not on how many sources exist in total.
        # Dividing by len(by_source) made the budget global instead: across 11
        # sources every donor gave 200 // 11 = 18 rows, which left `chat` with a
        # 28-example holdout while `extractive_qa`, which ships its own
        # validation split and donates nothing, kept all 200. Every small task
        # is a donor, so the tasks with the least data also got the least
        # validation -- writing, rewrite and shell would each have been steered
        # on ~14 examples, which is noise, not a curve.
        sources_per_task = Counter(task for task, _ in by_source)
        for pair, rows in by_source.items():
            per_source_target = max(
                1, val_per_task // max(1, sources_per_task[pair[0]]))
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
    pd.DataFrame(train_rows).to_parquet(train_cache)
    pd.DataFrame(val_rows).to_parquet(val_cache)

    print(f"\nWrote {len(train_rows):,} train / {len(val_rows):,} val conversations")
    for task in sorted({r["task"] for r in train_rows}):
        n = sum(1 for r in train_rows if r["task"] == task)
        v = sum(1 for r in val_rows if r["task"] == task)
        sources = sorted({r["source"] for r in val_rows if r["task"] == task})
        print(f"  {task:15s} {n:>8,} train | {v:>4,} val "
              f"from {', '.join(sources) or '-'}")

    return train_cache, val_cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Map-style dataset over normalized conversations, returning batches.

    Same contract as PretrainTextDataset: __getitem__(i) returns a *batch*
    (a list), so the DataLoader is used with batch_size=None.
    """

    def __init__(self, batch_size=8, split="train", task_weights=None,
                 total_examples=None, seed=SEED, epochs=1, cache_dir=None):
        self.batch_size = batch_size
        self.split = split

        train_cache, val_cache = cache_paths(cache_dir)
        path = train_cache if split == "train" else val_cache
        if not os.path.exists(path):
            build_sft_cache(cache_dir=cache_dir)
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

    def subset(self, task):
        """A shallow view of this dataset restricted to one task.

        Shares the underlying dataframe -- only the index is rebuilt -- so the
        trainer can hold one of these per task for per-task validation without
        re-reading the cache ten times. Returns None when the task has no rows.
        """
        ids = self._by_task.get(task)
        if ids is None or len(ids) == 0:
            return None
        view = copy.copy(self)
        view.index = ids
        view.tasks = [task]
        view._by_task = {task: ids}
        view._num_batches = math.ceil(len(ids) / self.batch_size)
        return view

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