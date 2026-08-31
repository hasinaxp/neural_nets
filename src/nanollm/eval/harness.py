"""Evaluation for small base and chat models.

Two things worth measuring at this scale:

* **Perplexity** on held-out shards -- the only number that tracks pretraining
  progress reliably.
* **Multiple-choice accuracy** by length-normalised likelihood, the standard
  zero-shot protocol (HellaSwag, ARC, PIQA and friends). A 186M model scores
  near chance on most of these; the useful signal is the trend across
  checkpoints, not the absolute number.

Generation quality is not scored here. Automatic scores for open-ended text
from a model this size are noise; read the samples.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..model import IGNORE_INDEX

__all__ = ["evaluate_perplexity", "score_continuations", "evaluate_multiple_choice",
           "MultipleChoiceExample", "load_jsonl_mc"]


@torch.no_grad()
def evaluate_perplexity(model, batches, device, amp_dtype=torch.bfloat16,
                        max_batches: Optional[int] = None) -> dict:
    """Token-level cross entropy over a fixed set of (xs, ys) batches."""
    was_training = model.training
    model.eval()

    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for i, (xs, ys) in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        xs, ys = xs.to(device), ys.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            hidden = model.forward_hidden(xs)
        chunk = model.loss_chunk_size or hidden.size(1)
        for j in range(0, hidden.size(1), chunk):
            logits = model.logit_proj(hidden[:, j:j + chunk]).float()
            target = ys[:, j:j + chunk]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                reduction="sum", ignore_index=IGNORE_INDEX)
            total_nll += nll.double()
            total_tokens += (target != IGNORE_INDEX).sum().double()

    if was_training:
        model.train()

    tokens = max(1.0, float(total_tokens.item()))
    mean_nll = float(total_nll.item()) / tokens
    return {
        "loss": mean_nll,
        "perplexity": math.exp(min(20.0, mean_nll)),
        "bits_per_token": mean_nll / math.log(2),
        "tokens": int(tokens),
    }


@dataclass
class MultipleChoiceExample:
    context: str
    choices: list[str]
    answer: int          # index into choices
    task: str = ""


@torch.no_grad()
def score_continuations(model, tokenizer, context: str, continuations: list[str],
                        device, amp_dtype=torch.bfloat16,
                        max_len: Optional[int] = None) -> list[tuple[float, float]]:
    """Log-likelihood of each continuation given the context.

    Returns ``[(total_logprob, per_token_logprob), ...]``. Both are reported
    because the two disagree often: raw likelihood favours short answers, and
    length-normalising is the usual correction.
    """
    was_training = model.training
    model.eval()

    bos = tokenizer.special_tokens.get("<|BOS|>")
    ctx_ids = ([bos] if bos is not None else []) + tokenizer.encode(context)
    limit = max_len or model.n_seq

    out = []
    for continuation in continuations:
        cont_ids = tokenizer.encode(continuation)
        if not cont_ids:
            out.append((float("-inf"), float("-inf")))
            continue

        ids = ctx_ids + cont_ids
        if len(ids) > limit:
            # Trim the context from the front; the continuation must survive
            # intact or the scores are not comparable across choices.
            ids = ids[len(ids) - limit:]
        n_cont = min(len(cont_ids), len(ids) - 1)

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            hidden = model.forward_hidden(idx)
        logits = model.logit_proj(hidden[:, -n_cont - 1:-1]).float()
        targets = idx[:, -n_cont:]
        logprobs = torch.log_softmax(logits, dim=-1)
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total = float(picked.sum().item())
        out.append((total, total / max(1, n_cont)))

    if was_training:
        model.train()
    return out


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, examples: Iterable[MultipleChoiceExample],
                             device, amp_dtype=torch.bfloat16,
                             log_every: int = 0, log=print) -> dict:
    """Zero-shot accuracy under both raw and length-normalised likelihood."""
    n = 0
    correct_raw = 0
    correct_norm = 0
    per_task: dict[str, list[int]] = {}

    for example in examples:
        scores = score_continuations(
            model, tokenizer, example.context, example.choices, device, amp_dtype)
        pick_raw = max(range(len(scores)), key=lambda i: scores[i][0])
        pick_norm = max(range(len(scores)), key=lambda i: scores[i][1])

        n += 1
        hit_raw = int(pick_raw == example.answer)
        hit_norm = int(pick_norm == example.answer)
        correct_raw += hit_raw
        correct_norm += hit_norm
        per_task.setdefault(example.task or "all", []).append(hit_norm)

        if log_every and n % log_every == 0:
            log(f"  {n} examples | acc_norm {correct_norm/n:.3f}")

    if n == 0:
        return {"n": 0, "accuracy": 0.0, "accuracy_raw": 0.0}

    result = {
        "n": n,
        "accuracy": correct_norm / n,          # length-normalised, the headline
        "accuracy_raw": correct_raw / n,
        "by_task": {task: sum(hits) / len(hits) for task, hits in per_task.items()},
    }
    return result


def load_jsonl_mc(path: str, limit: Optional[int] = None,
                  task: str = "") -> list[MultipleChoiceExample]:
    """Read multiple-choice examples from JSONL.

    Expected fields per line::

        {"context": "...", "choices": ["...", "..."], "answer": 0}

    ``label``/``gold`` are accepted as aliases for ``answer``.
    """
    examples = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            blob = json.loads(line)
            answer = blob.get("answer", blob.get("label", blob.get("gold")))
            examples.append(MultipleChoiceExample(
                context=blob.get("context") or blob.get("query") or blob.get("prompt", ""),
                choices=list(blob["choices"] if "choices" in blob else blob["endings"]),
                answer=int(answer),
                task=blob.get("task") or task or os.path.basename(path),
            ))
            if limit and len(examples) >= limit:
                break
    return examples
