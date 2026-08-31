"""Chat session state and streaming generation against the SFT/DPO template.

The prompt layout has to match ``nanollm.data.sft.render_conversation`` exactly
or the model is being asked to continue a format it never saw in training:

    <|BOS|> <|USER|> ...text... <|ASSISTANT|> ...reply... <|EOS|>

Generation stops at ``<|EOS|>``, which is the token SFT explicitly trained the
model to emit -- learning to stop is part of the objective, so a well-trained
checkpoint terminates on its own rather than running to the length cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import torch

__all__ = ["SamplingParams", "ChatSession", "stream_reply"]


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256

    def describe(self) -> str:
        return (f"temp={self.temperature} top_k={self.top_k} top_p={self.top_p} "
                f"min_p={self.min_p} rep={self.repetition_penalty} "
                f"max_new={self.max_new_tokens}")


@dataclass
class ChatSession:
    """Multi-turn history plus prompt construction.

    History is a flat list of ``{"role": "user"|"assistant", "content": str}``,
    the same shape the SFT dataset produces.
    """

    tokenizer: object
    n_seq: int
    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self.messages.clear()

    def undo(self) -> None:
        """Drop the last exchange, so /retry can resample it."""
        while self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

    def build_prompt(self, reserve: int) -> tuple[list[int], int]:
        """Token ids ending at ``<|ASSISTANT|>``, ready to continue from.

        Returns ``(ids, dropped_turns)``. When history plus the reply budget
        would overflow the context, the *oldest* turns are dropped rather than
        truncating the newest -- the current question is the part that must
        survive intact.
        """
        special = self.tokenizer.special_tokens
        bos, user, assistant = (special["<|BOS|>"], special["<|USER|>"],
                                special["<|ASSISTANT|>"])
        eos = special["<|EOS|>"]

        encoded = []
        for message in self.messages:
            marker = user if message["role"] == "user" else assistant
            body = self.tokenizer.encode(message["content"])
            # An assistant turn is closed with EOS, exactly as SFT rendered it.
            tail = [eos] if message["role"] == "assistant" else []
            encoded.append([marker] + body + tail)

        budget = self.n_seq - reserve - 2      # room for BOS and the final marker
        dropped = 0
        while True:
            total = sum(len(chunk) for chunk in encoded[dropped:])
            if total <= budget or dropped >= len(encoded) - 1:
                break
            dropped += 1

        ids = [bos]
        for chunk in encoded[dropped:]:
            ids.extend(chunk)
        ids.append(assistant)

        # Last resort: a single question longer than the whole context. Keep
        # the head marker and the tail of the question.
        if len(ids) > self.n_seq - reserve:
            keep = self.n_seq - reserve - 2
            ids = [bos, user] + ids[len(ids) - keep:]
        return ids, dropped


@torch.inference_mode()
def stream_reply(model, tokenizer, prompt_ids: list[int], params: SamplingParams,
                 device, amp_dtype=torch.bfloat16,
                 stop_check: Optional[Callable[[], bool]] = None) -> Iterator[str]:
    """Yield decoded text deltas as tokens are produced.

    Text is re-decoded from the full id list each step and only the new suffix
    is yielded. Decoding token-by-token would split multi-byte UTF-8 sequences
    and emit replacement characters mid-word.
    """
    eos_id = tokenizer.special_tokens["<|EOS|>"]
    valid_vocab = getattr(tokenizer, "vocab_size", None)

    budget = max(1, min(params.max_new_tokens, model.n_seq - len(prompt_ids)))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = model.make_kv_cache(1, len(prompt_ids) + budget, device)

    autocast = torch.autocast(device_type=device.type, dtype=amp_dtype,
                              enabled=device.type == "cuda")

    with autocast:
        hidden = model.forward_hidden(idx, start_pos=0, kv_cache=cache)
    logits = model.logit_proj(hidden[:, -1]).float()

    produced: list[int] = []
    text = ""
    position = len(prompt_ids)

    for _ in range(budget):
        if valid_vocab is not None and valid_vocab < model.vocab_size:
            logits[:, valid_vocab:] = float("-inf")

        if params.repetition_penalty != 1.0 and produced:
            # Penalise only what this reply has emitted. Penalising the prompt
            # too makes the model avoid the user's own words, which reads as
            # evasive.
            recent = torch.tensor([produced], device=device)
            gathered = torch.gather(logits, 1, recent)
            gathered = torch.where(gathered > 0,
                                   gathered / params.repetition_penalty,
                                   gathered * params.repetition_penalty)
            logits = logits.scatter(1, recent, gathered)

        next_token = model._sample(logits, params.temperature, params.top_k,
                                   params.top_p, params.min_p)
        token_id = int(next_token.item())
        if token_id == eos_id:
            break

        produced.append(token_id)
        decoded = tokenizer.decode(produced)
        if len(decoded) > len(text):
            yield decoded[len(text):]
            text = decoded

        if stop_check is not None and stop_check():
            break

        with autocast:
            hidden = model.forward_hidden(next_token, start_pos=position,
                                          kv_cache=cache)
        logits = model.logit_proj(hidden[:, -1]).float()
        position += 1
