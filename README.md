# nanollm

A complete training stack for a **186M parameter** decoder-only language model —
pretraining, supervised fine-tuning, DPO, and evaluation — sized for one GPU or
a small node.

```
896 dim · 18 layers · 14 heads (head_dim 64) · GQA 7:1 · 32k vocab · 2048 ctx
186,288,768 parameters (156.9M non-embedding, tied embeddings)
```

## Architecture

| Component | Choice | Why |
|---|---|---|
| Norm | RMSNorm, pre-norm | Cheaper than LayerNorm, no centering term needed |
| Attention | Grouped-query, 14 Q / 2 KV heads | 7× smaller KV cache at inference, no measurable quality cost at this scale |
| Position | RoPE, θ=10000 | Extrapolates better than learned embeddings; no parameters |
| QK-norm | RMSNorm per head, pre-RoPE | Attention-logit drift is *the* instability at depth and high LR |
| FFN | SwiGLU, hidden = 8/3·d → 2560 | Param-matched to a 4× GELU FFN, consistently better |
| Head dim | **64** | Flash-attention kernels are tuned for 32/64/128; 56 falls back to the slow math kernel |
| Loss | Chunked CE + z-loss (1e-4) | Keeps the logit tensor out of peak memory; z-loss stops log-Z drift |
| Embeddings | Tied input/output | 29M parameters saved and slightly better at this scale |


## Quickstart

```bash
pip install -e ".[all]"

python scripts/download_data.py pretrain --budget-gb 70   # raw corpus
python scripts/train_tokenizer.py --vocab-size 32768      # BPE, once
python scripts/prepare_data.py --workers 16 --max-tokens 7_000_000_000

python -m nanollm.train.pretrain --config configs/base.yaml
```

Multi-GPU is the same command under `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m nanollm.train.pretrain \
    --config configs/base.yaml
```

Override any field without editing the YAML:

```bash
python -m nanollm.train.pretrain --config configs/base.yaml \
    --set optim.peak_lr=3e-4 --set optim.schedule=wsd --set runtime.wandb=true
```

## The full pipeline

```
download_data.py  →  train_tokenizer.py  →  prepare_data.py  →  train.pretrain
                                                                      ↓
                                                    train.pretrain (mid-training)
                                                      on curated textbook prose
                                                                      ↓
                                                          train.sft (+ replay)
                                                                      ↓
                                                                 train.dpo
                                                                      ↓
                                                              evaluate.py
```

```bash
python scripts/download_midtrain.py                       # textbook corpus
python scripts/prepare_data.py --raw-dir dataset/raw_midtrain \
    --out-dir dataset/tokens_midtrain --no-wikipedia
python -m nanollm.train.pretrain --config configs/midtrain.yaml \
    --init-from artifacts/pretrain_checkpoint_latest.pt   # knowledge annealing

python -m nanollm.train.sft --config configs/base.yaml \
    --init-from artifacts/midtrain/pretrain_model.pt      # instruction tuning
python -m nanollm.train.dpo --config configs/base.yaml    # preference tuning
python scripts/evaluate.py --checkpoint artifacts/dpo_model.pt
python scripts/generate.py --prompt "The history of" --tokens 200   # base model
python scripts/chat.py                                              # chat model
```

### Chatting with the tuned model

`scripts/chat.py` picks the most post-trained checkpoint it can find
(`dpo_model.pt` → `sft_model.pt`) and streams replies token by token, reusing a
KV cache across the turn. It builds prompts with the exact template SFT trained
on — `<|BOS|> <|USER|> … <|ASSISTANT|> … <|EOS|>` — and stops at `<|EOS|>`,
which SFT explicitly trains the model to emit.

```bash
python scripts/chat.py                                   # interactive
python scripts/chat.py --prompt "Explain gravity." --once
python scripts/chat.py --checkpoint artifacts/sft_model.pt --temperature 0.6
python scripts/chat.py --checkpoint artifacts/dpo_model.pt --temperature 0.2
```

In-session commands: `/reset`, `/retry`, `/undo`, `/history`, `/tokens`,
`/think`, `/params`, `/set temperature=0.3`, `/help`, `/exit`.

`/think` asks for a scratchpad before the answer:

```
Thinking: A bloody mess is covered or stained with blood.
Answer: A) bloody mess
```

It is plain text, not a special token — the tokenizer's six specials are fixed
by the pretrained checkpoint, and a seventh would shift `merge_id_offset` and
invalidate every merge id. SFT attaches it only where the source data carries a
real rationale (ecqa, gsm8k, metamath), and trains the direct-answer contrast
alongside it so the mode is something you ask for rather than a house style.
`nanollm.data.sft.split_thinking` splits a reply back into the two parts.

When history outgrows the 2048-token context the **oldest** turns are dropped —
the current question is the part that has to survive intact.

## Mid-training on textbooks

Pretraining is 10B tokens of web crawl and synthetic prose, and it shows: the
base model writes fluent English and then says *Pride and Prejudice* was
written by Elizabeth Gaskell. A model this size cannot be argued out of that
with more of the same data, so there is a short stage between pretraining and
SFT that runs the same loop over a small curated corpus at a low LR.

| source | what it is |
|---|---|
| openstax | 76 OpenStax college textbooks, PDF-extracted |
| ncert | NCERT classes 10-12, English-medium subjects |
| libretexts-chem | LibreTexts chemistry pages |

```bash
python scripts/download_midtrain.py
python scripts/prepare_data.py --raw-dir dataset/raw_midtrain \
    --out-dir dataset/tokens_midtrain --no-wikipedia
python scripts/mix_shards.py --into dataset/tokens_midtrain \
    --from dataset/tokens --shards 2          # rehearsal, see below
python -m nanollm.train.pretrain --config configs/midtrain.yaml \
    --init-from artifacts/pretrain_checkpoint_latest.pt
```

The corpus is small — ~46.5M tokens (openstax 37.0M, libretexts-chem 6.9M,
ncert 2.6M) — so `configs/midtrain.yaml` runs 400 steps, about two epochs over
it. Pure textbook at that length drifts the model off general text the way pure
SFT does, and the pretrain loop has no replay knob, so the rehearsal has to be
in the data: `mix_shards.py` links a couple of the original 100M-token shards
in beside the textbook ones and rewrites the manifest, leaving the textbooks at
~19% of what the stage sees.

`--init-from` is not `--resume`: it loads weights, starts at step 0 with a
fresh schedule, and restores no optimizer state — the AdamW moments from the
end of a 38k-step cosine describe a different objective at a much larger LR.

The corpus arrives as PDFs and PDF-extracted text, so `nanollm.data.midtrain`
spends most of its lines undoing page layout: running headers are found by
looking for lines repeated across a quarter of a book's pages (every publisher
repeats something different, so a regex for "Physics" would delete the word
from the body too), lines that are mostly not letters are dropped as equation
debris, and lines that do not end a sentence are glued to the next one because
the PDF broke them at the column edge. Maths does not survive extraction and
is not meant to — the prose is the point.

**What this stage does and does not do.** It raises quality inside the subjects
the books cover and pulls the default register toward expository prose. It does
not delete anything pretraining learned — nothing is unlearned, it is
outweighed, and only where the textbooks have coverage. English literature
trivia is in neither OpenStax nor NCERT, so the Elizabeth Gaskell answer is not
what this fixes. At 169M parameters, knowledge capacity is the binding
constraint; this buys a better-calibrated model of what it was taught, not a
bigger one. Watch validation loss on the **original** pretraining shards while
it runs: if it climbs more than ~0.05 the stage has gone too far.

## Why pre-tokenized shards

Tokenising inside the training loop is the default failure mode of small LLM
projects: a pure-Python BPE caps throughput far below what the GPU can consume,
and it makes exact resume impossible. Here the corpus is tokenised **once** into
memory-mapped `uint16` shards (`nanollm/data/shards.py`), which buys:

- **No CPU bottleneck.** A batch is a memmap read; one process saturates an A100.
- **Exact resume.** Batch contents are a pure function of `(step, rank)`, so
  resuming at step *N* reproduces the stream without replaying the pipeline.
- **Half the bytes.** `uint16` covers a 32k vocab; at 6B tokens that is 12GB of
  page cache instead of 24GB.

Validation uses whole held-out **shards**, not a random slice of the training
shards — otherwise every validation window overlaps training data and the number
stops meaning anything.

### Disk

Shards cost **2 bytes per token** (uint16), so size `--max-tokens` to the disk
you have. Without it, `prepare_data.py` runs until the corpus ends — which on a
10BT sample is ~24GB of shards. It stops cleanly at `--min-free-gb` (default 5)
rather than filling the filesystem, and the shards written up to that point are
valid and usable.

| tokens | shards on disk | enough for |
|---|---|---|
| 3.7B | 7.4 GB | Chinchilla-optimal (20 tok/param) |
| 6.3B | 12.6 GB | a 24k-step run (34 tok/param) |
| 10.1B | 20.1 GB | the default 38.4k-step run |
| 11.8B | 23.6 GB | a 45k-step run (63 tok/param) |

Budget separately for checkpoints: ~2.2GB each (weights + AdamW moments), plus
one snapshot every `snapshot_every` steps.

## Measured throughput

Single A100 80GB, `configs/base.yaml`, micro_batch 16 × 2048 ctx, bf16:

| | tokens/sec | MFU | peak memory |
|---|---|---|---|
| `torch.compile` on | **88.3k** | **42.8%** | 41.3 / 80 GB |
| eager | 56.3k | 27.3% | 41.3 / 80 GB |

Compile is worth 1.57× and costs a one-off ~90s warmup on the first step. At
88.3k tok/s the full 10.1B-token run is about **32 hours on one A100**, or ~4
hours on 8.

## Token budget

The default run is 38,400 steps × 262,144 tokens = **10.1B tokens**, about 54
tokens per parameter. Chinchilla-optimal is ~20, and going past it is the right
call for a model this small: inference cost dominates, so you buy quality with
training tokens rather than parameters.

The mix that feeds it, by share of the download budget:

| source | share | what it is |
|---|---|---|
| fineweb-edu | 0.42 | classifier-filtered educational web text |
| cosmopedia | 0.38 | synthetic textbooks and stories |
| fineweb | 0.14 | general web crawl, for register diversity |
| python-edu | 0.06 | deduplicated Python from GitHub |

~70GB of raw parquet tokenises to roughly 10B tokens after the language filter
and the short-document drop in `split_long_text`. Download less and the run
simply wraps around the shards it has — which costs you a second epoch on part
of the corpus, not a crash.

Scale the batch to your GPU by trading `micro_batch_size` against
`grad_accum_steps` — the product sets the global batch, so the loss curve is
unchanged:

| GPU | micro_batch | grad_accum | global batch |
|---|---|---|---|
| A100/H100 80GB | 16 | 8 | 128 seqs (262k tok) |
| A100 40GB | 8 | 16 | 128 seqs |
| 4090 24GB | 4 | 32 | 128 seqs |
| 8×A100 | 16 | 1 | 128 seqs |

On <16GB cards also set `--set model.activation_checkpointing=true`.

## Layout

```
src/nanollm/
  config.py          typed config; a run is fully described by TrainConfig
  model.py           the transformer
  tokenizer.py       byte-level BPE
  data/
    shards.py        binary token shard format (write once, memmap forever)
    loader.py        deterministic (step, rank) → batch, CUDA prefetch
    prepare.py       corpus → shards, across a process pool
    sources.py       streaming readers for parquet / wikipedia
    sft.py  dpo.py   task mixtures, chat rendering, loss masking
  train/
    common.py        pieces shared by all three loops
    pretrain.py  sft.py  dpo.py
  eval/harness.py    perplexity + multiple-choice likelihood
  utils/             distributed, checkpoint, schedules, logging
configs/             base.yaml (the reference run), debug.yaml (fast smoke test)
scripts/             download / tokenizer / prepare / generate / evaluate
tools/               checkpoint and embedding-flow diagnostics
tests/               63 tests, no GPU required
rough/               scratch work and the superseded flat training scripts
```

## Things that are easy to get wrong, and how they are handled

- **Residual init applied twice.** The GPT-2 `1/sqrt(2L)` scaling lives in the
  model and nowhere else. Applying it again in the training script squares the
  factor and starts the run at `std/(2L)`. Covered by a test.
- **Non-finite gradients.** Scaling grads by zero does *not* neutralise a NaN
  (`NaN * 0 = NaN`), and a NaN reaching AdamW's `exp_avg` poisons every
  subsequent step even after gradients recover. `clip_and_step` sanitises with
  `nan_to_num_`, without a host sync.
- **Non-causal prefill.** Cached prefill must stay causal, or the prompt is
  bidirectional at inference and unidirectional in training. A test asserts
  cached decoding equals a full forward pass.
- **`padding_idx` with tied embeddings** permanently pins token 0's output
  logit to 0, so it can never be predicted. Not used.
- **Prompt tokens in the SFT loss.** Loss is on assistant tokens only; otherwise
  most capacity goes to reproducing prompts.
- **Catastrophic forgetting in SFT.** 25% of micro-batches are pretraining
  replay, and the LR is 1e-5, not 3e-5.
- **A downloader that only handles parquet.** `datasets` is not a dependency
  here, so the old `load_dataset` fallback raised `ModuleNotFoundError` for
  every json/jsonl/csv repo and `main()` logged it as one `FAILED` line. Five
  configured sources were absent from the mixture with nothing downstream to
  notice. `download_plain_files` reads them with pandas instead.
- **A validation budget divided by the wrong denominator.** `val_per_task` was
  split across *every* source rather than each task's own, so tasks that ship
  no validation split were steered on ~14 examples. Every small task is a
  donor, so the tasks with the least data also got the least validation.
- **An aggregate SFT val curve over ten tasks.** A task that never learns hides
  inside the weighted average. `sft.val_task_batches` reports each one.
- **DPO collapse.** Pure DPO is satisfied by pushing *both* log-probs down as
  long as the margin grows. An NLL term on the chosen reply anchors it.
- **A mixture whose weights describe examples while the gradient follows
  tokens.** The loss is a mean over supervised tokens, so an example pulls in
  proportion to its reply length. At the configured mixture that hands `chat`
  78% of the gradient against its nominal 24%, and `extractive_qa` 0.4% against
  12% — the carefully-tuned `TASK_WEIGHTS` were largely fiction. `sft.loss_normalize`
  selects which one the run means.
- **An EOS token diluted by reply length.** Learning to stop is one token out
  of however many the reply has, so a 6-token answer trains stopping ~25x
  harder than a 155-token draft. Measured: P(EOS) at the correct stopping
  position falls from 0.84 on replies under 16 tokens to 0.55 past 257, and the
  tasks that ramble past the length cap (`writing` 0.39, `instruct` 0.43,
  `chat` 0.61) are exactly the long-reply ones. `sft.eos_loss_share` pins EOS at
  a floor share of each example's loss, and only ever upward — short replies
  already stop at ~0.99 and pushing them harder just trades answer for stop.
- **A task that is one command wearing a corpus as a hat.** 43% of the `shell`
  replies opened with `find` against 0.9% for `ls`, because nl2bash and nl2sh
  are `find` corpora; the task weight bought flag trivia, and "list the files
  including hidden ones" came back as a `find` pipeline ending in `rm -f`.
  `TASK_HEAD_COMMAND_CAP` thins any single opening command down to a share of
  its task.
- **No answer to "hi".** The greeting rows exist and are correct — and are
  1,872 first exchanges of 6-to-8-turn conversations, so the eight tokens of
  "Hello! How can I help you today?" were ~0.04% of an epoch's gradient and the
  model answered `hi` with an essay. The `smalltalk` task lifts those exchanges
  out standalone and adds the turns the corpus has none of: thanks, goodbye,
  and the identity questions that otherwise get answered out of whatever
  first-person narration the corpus happens to contain ("I am Elena, a young
  woman from a small town in the Czech Republic").
- **Sampling defaults inherited from a larger model.** At temperature 0.8 this
  model answers "who wrote Pride and Prejudice" with "Elizabeth Taylor"; greedy
  gets Jane Austen. But cutting temperature alone drives it into repetition
  loops, so the default pairs a lower temperature with `min_p` and a stronger
  repetition penalty rather than just turning the dial down.
- **Sources read in blocks.** Reading corpora one after another steps the loss
  at each boundary. `iter_all_documents` interleaves by size instead.

## Tests

```bash
pytest              # 63 tests, CPU only, ~40s
pytest -m 'not slow' -k 'not integration'   # skip the subprocess run
```

## Legacy imports

The original flat module names still work and forward to the package:

```python
from simple_transformer import Transformer   # → nanollm.model
from config import CONFIG                    # → nanollm.config
from tokenizer import Tokenizer              # → nanollm.tokenizer
from sft_dataset import SFTDataset           # → nanollm.data.sft
python train_transformer_pretraining.py      # → nanollm.train.pretrain
```
--------------------------
```bash
# 1. Mid-training — knowledge annealing on textbooks (~15 min)
python scripts/download_midtrain.py
python scripts/prepare_data.py --raw-dir dataset/raw_midtrain \
    --out-dir dataset/tokens_midtrain --no-wikipedia
python scripts/mix_shards.py --into dataset/tokens_midtrain \
    --from dataset/tokens --shards 2          # rehearsal — don't skip this
python -m nanollm.train.pretrain --config configs/midtrain.yaml \
    --init-from artifacts/pretrain_checkpoint_latest.pt

# 2. SFT off the annealed checkpoint (~36 min)
python -m nanollm.train.sft --config configs/base.yaml \
    --init-from artifacts/midtrain/pretrain_model.pt

# 3. DPO (~22 min)
python -m nanollm.train.dpo --config configs/base.yaml

# 4. Check it
python scripts/evaluate.py --checkpoint artifacts/dpo_model.pt
python scripts/chat.py

```