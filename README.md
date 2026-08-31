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

python scripts/download_data.py pretrain --budget-gb 40   # raw corpus
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
                                                          train.sft (+ replay)
                                                                      ↓
                                                                 train.dpo
                                                                      ↓
                                                              evaluate.py
```

```bash
python -m nanollm.train.sft --config configs/base.yaml    # instruction tuning
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
```

In-session commands: `/reset`, `/retry`, `/undo`, `/history`, `/tokens`,
`/params`, `/set temperature=0.3`, `/help`, `/exit`.

When history outgrows the 2048-token context the **oldest** turns are dropped —
the current question is the part that has to survive intact.

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
| 6.3B | 12.6 GB | the default 24k-step run |
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
88.3k tok/s the full 6.3B-token run is about **20 hours on one A100**, or ~2.5
hours on 8.

## Token budget

The default run is 24,000 steps × 262,144 tokens = **6.3B tokens**, about 34
tokens per parameter. Chinchilla-optimal is ~20, and going past it is the right
call for a model this small: inference cost dominates, so you buy quality with
training tokens rather than parameters.

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
- **DPO collapse.** Pure DPO is satisfied by pushing *both* log-probs down as
  long as the margin grows. An NLL term on the chosen reply anchors it.
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
