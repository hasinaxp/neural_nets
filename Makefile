.PHONY: help install data tokenizer shards pretrain sft dpo eval sample chat smoke test lint clean

PY      ?= python
CONFIG  ?= configs/base.yaml
GPUS    ?= 1

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with every extra
	pip install -e ".[all]"

data:  ## download the pretraining corpus
	$(PY) scripts/download_data.py pretrain --budget-gb 40

tokenizer:  ## train the BPE tokenizer
	$(PY) scripts/train_tokenizer.py --vocab-size 32768

shards:  ## tokenize the corpus into binary shards
	$(PY) scripts/prepare_data.py --config $(CONFIG)

pretrain:  ## pretrain (GPUS=8 for multi-GPU)
ifeq ($(GPUS),1)
	$(PY) -m nanollm.train.pretrain --config $(CONFIG)
else
	torchrun --standalone --nproc_per_node=$(GPUS) -m nanollm.train.pretrain --config $(CONFIG)
endif

sft:  ## supervised fine-tuning
	$(PY) -m nanollm.train.sft --config $(CONFIG)

dpo:  ## preference tuning
	$(PY) -m nanollm.train.dpo --config $(CONFIG)

eval:  ## evaluate the latest checkpoint
	$(PY) scripts/evaluate.py --config $(CONFIG)

sample:  ## generate from the latest checkpoint (base model)
	$(PY) scripts/generate.py --config $(CONFIG)

chat:  ## interactive chat with the DPO (or SFT) model
	$(PY) scripts/chat.py

smoke:  ## end-to-end run on the debug config
	$(PY) -m nanollm.train.pretrain --config configs/debug.yaml --dry-run

test:  ## run the test suite
	$(PY) -m pytest

lint:
	ruff check src tests scripts

clean:  ## remove caches and temp files
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
