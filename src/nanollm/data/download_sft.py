import argparse
import json
import os
import re

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm


DATASET_FOLDER = "dataset/sft"
REPO_TYPE = "dataset"
MANIFEST_FILE = f"{DATASET_FOLDER}/manifest.json"

# Splits we keep. "test" is dropped -- we build our own held-out sets from
# validation so the published test sets stay clean for later comparison.
KEEP_SPLITS = ("train", "validation", "valid", "dev")


# Every entry lands in dataset/sft/<key>/*.parquet, same shape as
# dataset/raw/<key>/*.parquet on the pretrain side.
#
#   id        HF repo id
#   task      what this data teaches; the loader uses it to weight the mixture
#             (a normalizer may override it per row -- see ROUTERS in sft.py)
#   max_rows  subsample cap (None = take everything). Applied per split.
#   config    HF config/subset name, when the repo has more than one
#   files     explicit {split: remote path} for repos that ship no parquet.
#             Needed because `datasets` is not installed here, so the old
#             load_dataset() fallback silently failed and left five configured
#             sources with zero rows on disk. json/jsonl/csv are read directly
#             with pandas and written out as parquet.
#   enabled   set False to skip by default
REPOS = {
    # ---- chat / small talk -------------------------------------------------
    "smoltalk": {
        "id": "HuggingFaceTB/smol-smoltalk",
        "task": "chat",
        "max_rows": None,
    },
    # Routed per row by category in sft.py: Generation/Brainstorm become
    # `writing`, Rewrite becomes `rewrite`, the rest stay `chat`. 8.7k of these
    # 19k rows are human-written Generation examples -- the single best drafting
    # data in the whole mixture, and until now all of it was buried in `chat`
    # where its weight was set by how much small talk we wanted.
    "no-robots": {
        "id": "HuggingFaceH4/no_robots",
        "task": "chat",
        "max_rows": None,
    },
    "soda": {
        "id": "allenai/soda",
        "task": "chat",
        "max_rows": 60_000,
        # Off. soda is narrative dialogue between two *named characters*, and
        # the normalizer can only map them onto user/assistant by turn parity.
        # That teaches the model to play a random person in a chat rather than
        # to be an assistant -- inside the highest-weighted task. smol-smoltalk
        # covers multi-turn chat with an actual assistant on one side.
        "enabled": False,
    },
    # ---- extractive QA -----------------------------------------------------
    "squad-v2": {
        "id": "rajpurkar/squad_v2",
        "task": "extractive_qa",
        "max_rows": 80_000,
    },
    "sciq": {
        "id": "allenai/sciq",
        "task": "extractive_qa",
        "max_rows": None,
    },
    # ---- summarization -----------------------------------------------------
    # Samsung/samsum was here and is gone: the repo now 401s for anonymous
    # clients, which is why dataset/sft/samsum/ is empty. dialogsum covers the
    # same dialogue-summary shape and xsum covers the article shape.
    "dialogsum": {
        "id": "knkarthick/dialogsum",
        "task": "summarization",
        "max_rows": None,
        "files": {"train": "train.csv", "validation": "validation.csv"},
    },
    "xsum": {
        "id": "EdinburghNLP/xsum",
        "task": "summarization",
        "max_rows": 40_000,
        # On: sft.seq_len is 2048 now, so the long articles this was disabled
        # for mostly fit. It is also the only non-dialogue summarization source
        # here -- without it the task is entirely chat transcripts.
    },
    # ---- rewriting / rephrasing -------------------------------------------
    # CoEdIT is the whole reason `rewrite` can be its own task: 69k rows of
    # paired src/tgt across six edit types (gec, paraphrase, simplification,
    # coherence, neutralize, clarity). The upstream `src` field already carries
    # an instruction prefix; the normalizer splits it off and re-wraps with our
    # own paraphrases, so the model keys off the behaviour, not one sentence.
    "coedit": {
        "id": "grammarly/coedit",
        "task": "rewrite",
        "max_rows": None,
        "files": {"train": "train.jsonl", "validation": "validation.jsonl"},
    },
    # ---- linux shell -------------------------------------------------------
    # Two sources so the task is not one author's phrasing habits. nl2bash is
    # the classic hand-built set, NL2SH-ALFA is larger and noisier; capped so
    # it cannot swamp the cleaner one.
    "nl2bash": {
        "id": "AnishJoshi/nl2bash-custom",
        "task": "shell",
        "max_rows": None,
        "files": {"train": "data/train.json", "validation": "data/dev.json"},
    },
    "nl2sh": {
        "id": "westenfelder/NL2SH-ALFA",
        "task": "shell",
        "max_rows": 25_000,
        "files": {"train": "train.csv"},
    },
    # ---- text to SQL -------------------------------------------------------
    "sql-create-context": {
        "id": "b-mc2/sql-create-context",
        "task": "sql",
        "max_rows": None,            # 78.6k, schema is in the prompt
        "files": {"train": "sql_create_context_v4.json"},
    },
    "synthetic-text-to-sql": {
        "id": "gretelai/synthetic_text_to_sql",
        "task": "sql",
        "max_rows": None,            # ~106k; filter complexity at load time
    },
    # ---- instruction variety ----------------------------------------------
    # Also routed per row: creative_writing and brainstorming become `writing`.
    "dolly": {
        "id": "databricks/databricks-dolly-15k",
        "task": "instruct",
        "max_rows": None,
        "files": {"train": "databricks-dolly-15k.jsonl"},
    },
    # ---- math (chain-of-thought word problems) -----------------------------
    # gsm8k alone is ~7.5k rows, and at a 0.10 task weight that was small
    # enough to set the size of the entire SFT epoch through the min() in
    # _build_mixture -- 88% of the chat data went unused so gsm8k could be
    # repeated 3x. These two bring the math pool to ~130k so it stops being
    # the binding constraint; the mixture sizing is also fixed independently.
    "gsm8k": {
        "id": "openai/gsm8k",
        "task": "math",
        "config": "main",            # repo also has a "socratic" variant
        "max_rows": None,            # ~7.5k train rows
    },
    "metamath": {
        "id": "meta-math/MetaMathQA",
        "task": "math",
        # Raised from 60k. metamath closes every solution with "The answer
        # is: X", so ~70% of its rows can be split into a Thinking: block and a
        # final answer. It is the only large maths source where that is true,
        # so it should be the one carrying the task.
        "max_rows": 90_000,          # ~395k upstream, GSM8K/MATH augmentations
        "files": {"train": "MetaMathQA-395K.json"},
    },
    "orca-math": {
        "id": "microsoft/orca-math-word-problems-200k",
        "task": "math",
        # Cut from 60k. orca-math's solutions end in a free-form sentence with
        # no answer marker, so they stay in their original shape -- fine on its
        # own, but at 60k rows it outnumbered the think-formatted sources 8:1
        # and diluted the format it cannot participate in. Kept for the variety
        # of its problem phrasings.
        "max_rows": 25_000,          # ~200k upstream, worked solutions
    },
    # ---- reasoning (grounded multiple-choice) -------------------------------
    # arc/commonsense_qa ship no rationale, so their replies are the bare
    # choice. ecqa is CommonsenseQA *with* a human-written explanation, which
    # is what makes a Thinking: block on this task real rather than invented --
    # see THINK_PROMPTS in sft.py.
    "ecqa": {
        "id": "yangdong/ecqa",
        "task": "reasoning",
        "max_rows": None,            # ~7.6k train rows, human rationales
    },
    "arc-challenge": {
        "id": "allenai/ai2_arc",
        "task": "reasoning",
        "config": "ARC-Challenge",
        "max_rows": None,            # ~1.1k train rows
    },
    "arc-easy": {
        "id": "allenai/ai2_arc",
        "task": "reasoning",
        "config": "ARC-Easy",        # ~2.25k more rows, same schema
        "max_rows": None,
    },
    "commonsense-qa": {
        "id": "tau/commonsense_qa",
        "task": "reasoning",
        "max_rows": None,            # ~9.7k train rows; test split has no labels
    },
}

def prepare_folders():
    for key in REPOS:
        os.makedirs(f"{DATASET_FOLDER}/{key}", exist_ok=True)


_SPLIT_TOKEN = re.compile(r"[^a-z0-9]")


def _split_tokens(path):
    """Path -> the set of word-ish tokens in it, so "latest" never reads as
    "test" and "contest/" never reads as a test split."""
    return set(t for t in _SPLIT_TOKEN.split(path.lower()) if t)


def wanted_split(path):
    """True if this remote path belongs to a split we keep."""
    tokens = _split_tokens(path)
    if "test" in tokens:
        return False
    return bool(tokens & set(KEEP_SPLITS))


def download_parquet_files(key, force=False):
    """Fast path: the repo already ships parquet, so just copy it across.

    Returns the list of local paths, or None if the repo has no parquet files
    (csv/json/loading-script repos fall through to the datasets path).
    """
    repo = REPOS[key]
    all_files = [
        f
        for f in list_repo_files(repo["id"], repo_type=REPO_TYPE)
        if f.endswith(".parquet")
    ]

    if not all_files:
        return None

    files_to_download = [f for f in all_files if wanted_split(f)]
    if not files_to_download:
        # Some repos don't put the split in the filename; take everything.
        files_to_download = all_files

    if repo.get("config"):
        scoped = [f for f in files_to_download if repo["config"] in f]
        if scoped:
            files_to_download = scoped

    print(f"Found {len(all_files)} parquet files, keeping {len(files_to_download)}.")

    written = []
    for filepath in tqdm(files_to_download, desc=f"Downloading {key}"):
        dest_path = f"{DATASET_FOLDER}/{key}/{os.path.basename(filepath)}"

        if os.path.exists(dest_path) and not force:
            written.append(dest_path)
            continue

        downloaded_path = hf_hub_download(
            repo_id=repo["id"],
            filename=filepath,
            repo_type=REPO_TYPE,
        )

        with open(downloaded_path, "rb") as src, open(dest_path, "wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)

        written.append(dest_path)

    return written


_PLAIN_READERS = {
    ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".json": lambda p: pd.read_json(p),
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
}


def download_plain_files(key, force=False):
    """Repos that ship json/jsonl/csv instead of parquet.

    This path exists because `datasets` is not installed in this environment,
    so download_via_datasets() raised ModuleNotFoundError for every non-parquet
    source and main() swallowed it as "FAILED <key>". Five configured sources
    (dolly, dialogsum, metamath, sql-create-context, and the now-gated samsum)
    were silently absent from the mixture as a result -- the loader only warns
    about tasks with *no* data at all, and each of those tasks had at least one
    parquet source keeping it alive.

    Reads each file named in REPOS[key]["files"] with pandas and writes it out
    as parquet, so everything downstream sees the same format the fast path
    produces. Returns None if the repo declares no explicit file list.
    """
    repo = REPOS[key]
    files = repo.get("files")
    if not files:
        return None

    written = []
    for split, remote in files.items():
        dest_path = f"{DATASET_FOLDER}/{key}/{split}.parquet"
        if os.path.exists(dest_path) and not force:
            print(f"  {split}: exists, skipping")
            written.append(dest_path)
            continue

        suffix = os.path.splitext(remote)[1].lower()
        reader = _PLAIN_READERS.get(suffix)
        if reader is None:
            print(f"  {split}: no reader for {suffix}, skipping {remote}")
            continue

        local = hf_hub_download(repo_id=repo["id"], filename=remote,
                                repo_type=REPO_TYPE)
        df = reader(local)

        # Cap here rather than in subsample_in_place(): that helper keys off
        # "train" appearing in the filename, and these are already named by
        # split, so a capped validation file would slip through uncapped.
        cap = repo.get("max_rows")
        if cap and split == "train" and len(df) > cap:
            print(f"  {split}: capping {len(df)} -> {cap} rows")
            df = df.sample(n=cap, random_state=1337).reset_index(drop=True)

        # Object columns holding dicts/lists survive the parquet round-trip
        # fine; columns that are entirely null do not, so drop them.
        df = df.dropna(axis=1, how="all")
        df.to_parquet(dest_path)
        print(f"  {split}: {len(df):,} rows -> {dest_path}")
        written.append(dest_path)

    return written or None


def download_via_datasets(key, force=False):
    """Fallback: load through `datasets` and write parquet ourselves.

    Needed for repos that ship csv/json instead of parquet, and for anything
    with a max_rows cap, since subsampling means we have to rewrite the file.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            f"{key} ships no parquet and has no 'files' entry in REPOS, so it "
            f"needs the `datasets` package, which is not installed. Either add "
            f"an explicit files={{split: path}} mapping for it (preferred -- see "
            f"download_plain_files) or `pip install datasets`.") from e

    repo = REPOS[key]
    print(f"Loading {repo['id']} through the datasets library...")

    ds = load_dataset(repo["id"], repo.get("config"))
    written = []

    for split in ds:
        if not any(s in split.lower() for s in KEEP_SPLITS):
            continue

        part = ds[split]
        cap = repo.get("max_rows")
        if cap and len(part) > cap:
            part = part.shuffle(seed=1337).select(range(cap))
            print(f"  {split}: capped {len(ds[split])} -> {len(part)} rows")

        dest_path = f"{DATASET_FOLDER}/{key}/{split}.parquet"
        if os.path.exists(dest_path) and not force:
            print(f"  {split}: exists, skipping")
            written.append(dest_path)
            continue

        part.to_parquet(dest_path)
        written.append(dest_path)

    return written


def subsample_in_place(key, paths):
    """Apply max_rows to files copied by the fast path."""
    cap = REPOS[key].get("max_rows")
    if not cap:
        return paths

    import pyarrow.parquet as pq

    train_paths = [p for p in paths if "train" in os.path.basename(p).lower()]
    if not train_paths:
        train_paths = paths

    total = sum(pq.ParquetFile(p).metadata.num_rows for p in train_paths)
    if total <= cap:
        return paths

    import pandas as pd

    print(f"  capping {total} -> {cap} rows")
    keep_frac = cap / total
    for p in train_paths:
        df = pd.read_parquet(p)
        n = max(1, int(len(df) * keep_frac))
        df.sample(n=n, random_state=1337).reset_index(drop=True).to_parquet(p)

    return paths


def describe(paths):
    """Row counts and column names, so the loader knows what it's mapping."""
    import pyarrow.parquet as pq

    rows = 0
    columns = []
    for p in paths:
        try:
            meta = pq.ParquetFile(p)
            rows += meta.metadata.num_rows
            if not columns:
                columns = [f.name for f in meta.schema_arrow]
        except Exception as e:
            print(f"  could not read {p}: {e}")
    return rows, columns


def download_dataset(key, force=False):
    repo = REPOS[key]
    print(f"Listing {key} ({repo['id']}) files...")

    # Always try parquet first, cap or no cap. Gating this on max_rows sent
    # every capped source through load_dataset(), which materialises the whole
    # repo before subsampling -- soda downloaded 1.5M rows to keep 60k. The cap
    # is applied locally afterwards by subsample_in_place().
    paths = None
    # An explicit file list means the repo has no usable parquet; go straight
    # to the pandas path instead of listing the repo and guessing.
    if repo.get("files"):
        paths = download_plain_files(key, force=force)
    else:
        try:
            paths = download_parquet_files(key, force=force)
        except Exception as e:
            print(f"  parquet path failed ({e}); falling back to datasets")
        if paths is not None:
            paths = subsample_in_place(key, paths)

    if paths is None:
        paths = download_via_datasets(key, force=force)

    rows, columns = describe(paths)
    print(f"  {key}: {rows:,} rows | columns: {columns}")

    return {
        "id": repo["id"],
        "task": repo["task"],
        "folder": f"{DATASET_FOLDER}/{key}",
        "files": [os.path.basename(p) for p in paths],
        "rows": rows,
        "columns": columns,
    }


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_manifest(manifest):
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, MANIFEST_FILE)


def main():
    parser = argparse.ArgumentParser(description="Download SFT datasets")
    parser.add_argument("--only", nargs="*", help="download only these keys")
    parser.add_argument("--task", nargs="*", help="download only these tasks")
    parser.add_argument("--all", action="store_true",
                        help="include entries marked enabled=False")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--list", action="store_true", help="list configured datasets")
    args = parser.parse_args()

    if args.list:
        for key, repo in REPOS.items():
            state = "" if repo.get("enabled", True) else "  (disabled)"
            cap = f", cap {repo['max_rows']:,}" if repo.get("max_rows") else ""
            print(f"{key:24s} {repo['task']:15s} {repo['id']}{cap}{state}")
        return

    keys = list(REPOS)
    if args.only:
        unknown = [k for k in args.only if k not in REPOS]
        if unknown:
            raise SystemExit(f"Unknown keys: {unknown}")
        keys = args.only
    if args.task:
        keys = [k for k in keys if REPOS[k]["task"] in args.task]
    if not args.all and not args.only:
        keys = [k for k in keys if REPOS[k].get("enabled", True)]

    prepare_folders()
    manifest = load_manifest()

    for key in keys:
        print(f"\n--- Downloading dataset: {key} ---")
        try:
            manifest[key] = download_dataset(key, force=args.force)
            save_manifest(manifest)
        except Exception as e:
            print(f"FAILED {key}: {e}")

    print(f"\n--- Summary ({MANIFEST_FILE}) ---")
    by_task = {}
    for key, entry in manifest.items():
        by_task.setdefault(entry["task"], 0)
        by_task[entry["task"]] += entry["rows"]
        print(f"{key:24s} {entry['task']:15s} {entry['rows']:>9,} rows")

    total = sum(by_task.values())
    print()
    for task, rows in sorted(by_task.items(), key=lambda kv: -kv[1]):
        share = 100 * rows / total if total else 0
        print(f"{task:15s} {rows:>9,} rows  ({share:5.1f}% of raw pool)")
    print(f"{'TOTAL':15s} {total:>9,} rows")
    print("\nRaw shares are not the training mixture -- weight per task in the loader.")


if __name__ == "__main__":
    main()