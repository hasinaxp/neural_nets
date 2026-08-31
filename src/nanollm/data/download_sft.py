import argparse
import json
import os

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
#   max_rows  subsample cap (None = take everything). Applied per split.
#   config    HF config/subset name, when the repo has more than one
#   enabled   set False to skip by default
REPOS = {
    # ---- chat / small talk -------------------------------------------------
    "smoltalk": {
        "id": "HuggingFaceTB/smol-smoltalk",
        "task": "chat",
        "max_rows": None,
    },
    "no-robots": {
        "id": "HuggingFaceH4/no_robots",
        "task": "chat",
        "max_rows": None,
    },
    "soda": {
        "id": "allenai/soda",
        "task": "chat",
        "max_rows": 60_000,          # 1.5M rows upstream; way past what 110M needs
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
    "samsum": {
        "id": "Samsung/samsum",
        "task": "summarization",
        "max_rows": None,
    },
    "dialogsum": {
        "id": "knkarthick/dialogsum",
        "task": "summarization",
        "max_rows": None,
    },
    "xsum": {
        "id": "EdinburghNLP/xsum",
        "task": "summarization",
        "max_rows": 40_000,
        "enabled": False,            # long inputs; enable once 1024 ctx is not the cap
    },
    # ---- text to SQL -------------------------------------------------------
    "sql-create-context": {
        "id": "b-mc2/sql-create-context",
        "task": "sql",
        "max_rows": None,            # 78.6k, schema is in the prompt
    },
    "synthetic-text-to-sql": {
        "id": "gretelai/synthetic_text_to_sql",
        "task": "sql",
        "max_rows": None,            # ~106k; filter complexity at load time
    },
    # ---- instruction variety ----------------------------------------------
    "dolly": {
        "id": "databricks/databricks-dolly-15k",
        "task": "instruct",
        "max_rows": None,
    },
    # ---- math (chain-of-thought word problems) -----------------------------
    "gsm8k": {
        "id": "openai/gsm8k",
        "task": "math",
        "config": "main",            # repo also has a "socratic" variant
        "max_rows": None,            # ~7.5k train rows
    },
    # ---- reasoning (grounded multiple-choice) -------------------------------
    "arc-challenge": {
        "id": "allenai/ai2_arc",
        "task": "reasoning",
        "config": "ARC-Challenge",   # repo also has ARC-Easy
        "max_rows": None,            # ~1.1k train rows
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


def wanted_split(path):
    """True if this remote path belongs to a split we keep."""
    lowered = path.lower()
    if "test" in lowered and not any(s in lowered for s in ("latest",)):
        return False
    return any(s in lowered for s in KEEP_SPLITS)


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


def download_via_datasets(key, force=False):
    """Fallback: load through `datasets` and write parquet ourselves.

    Needed for repos that ship csv/json instead of parquet, and for anything
    with a max_rows cap, since subsampling means we have to rewrite the file.
    """
    from datasets import load_dataset

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

    paths = None
    if not repo.get("max_rows"):
        # No cap, so a straight parquet copy is enough.
        try:
            paths = download_parquet_files(key, force=force)
        except Exception as e:
            print(f"  parquet path failed ({e}); falling back to datasets")

    if paths is None:
        paths = download_via_datasets(key, force=force)
    else:
        paths = subsample_in_place(key, paths)

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