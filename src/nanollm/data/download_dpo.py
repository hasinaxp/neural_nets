"""Download preference-pair datasets for DPO.

Same layout as the SFT side: everything lands in dataset/dpo/<key>/*.parquet
with a manifest.json describing what arrived, so dpo_dataset.py can normalize
without re-deriving five different schemas.

Repo ids on the Hub do get renamed and gated. If one 404s, the run continues
with the rest and the failure is printed -- check `--list` against the Hub
rather than assuming the entry is wrong.
"""

import argparse
import json
import os

from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm


DATASET_FOLDER = "dataset/dpo"
REPO_TYPE = "dataset"
MANIFEST_FILE = f"{DATASET_FOLDER}/manifest.json"

KEEP_SPLITS = ("train", "validation", "valid", "dev", "test")

# Preference data is small enough that we keep the published test splits here,
# unlike the SFT side -- several of these repos ship *only* train + test.

#   id          HF repo id
#   task        what the pairs teach; the loader weights the mixture by this
#   schema      which normalizer in dpo_dataset.py handles the columns
#   max_rows    subsample cap (None = everything), applied per split
#   keep_files  only take files whose name contains this (ultrafeedback ships
#               three unrelated split families in one repo)
#   enabled     False to skip unless asked for by name
REPOS = {
    # ---- general helpfulness, GPT-4-judged -------------------------------
    "ultrafeedback": {
        "id": "HuggingFaceH4/ultrafeedback_binarized",
        "task": "helpful",
        "schema": "messages_pair",
        "max_rows": 40_000,
        "keep_files": "prefs",     # skip the _sft and _gen split families
    },
    # ---- helpfulness + harmlessness, human-judged -------------------------
    "hh-rlhf": {
        "id": "Anthropic/hh-rlhf",
        "task": "harmless",
        "schema": "hh_transcript",
        "max_rows": 30_000,
    },
    # ---- instruction-following quality ------------------------------------
    "orca-dpo": {
        "id": "argilla/distilabel-intel-orca-dpo-pairs",
        "task": "instruct",
        "schema": "system_input_pair",
        "max_rows": None,          # ~12.9k rows
    },
    # ---- community-voted answers; noisy, long, off by default -------------
    "shp": {
        "id": "stanfordnlp/SHP",
        "task": "helpful",
        "schema": "shp",
        "max_rows": 20_000,
        "enabled": False,          # reddit prose; long and stylistically odd
    },
    "rm-static": {
        "id": "Dahoas/rm-static",
        "task": "helpful",
        "schema": "hh_prompt_pair",
        "max_rows": 20_000,
        "enabled": False,          # largely overlaps hh-rlhf
    },
}


def prepare_folders():
    for key in REPOS:
        os.makedirs(f"{DATASET_FOLDER}/{key}", exist_ok=True)


def wanted_split(path):
    return any(s in path.lower() for s in KEEP_SPLITS)


def download_parquet_files(key, force=False):
    """Fast path: the repo already ships parquet, so copy it across.

    Returns None when the repo has no parquet at all (hh-rlhf ships jsonl.gz),
    which sends the caller to the `datasets` fallback.
    """
    repo = REPOS[key]
    all_files = [f for f in list_repo_files(repo["id"], repo_type=REPO_TYPE)
                 if f.endswith(".parquet")]
    if not all_files:
        return None

    files = [f for f in all_files if wanted_split(f)] or all_files
    if repo.get("keep_files"):
        scoped = [f for f in files if repo["keep_files"] in f]
        if not scoped:
            raise RuntimeError(
                f"{key}: no file matched keep_files={repo['keep_files']!r} "
                f"among {files[:6]}")
        files = scoped
    if repo.get("config"):
        files = [f for f in files if repo["config"] in f] or files

    print(f"Found {len(all_files)} parquet files, keeping {len(files)}.")

    written = []
    for filepath in tqdm(files, desc=f"Downloading {key}"):
        dest = f"{DATASET_FOLDER}/{key}/{os.path.basename(filepath)}"
        if os.path.exists(dest) and not force:
            written.append(dest)
            continue
        src_path = hf_hub_download(repo_id=repo["id"], filename=filepath,
                                   repo_type=REPO_TYPE)
        with open(src_path, "rb") as src, open(dest, "wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
        written.append(dest)
    return written


def download_via_datasets(key, force=False):
    """Fallback for repos that ship jsonl/csv, and for anything with a cap."""
    from datasets import load_dataset

    repo = REPOS[key]
    print(f"Loading {repo['id']} through the datasets library...")
    ds = load_dataset(repo["id"], repo.get("config"))

    written = []
    for split in ds:
        lowered = split.lower()
        if not any(s in lowered for s in KEEP_SPLITS):
            continue
        if repo.get("keep_files") and repo["keep_files"] not in lowered:
            continue

        part = ds[split]
        cap = repo.get("max_rows")
        if cap and len(part) > cap:
            part = part.shuffle(seed=1337).select(range(cap))
            print(f"  {split}: capped {len(ds[split])} -> {len(part)} rows")

        dest = f"{DATASET_FOLDER}/{key}/{split}.parquet"
        if os.path.exists(dest) and not force:
            print(f"  {split}: exists, skipping")
            written.append(dest)
            continue
        part.to_parquet(dest)
        written.append(dest)
    return written


def subsample_in_place(key, paths):
    """Apply max_rows to files copied by the fast path."""
    cap = REPOS[key].get("max_rows")
    if not cap:
        return paths

    import pandas as pd
    import pyarrow.parquet as pq

    train_paths = [p for p in paths if "train" in os.path.basename(p).lower()] or paths
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in train_paths)
    if total <= cap:
        return paths

    print(f"  capping {total:,} -> {cap:,} rows")
    keep_frac = cap / total
    for p in train_paths:
        df = pd.read_parquet(p)
        n = max(1, int(len(df) * keep_frac))
        df.sample(n=n, random_state=1337).reset_index(drop=True).to_parquet(p)
    return paths


def describe(paths):
    import pyarrow.parquet as pq

    rows, columns = 0, []
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
        "schema": repo["schema"],
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
    parser = argparse.ArgumentParser(description="Download DPO preference datasets")
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
            print(f"{key:16s} {repo['task']:10s} {repo['schema']:18s} "
                  f"{repo['id']}{cap}{state}")
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
        by_task[entry["task"]] = by_task.get(entry["task"], 0) + entry["rows"]
        print(f"{key:16s} {entry['task']:10s} {entry['rows']:>9,} pairs")

    total = sum(by_task.values())
    print()
    for task, rows in sorted(by_task.items(), key=lambda kv: -kv[1]):
        share = 100 * rows / total if total else 0
        print(f"{task:10s} {rows:>9,} pairs  ({share:5.1f}% of raw pool)")
    print(f"{'TOTAL':10s} {total:>9,} pairs")
    print("\nRaw shares are not the training mixture -- weight per task in the loader.")


if __name__ == "__main__":
    main()
