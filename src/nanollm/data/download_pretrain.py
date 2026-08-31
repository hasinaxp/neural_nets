"""Download the pretraining corpus from the Hugging Face hub.

Replaces a hardcoded list of file indices with a size budget per source. The
old approach broke whenever a repo was re-sharded -- index 142 is a different
file this month than it was last month -- and it copied every file a second
time into ``dataset/raw`` instead of linking the hub cache.

    python scripts/download_data.py pretrain --budget-gb 40
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

__all__ = ["SOURCES", "download_corpus", "Source"]


@dataclass
class Source:
    repo_id: str
    # Share of the total byte budget. fineweb-edu is weighted highest: for a
    # model this small, filtered educational text buys more per token than raw
    # crawl, and cosmopedia's synthetic prose keeps the register varied.
    weight: float
    prefix: str = ""          # only take files under this path
    note: str = ""


SOURCES: dict[str, Source] = {
    "fineweb-edu": Source(
        "HuggingFaceFW/fineweb-edu", weight=0.50, prefix="sample/10BT",
        note="classifier-filtered educational web text"),
    "cosmopedia": Source(
        "HuggingFaceTB/cosmopedia", weight=0.30,
        note="synthetic textbooks and stories"),
    "fineweb": Source(
        "HuggingFaceFW/fineweb", weight=0.20, prefix="sample/10BT",
        note="general web crawl, for register diversity"),
}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_corpus(
    out_dir: str = "dataset/raw",
    budget_gb: float = 40.0,
    sources: Optional[list[str]] = None,
    symlink: bool = True,
    log=print,
) -> dict[str, list[str]]:
    """Fetch parquet files up to a per-source share of ``budget_gb``.

    Files are symlinked out of the hub cache by default rather than copied --
    a full copy doubles the disk cost for no benefit. Pass ``symlink=False``
    on filesystems that cannot link (some network mounts).
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    names = sources or list(SOURCES)
    total_weight = sum(SOURCES[n].weight for n in names) or 1.0
    budget_bytes = budget_gb * 1024 ** 3
    fetched: dict[str, list[str]] = {}

    for name in names:
        source = SOURCES[name]
        share = budget_bytes * source.weight / total_weight
        dest_dir = os.path.join(out_dir, name)
        os.makedirs(dest_dir, exist_ok=True)

        log(f"\n--- {name} ({source.note}) ---")
        log(f"    repo {source.repo_id} | budget {_human(share)}")

        # Ask for sizes up front so the budget is enforced before downloading,
        # not after.
        try:
            info = api.repo_info(source.repo_id, repo_type="dataset", files_metadata=True)
            entries = [
                (s.rfilename, s.size or 0)
                for s in info.siblings
                if s.rfilename.endswith(".parquet")
                and (not source.prefix or s.rfilename.startswith(source.prefix))
            ]
        except Exception as e:
            log(f"    could not list {source.repo_id}: {e}")
            fetched[name] = []
            continue

        entries.sort()
        if not entries:
            log(f"    no parquet files under {source.prefix or '/'}")
            fetched[name] = []
            continue

        taken, used = [], 0
        for filename, size in entries:
            if size and used + size > share and taken:
                break
            used += size
            taken.append(filename)

        log(f"    {len(taken)} of {len(entries)} files, ~{_human(used)}")

        paths = []
        try:
            from tqdm import tqdm
            iterator = tqdm(taken, desc=f"  {name}", unit="file")
        except ImportError:
            iterator = taken

        for filename in iterator:
            cached = hf_hub_download(
                repo_id=source.repo_id, filename=filename, repo_type="dataset")
            dest = os.path.join(dest_dir, os.path.basename(filename))
            if os.path.exists(dest) or os.path.islink(dest):
                paths.append(dest)
                continue
            if symlink:
                try:
                    os.symlink(os.path.realpath(cached), dest)
                except OSError:
                    shutil.copy2(cached, dest)
            else:
                shutil.copy2(cached, dest)
            paths.append(dest)

        fetched[name] = paths
        log(f"    -> {dest_dir} ({len(paths)} files)")

    on_disk = sum(
        os.path.getsize(os.path.realpath(p))
        for paths in fetched.values() for p in paths if os.path.exists(p))
    log(f"\ncorpus ready: {_human(on_disk)} across "
        f"{sum(len(p) for p in fetched.values())} files in {out_dir}")
    return fetched
