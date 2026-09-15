"""Mid-training corpus: curated textbook prose.

Pretraining is 10B tokens of web crawl and synthetic prose, and it shows. The
finished base model writes fluent English and then says Pride and Prejudice was
written by Elizabeth Gaskell -- fluency came from the crawl and so did the
confident wrong answer. A model this small cannot be argued out of that with
more of the same data; the standard fix is a short final stage on a small,
curated, high-quality corpus (variously "mid-training", "annealing", or a decay
phase), run at a low LR so it moves the distribution without restarting the run.

What this stage can and cannot do is worth being blunt about:

* It **can** raise quality inside the subjects the books cover -- physics,
  chemistry, biology, maths, economics, history, civics -- and it pulls the
  model's default register toward expository prose that states things once and
  correctly, rather than web text that repeats and hedges.
* It **cannot** delete what pretraining learned. Nothing is unlearned here, it
  is outweighed, and only where the textbooks actually have coverage. English
  literature trivia is not in OpenStax or NCERT, so "who wrote Pride and
  Prejudice" is not what this stage fixes.
* At 169M parameters knowledge capacity is the binding constraint. This buys a
  better-calibrated model of the things it was taught, not a bigger one.

Output is parquet with a ``text`` column written into a raw directory, which is
exactly what ``scripts/prepare_data.py`` already consumes -- so the tokenizer,
the shard writer and the loader are unchanged, and the stage is just a second
(much smaller) shard set to train on.

    python scripts/download_midtrain.py --out-dir dataset/raw_midtrain
    python scripts/prepare_data.py --raw-dir dataset/raw_midtrain \\
        --out-dir dataset/tokens_midtrain --no-wikipedia
    python -m nanollm.train.pretrain --config configs/midtrain.yaml \\
        --init-from artifacts/pretrain_checkpoint_latest.pt
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Optional

import pandas as pd

__all__ = [
    "MIDTRAIN_SOURCES", "MidtrainSource", "download_midtrain",
    "build_midtrain_corpus", "clean_textbook_text", "extract_pdf_text",
    "iter_openstax", "iter_ncert", "iter_libretexts",
]


@dataclass
class MidtrainSource:
    repo_id: str
    kind: str                       # txt | pdf | parquet
    note: str = ""
    # Only files whose path matches are kept. NCERT ships Hindi-medium editions
    # of every subject alongside the English ones; this is an English-only
    # corpus and the tokenizer is an English BPE, so the Devanagari half would
    # be spent teaching the model to model bytes it will never generate.
    include: Optional[str] = None
    exclude: Optional[str] = None
    text_column: str = "text"
    subdir: str = ""


MIDTRAIN_SOURCES: dict[str, MidtrainSource] = {
    "openstax": MidtrainSource(
        "crumb/openstax-text", kind="txt",
        note="76 OpenStax college textbooks, PDF-extracted"),
    "ncert": MidtrainSource(
        "AdithyaSNair/ncert-textbooks-10-12", kind="pdf",
        include=r"(English_Medium|Languages/English)",
        exclude=r"(Hindi_Medium|Hindi_[A-Za-z]+/)",
        note="NCERT classes 10-12, English medium"),
    "libretexts-chem": MidtrainSource(
        "Hack90/libre_chem_textbooks", kind="parquet", text_column="html",
        note="LibreTexts chemistry pages"),
}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
# Everything here arrives from a PDF, which means the text is laid out for a
# page rather than for a reader: lines break mid-sentence at the column edge,
# the running header and the page number sit inside the text, and every
# equation has been flattened into a row of loose symbols. Feeding that in raw
# teaches the model to break lines mid-sentence and to emit orphaned operators.

# A line that is mostly not letters: an equation fragment, a table row, a rule
# of dots in a contents list. Formula-heavy books are the reason this exists.
_ALPHA = re.compile(r"[A-Za-z]")
# Page furniture: a bare number, a number with a chapter name, "Page 12 of 40".
_PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d{1,4}\s*(of\s+\d{1,4})?\s*$", re.I)
# A heading that is entirely uppercase and short -- "EXAMPLE 2.2", "SUMMARY".
_SHOUTED = re.compile(r"^[^a-z]{1,40}$")
# Sentence-ending punctuation, used to decide whether a line wrapped or ended.
_ENDS_SENTENCE = re.compile(r"[.!?:;\"')\]]\s*$")
# A hyphen at a line break splitting one word across two lines.
_SOFT_HYPHEN = re.compile(r"(\w)-\s*$")

MIN_LINE_ALPHA_FRAC = 0.55     # below this a line is an equation or a table row
MIN_PARAGRAPH_CHARS = 200      # shorter survivors are captions and headings
MIN_LATIN_FRAC = 0.90          # English-only corpus; a guard, not a filter
HEADER_REPEAT_FRAC = 0.25      # a line on >25% of pages is a running header


def _alpha_frac(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    return len(_ALPHA.findall(stripped)) / len(stripped)


def _latin_frac(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def _running_headers(pages: list[str]) -> set[str]:
    """Lines repeated across a quarter of the pages: header, footer, book title.

    Detected per document rather than by pattern, because every publisher picks
    a different piece of furniture to repeat and a regex for "Physics" would
    also delete the word in the body text.
    """
    if len(pages) < 8:
        return set()
    counts: Counter = Counter()
    for page in pages:
        # set(): a header counts once per page however often it appears.
        counts.update({line.strip() for line in page.splitlines()
                       if 0 < len(line.strip()) <= 80})
    cutoff = max(3, int(HEADER_REPEAT_FRAC * len(pages)))
    return {line for line, n in counts.items() if n >= cutoff}


def clean_textbook_text(text: str, drop_lines: Optional[set] = None) -> str:
    """Page-laid-out text -> paragraphs.

    Drops page furniture and equation debris, then rejoins the lines the PDF
    broke at the column edge: a line that does not end a sentence is a wrap and
    is glued to the next one.
    """
    drop_lines = drop_lines or set()
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            kept.append("")                      # paragraph break, preserved
            continue
        if line in drop_lines or _PAGE_NUMBER.match(line):
            continue
        if _SHOUTED.match(line) or _alpha_frac(line) < MIN_LINE_ALPHA_FRAC:
            continue
        kept.append(line)

    # Rejoin wrapped lines into paragraphs.
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= MIN_PARAGRAPH_CHARS and _latin_frac(text) >= MIN_LATIN_FRAC:
            paragraphs.append(text)
        buffer.clear()

    for line in kept:
        if not line:
            flush()
            continue
        if buffer and _SOFT_HYPHEN.search(buffer[-1]):
            buffer[-1] = _SOFT_HYPHEN.sub(r"\1", buffer[-1]) + line
            continue
        buffer.append(line)
        if _ENDS_SENTENCE.search(line) and len(" ".join(buffer)) > 600:
            flush()                              # keep paragraphs bounded
    flush()
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def extract_pdf_text(path: str) -> list[str]:
    """Per-page text. Kept as a list so running headers can be found."""
    import pymupdf                                # optional: only midtrain needs it

    try:
        doc = pymupdf.open(path)
    except Exception as e:                        # a truncated or encrypted file
        print(f"  could not open {os.path.basename(path)}: {e}")
        return []
    pages = []
    try:
        for page in doc:
            try:
                pages.append(page.get_text())
            except Exception:
                continue
    finally:
        doc.close()
    return pages


def _matches(path: str, source: MidtrainSource) -> bool:
    if source.include and not re.search(source.include, path):
        return False
    if source.exclude and re.search(source.exclude, path):
        return False
    return True


def iter_openstax(root: str, source: MidtrainSource) -> Iterator[tuple[str, str]]:
    for path in sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)):
        if not _matches(path, source):
            continue
        with open(path, errors="ignore") as f:
            raw = f.read()
        # One .txt per book, already page-concatenated; split it back into
        # pages on form feeds where present so headers can still be spotted.
        pages = raw.split("\f") if "\f" in raw else [raw]
        cleaned = clean_textbook_text(raw, _running_headers(pages))
        if cleaned:
            yield os.path.basename(path), cleaned


def iter_ncert(root: str, source: MidtrainSource) -> Iterator[tuple[str, str]]:
    for path in sorted(glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True)):
        if not _matches(path, source):
            continue
        pages = extract_pdf_text(path)
        if not pages:
            continue
        cleaned = clean_textbook_text("\n".join(pages), _running_headers(pages))
        if cleaned:
            yield os.path.relpath(path, root), cleaned


_TAGS = re.compile(r"<[^>]+>")


def iter_libretexts(root: str, source: MidtrainSource) -> Iterator[tuple[str, str]]:
    for path in sorted(glob.glob(os.path.join(root, "**", "*.parquet"),
                                 recursive=True)):
        frame = pd.read_parquet(path)
        column = (source.text_column if source.text_column in frame.columns
                  else frame.columns[0])
        for i, value in enumerate(frame[column].astype(str)):
            cleaned = clean_textbook_text(_TAGS.sub(" ", value))
            if cleaned:
                yield f"{os.path.basename(path)}:{i}", cleaned


READERS = {"txt": iter_openstax, "pdf": iter_ncert, "parquet": iter_libretexts}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def download_midtrain(cache_dir: str = "dataset/midtrain_raw",
                      sources: Optional[list[str]] = None) -> dict[str, str]:
    """Fetch each source's repo. Returns {name: local path}."""
    from huggingface_hub import snapshot_download

    paths = {}
    for name in (sources or list(MIDTRAIN_SOURCES)):
        source = MIDTRAIN_SOURCES[name]
        target = os.path.join(cache_dir, name)
        print(f"[{name}] {source.repo_id} -- {source.note}")
        try:
            paths[name] = snapshot_download(
                repo_id=source.repo_id, repo_type="dataset", local_dir=target)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    return paths


def build_midtrain_corpus(cache_dir: str = "dataset/midtrain_raw",
                          out_dir: str = "dataset/raw_midtrain",
                          sources: Optional[list[str]] = None,
                          rows_per_file: int = 20000) -> str:
    """Clean every source into parquet that prepare_data.py can tokenize."""
    os.makedirs(out_dir, exist_ok=True)
    for name in (sources or list(MIDTRAIN_SOURCES)):
        source = MIDTRAIN_SOURCES[name]
        root = os.path.join(cache_dir, name)
        if not os.path.isdir(root):
            print(f"[{name}] not downloaded, skipping")
            continue
        reader = READERS[source.kind]
        rows, docs, chars, part = [], 0, 0, 0
        print(f"[{name}] reading {root}")
        for doc_id, text in reader(root, source):
            rows.append({"text": text, "doc": doc_id, "source": name})
            docs += 1
            chars += len(text)
            if len(rows) >= rows_per_file:
                _write_part(out_dir, name, part, rows)
                part, rows = part + 1, []
        if rows:
            _write_part(out_dir, name, part, rows)
        print(f"[{name}] {docs:,} documents, {chars/1e6:.1f}M chars "
              f"(~{chars/4/1e6:.1f}M tokens)")
    return out_dir


def _write_part(out_dir: str, name: str, part: int, rows: list[dict]) -> None:
    """One folder per source.

    ``sources.list_parquet_files`` only walks one level down and
    ``sources.source_of`` reads the source name off the parent directory, so a
    file written flat into out_dir is silently invisible to prepare_data --
    it reports 0 documents and writes 0 shards rather than failing.
    """
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}-{part:03d}.parquet")
    pd.DataFrame(rows).to_parquet(path)
    print(f"  wrote {path} ({len(rows):,} rows)")
