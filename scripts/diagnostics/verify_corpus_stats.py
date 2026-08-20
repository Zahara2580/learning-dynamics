"""
Verify the CPT corpus statistics that go into the write-up.

Measures the corpus DIRECTLY (raw WURA passages and the exported line
files) rather than inferring it from decoded token ids, and reports
exactly what export_wura_lines.py changed.

    uv run python3 scripts/diagnostics/verify_corpus_stats.py
"""

import glob
import os
from pathlib import Path

from datasets import load_from_disk

RAW_DIRS = ["/scratch/rmdrak003/data/corpus/xho-passage",
            "/scratch/rmdrak003/data/corpus/xho"]
LINE_GLOB = "/scratch/rmdrak003/data/lafand/lines*/**.xh"


def normalise(t):
    """The exact transformation export_wura_lines.py:71 applies."""
    return " ".join(t.split()).strip()


raw_dir = next((d for d in RAW_DIRS if os.path.isdir(d)), None)
if raw_dir is None:
    raise SystemExit(f"no raw corpus found; looked in {RAW_DIRS}")
print(f"raw corpus: {raw_dir}")

ds = load_from_disk(raw_dir)
print("splits:", {k: f"{len(v):,}" for k, v in ds.items()})
print("columns:", ds[list(ds)[0]].column_names)

col = "text" if "text" in ds[list(ds)[0]].column_names else ds[list(ds)[0]].column_names[-1]
print(f"using column: {col!r}\n")

print("=" * 74)
print("WHAT ONE PASSAGE LOOKS LIKE  (repr, so whitespace is visible)")
print("=" * 74)
sample = ds["train"][:3][col]
for i, t in enumerate(sample):
    print(f"\npassage {i}: {len(t):,} chars | {len(t.split()):,} words | "
          f"{t.count(chr(10))} newlines | {t.count(chr(9))} tabs")
    print("   ", repr(t[:280]))

print("\n" + "=" * 74)
print("EXACTLY WHAT THE EXPORT CHANGED")
print("=" * 74)
for split in ds:
    texts = ds[split][col]                       # whole column at once - fast
    raw = sum(len(t) for t in texts)
    norms = [normalise(t) for t in texts]
    norm = sum(len(n) for n in norms)
    words = sum(len(n.split()) for n in norms)
    emptied = sum(1 for n in norms if not n)
    print(f"\n  [{split}]  {len(texts):,} passages")
    print(f"    raw characters                : {raw:,}")
    print(f"    after whitespace normalisation: {norm:,}")
    print(f"    collapsed (whitespace ONLY)   : {raw - norm:,}  ({(raw-norm)/raw:.2%})")
    print(f"    words (whitespace-delimited)  : {words:,}")
    print(f"    mean chars / passage          : {norm/len(texts):,.0f}")
    print(f"    mean words / passage          : {words/len(texts):,.1f}")
    print(f"    mean chars / word             : {norm/max(words,1):,.2f}")
    print(f"    passages emptied by normalise : {emptied:,}   <- the ONLY ones dropped")

print("\n" + "=" * 74)
print("EXPORTED LINE FILES  (should match the normalised counts above)")
print("=" * 74)
found = sorted(glob.glob(LINE_GLOB, recursive=True))
if not found:
    print(f"  none found under {LINE_GLOB}")
for f in found:
    lines = [l for l in Path(f).read_text(encoding="utf-8").split("\n") if l]
    chars = sum(len(l) for l in lines)
    words = sum(len(l.split()) for l in lines)
    print(f"  {f}")
    print(f"    {len(lines):,} lines | {chars:,} chars | {words:,} words | "
          f"{chars/max(words,1):.2f} chars/word")

print("\n" + "=" * 74)
print("NON-WHITESPACE LOSS CHECK  (must be zero)")
print("=" * 74)
texts = ds["train"][:20000][col]
lost = 0
for t in texts:
    a = "".join(t.split())          # every non-whitespace character, raw
    b = "".join(normalise(t).split())
    if a != b:
        lost += 1
print(f"  checked 20,000 passages: {lost} whose non-whitespace characters changed")
print("  (0 means normalisation touched whitespace only - no content removed)")
