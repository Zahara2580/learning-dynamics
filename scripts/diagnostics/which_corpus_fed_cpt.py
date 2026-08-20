"""
Which line file actually fed CPT: lines/ or lines-passage/?

Decodes the real train.source the models trained on and compares it,
byte-for-byte where possible, against both candidate exports. Also checks
the segment length distribution, which separates the two decisively: a
681k-line export of ~163-char lines produces short segments, while a
69,713-passage export of ~1,581-char passages produces mostly full
512-token ones.

    uv run python3 scripts/diagnostics/which_corpus_fed_cpt.py
"""

import datetime
import os
from pathlib import Path

from transformers import AutoTokenizer

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/data/lafand/t5"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/data/lafand/byt5"),
}
CANDIDATES = {
    "lines":         "/scratch/rmdrak003/data/lafand/lines/train.xh",
    "lines-passage": "/scratch/rmdrak003/data/lafand/lines-passage/train.xh",
}

print("=" * 74)
print("CANDIDATE EXPORTS")
print("=" * 74)
first_line = {}
for name, path in CANDIDATES.items():
    if not os.path.exists(path):
        print(f"  {name:14} MISSING ({path})"); continue
    with open(path, encoding="utf-8") as f:
        first_line[name] = f.readline().rstrip("\n")
    st = os.stat(path)
    n = sum(1 for _ in open(path, encoding="utf-8"))
    mtime = datetime.datetime.fromtimestamp(st.st_mtime)
    print(f"  {name:14} {n:>9,} lines   modified {mtime:%Y-%m-%d %H:%M}")
    print(f"                 first line: {first_line[name][:150]!r}")

for model, (tok_id, ddir) in MODELS.items():
    src = Path(ddir) / "train.source"
    if not src.exists():
        print(f"\n{model}: {src} missing"); continue
    tok = AutoTokenizer.from_pretrained(tok_id)

    print("\n" + "=" * 74)
    print(f"{model}: WHAT THE MODEL ACTUALLY TRAINED ON")
    print("=" * 74)

    # decode the first training example (sentinels stripped)
    with open(src) as f:
        first = [int(x) for x in f.readline().split()]
    decoded = tok.decode(first, skip_special_tokens=True)
    print(f"  train.source line 1, decoded ({len(first)} tokens):")
    print(f"    {decoded[:150]!r}")
    # Span corruption masks ~15% of tokens, so an exact substring test fails
    # even on the correct source. Compare word OVERLAP instead, and - the
    # decisive part - check whether the example runs PAST where each
    # candidate's first line ends.
    dec_words = set(w.strip(".,()|").lower() for w in decoded.split())
    for name, line in first_line.items():
        cand = [w.strip(".,()|").lower() for w in line.split()]
        overlap = sum(1 for w in cand[:40] if w in dec_words) / min(len(cand), 40)
        beyond = ""
        tail = cand[-3:]
        if tail and not any(w in dec_words for w in tail):
            beyond = "  (example does NOT stop where this file's line 1 stops)"
        print(f"    word overlap with {name:14}: {overlap:.0%}{beyond}")

    # segment length distribution - the decisive quantitative check
    lens = []
    with open(src) as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            lens.append(len(line.split()))
    n_full = sum(1 for L in lens if L >= 400)
    n_short = sum(1 for L in lens if L < 200)
    total = sum(1 for _ in open(src))
    print(f"\n  segment token lengths (first {len(lens):,} of {total:,}):")
    print(f"    mean {sum(lens)/len(lens):.0f} | >=400 tokens: {n_full/len(lens):.1%} "
          f"| <200 tokens: {n_short/len(lens):.1%}")
    print(f"    -> a 163-char/line export would give ~100-token segments "
          f"(nearly all <200)")
    print(f"    -> a 1,581-char/passage export gives mostly full 512-token segments")

    # predicted example counts from each candidate
    print(f"\n  actual train.source segments: {total:,}")
    for name, path in CANDIDATES.items():
        if not os.path.exists(path):
            continue
        nlines = sum(1 for _ in open(path, encoding="utf-8"))
        # each source line yields ceil(tokens/512) segments, so segments >= lines
        ratio = total / nlines
        if ratio < 0.99:
            verdict = "IMPOSSIBLE - fewer segments than input lines"
        elif ratio < 1.05:
            verdict = "would mean ~1 segment per line (short lines)"
        else:
            verdict = f"{ratio:.1f} segments per line (long, multi-segment inputs)"
        print(f"    from {name:14} ({nlines:>8,} lines): {verdict}")
