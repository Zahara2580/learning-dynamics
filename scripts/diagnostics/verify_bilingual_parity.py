"""
Is the bilingual CPT corpus actually 1:1 isiXhosa:English?

The whole "did English stop the regression?" question rests on English being
present at the proportion we believe. That was never measured - it was read
off the build scripts. This measures it.

Language ID uses the same whole-word English function-word score that
separated 1,078 prediction files into two non-overlapping clusters
(EN 0.980-0.998, XH 0.012-0.173), so the threshold sits in an empty band.

    uv run python3 scripts/diagnostics/verify_bilingual_parity.py
"""

import glob
import os
import re

EN = re.compile(r"\b(the|and|of|to|is|in|that|was|for|with|a|it)\b", re.I)
THRESH = 0.5          # sits in the empty band between the two clusters

SEARCH = [
    "/scratch/rmdrak003/data/lafand-bilingual/lines*/*.txt",
    "/scratch/rmdrak003/data/lafand-bilingual/lines*/*.xh",
    "/scratch/rmdrak003/data/lafand-bilingual/*.txt",
    "/scratch/rmdrak003/data/lafand/lines*bilingual*/*",
    "/scratch/rmdrak003/data/lafand/*bilingual*.txt",
]
PREPROCESSED = "/scratch/rmdrak003/data/lafand-bilingual/*/train.source"


def classify(line):
    return "en" if EN.search(line) else "xh"


print("=" * 74)
print("BILINGUAL LINE FILES FOUND")
print("=" * 74)
found = sorted({p for pat in SEARCH for p in glob.glob(pat) if os.path.isfile(p)})
if not found:
    print("  none matched. Look manually:")
    print("    ls -la /scratch/rmdrak003/data/lafand-bilingual/")
    print("    find /scratch/rmdrak003/data -name '*bilingual*' -maxdepth 3")

for path in found:
    lines = [l for l in open(path, encoding="utf-8").read().split("\n") if l.strip()]
    if not lines:
        continue
    tags = [classify(l) for l in lines]
    n_en = tags.count("en")
    n_xh = tags.count("xh")
    chars_en = sum(len(l) for l, t in zip(lines, tags) if t == "en")
    chars_xh = sum(len(l) for l, t in zip(lines, tags) if t == "xh")
    print(f"\n  {path}")
    print(f"    lines           : {len(lines):,}")
    print(f"    classified EN   : {n_en:,}  ({n_en/len(lines):.1%})")
    print(f"    classified XH   : {n_xh:,}  ({n_xh/len(lines):.1%})")
    print(f"    passage ratio   : {n_en/max(n_xh,1):.3f} EN per XH   "
          f"(1.000 = exact parity)")
    print(f"    characters EN   : {chars_en:,}")
    print(f"    characters XH   : {chars_xh:,}")
    print(f"    CHARACTER ratio : {chars_en/max(chars_xh,1):.3f}  "
          f"<- parity was decided on PASSAGES, so this may differ")
    print(f"    sample EN: {next((l for l,t in zip(lines,tags) if t=='en'), '')[:90]!r}")
    print(f"    sample XH: {next((l for l,t in zip(lines,tags) if t=='xh'), '')[:90]!r}")

print("\n" + "=" * 74)
print("PREPROCESSED BILINGUAL DATA (what CPT actually consumed)")
print("=" * 74)
for src in sorted(glob.glob(PREPROCESSED)):
    n = sum(1 for _ in open(src))
    prov = os.path.join(os.path.dirname(src), "train.provenance.json")
    print(f"  {src}: {n:,} segments"
          f"{'   [provenance present]' if os.path.exists(prov) else '   [no provenance]'}")
