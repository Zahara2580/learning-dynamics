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

# Language ID by the FRACTION of words that are English function words.
#
# An earlier version asked "does this line contain any English word?" - that
# works on single-sentence predictions but SATURATES on 185-word passages,
# where one brand name or "Co., Ltd." flips the whole passage. English running
# text is ~20-30% function words; isiXhosa is ~0-3% (only borrowings). The
# threshold sits in the empty band between those.
EN = re.compile(r"\b(the|and|of|to|is|in|that|was|for|with|a|it|on|as|at|by|"
                r"this|from|are|be|has|have|not|but|or|an|we|they|he|she|his|"
                r"her|its|which|will|would|been|were|had|you|all|can|their)\b", re.I)
THRESH = 0.08     # >= 8% English function words => English


def en_fraction(text):
    words = text.split()
    return len(EN.findall(text)) / len(words) if words else 0.0


SEARCH = [
    "/scratch/rmdrak003/data/lafand/lines-passage/*bilingual*.txt",
    "/scratch/rmdrak003/data/lafand-bilingual/lines*/*",
    "/scratch/rmdrak003/data/lafand/*bilingual*.txt",
]
PREPROCESSED = "/scratch/rmdrak003/data/lafand-bilingual/*/train.source"
# monolingual segment counts, for the "how much did adding English add?" ratio
MONO = {"t5": 161_722, "byt5": 250_840, "nguni-byt5": 250_840}
TOKENIZERS = {"t5": "google-t5/t5-large", "byt5": "google/byt5-large",
              "nguni-byt5": "google/byt5-large"}
SAMPLE = 4000        # segments to decode per model for the language split


def classify(line):
    return "en" if en_fraction(line) >= THRESH else "xh"


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
    fr = sorted(en_fraction(l) for l in lines)
    print(f"    EN-fraction distribution: min {fr[0]:.3f} | p25 {fr[len(fr)//4]:.3f} | "
          f"median {fr[len(fr)//2]:.3f} | p75 {fr[3*len(fr)//4]:.3f} | max {fr[-1]:.3f}")
    print(f"      (bimodal with an empty middle => the threshold is doing no work)")
    print(f"    sample EN: {next((l for l,t in zip(lines,tags) if t=='en'), '')[:88]!r}")
    print(f"    sample XH: {next((l for l,t in zip(lines,tags) if t=='xh'), '')[:88]!r}")
    # parity is provable without any classifier: the merge is xho + an equal
    # SAMPLE of eng, so the total must be exactly twice the xho passage count
    XHO = {139426: 69713, 15692: 7846}
    if len(lines) in XHO:
        h = XHO[len(lines)]
        print(f"    ARITHMETIC CHECK: {len(lines):,} = 2 x {h:,} xho passages -> "
              f"passage parity is EXACT by construction")

print("\n" + "=" * 74)
print("WHAT CPT ACTUALLY CONSUMED  (segments, decoded and language-classified)")
print("=" * 74)
print("  Passage parity does NOT imply segment parity: a tokenizer that")
print("  compresses English better than isiXhosa turns an equal number of")
print("  passages into unequal numbers of 512-token segments.\n")

import random

from transformers import AutoTokenizer

for src in sorted(glob.glob(PREPROCESSED)):
    model = os.path.basename(os.path.dirname(src))
    total = sum(1 for _ in open(src))
    tok = AutoTokenizer.from_pretrained(TOKENIZERS.get(model, "google/byt5-large"))

    random.seed(0)
    keep = set(random.sample(range(total), min(SAMPLE, total)))
    n_en = n_xh = 0
    with open(src) as f:
        for i, line in enumerate(f):
            if i not in keep:
                continue
            text = tok.decode([int(x) for x in line.split()], skip_special_tokens=True)
            if en_fraction(text) >= THRESH:
                n_en += 1
            else:
                n_xh += 1
    n = n_en + n_xh
    mono = MONO.get(model)
    print(f"  {model}")
    print(f"    total segments      : {total:,}"
          + (f"   (monolingual arm: {mono:,}, x{total/mono:.2f})" if mono else ""))
    print(f"    sampled             : {n:,}")
    print(f"    English segments    : {n_en/n:>6.1%}")
    print(f"    isiXhosa segments   : {n_xh/n:>6.1%}")
    print(f"    ratio EN:XH         : {n_en/max(n_xh,1):.3f}   (1.000 = equal exposure)")
    prov = os.path.join(os.path.dirname(src), "train.provenance.json")
    if not os.path.exists(prov):
        print(f"    [no provenance file - predates the provenance change]")
