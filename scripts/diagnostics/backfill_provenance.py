"""
Record, beside the existing CPT data, which corpus export produced it.

The preprocessing scripts were run by hand before they wrote provenance, so
this was only established forensically (which_corpus_fed_cpt.py). One-off:
writes the finding next to the data so nobody has to redo that work.

    uv run python3 scripts/diagnostics/backfill_provenance.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATA = "/scratch/rmdrak003/data/lafand"
SOURCE_EXPORT = f"{DATA}/lines-passage"
MODELS = ["t5", "byt5", "nguni-byt5"]

EVIDENCE = [
    "Decoded train.source line 1 runs 498 tokens and continues past the point "
    "where lines/train.xh line 1 ends ('...Imbali YaseIndonesia'), into text "
    "present only in lines-passage/train.xh.",
    "65.5% of t5 segments and 78.3% of byt5 segments are >= 400 tokens; a "
    "163-char/line export cannot produce segments that long.",
    "lines/ would imply 0.24 (t5) / 0.37 (byt5) segments per input line. "
    "Segmentation only ever splits, so a ratio below 1.0 is impossible. "
    "lines-passage/ gives 2.32 and 3.60, matching 1,581-char passages.",
    "mtimes: lines/ 2026-07-15, lines-passage/ 2026-07-16 (supersedes).",
]


def head_digest(path, n=1 << 20):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(n)).hexdigest()


for model in MODELS:
    d = Path(DATA) / model
    if not d.is_dir():
        print(f"  skip {d} (not found)")
        continue
    for split, stem in (("train", "train"), ("dev", "dev")):
        src = d / f"{stem}.source"
        if not src.exists():
            continue
        line_file = Path(SOURCE_EXPORT) / f"{stem}.xh"
        rec = {
            "RECONSTRUCTED": True,
            "note": "Written after the fact by backfill_provenance.py. The "
                    "preprocessing run predates automatic provenance; the input "
                    "was established forensically, not read from a log.",
            "input_text": str(line_file),
            "input_lines": sum(1 for _ in open(line_file, encoding="utf-8"))
                           if line_file.exists() else None,
            "input_sha256_first_1mb": head_digest(line_file) if line_file.exists() else None,
            "output_examples": sum(1 for _ in open(src)),
            "superseded_export_NOT_used": f"{DATA}/lines",
            "evidence": EVIDENCE,
            "verified_by": "scripts/diagnostics/which_corpus_fed_cpt.py",
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        out = d / f"{stem}.provenance.json"
        out.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"  wrote {out}  ({rec['output_examples']:,} examples "
              f"from {rec['input_lines']:,} lines)"
              if rec["input_lines"] else f"  wrote {out}")
