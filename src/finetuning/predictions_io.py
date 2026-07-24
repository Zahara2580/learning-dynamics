"""Prediction files as JSONL, one {"i", "pred"} object per line.

ByT5 emits raw bytes and can produce a newline inside a prediction, so a
newline-joined text file would silently misalign every later line. JSONL
keyed by test-set index is robust to any bytes the model emits.
"""

import json
from pathlib import Path
from typing import Union


def write_predictions(path: Union[str, Path], predictions: list[str]) -> None:
    """Write predictions as index-keyed JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(predictions):
            f.write(json.dumps({"i": i, "pred": p}, ensure_ascii=False) + "\n")


def read_predictions(path: Union[str, Path]) -> list[str]:
    """Read predictions back in index order."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["i"])
    return [r["pred"] for r in rows]
