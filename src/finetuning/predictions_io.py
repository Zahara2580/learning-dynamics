"""Save and load predictions as indexed JSONL records."""

import json
from pathlib import Path
from typing import Union


def write_predictions(path: Union[str, Path], predictions: list[str]) -> None:
    """Write predictions as JSONL records keyed by example index."""
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(predictions):
            f.write(json.dumps({"i": i, "pred": p}, ensure_ascii=False) + "\n")


def read_predictions(path: Union[str, Path]) -> list[str]:
    """Read predictions in index order and reject missing or duplicate indices."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("i")) is not int
                or not isinstance(row.get("pred"), str)):
            raise ValueError(f"{path}: expected integer i and string pred in every row")
    rows.sort(key=lambda r: r["i"])
    if [r["i"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: prediction indices must be unique and contiguous from zero")
    return [r["pred"] for r in rows]
