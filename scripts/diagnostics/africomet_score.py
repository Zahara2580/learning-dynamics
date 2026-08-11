"""
Score saved MT predictions with AfriCOMET (masakhane/africomet-mtl).

Standalone on purpose: runs in a dedicated venv (unbabel-comet + datasets),
never the project environment - COMET's dependency pins must not touch the
frozen lockfile the training jobs use.

Walks every known predictions dir, scores each {"i","pred"} file against
FLORES devtest (src, ref) for its direction, and appends one row per file
to results/africomet/scores.jsonl (per-segment scores go to
results/africomet/segments/). Files already in scores.jsonl are skipped,
so reruns only score what is new - safe to resubmit as arms finish.
Identical prediction sets (best_epoch == last_epoch) are scored once and
reused via a content hash.

Usage (inside the comet venv):
    python scripts/diagnostics/africomet_score.py
    python scripts/diagnostics/africomet_score.py --limit-files 2   # smoke test
"""

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

# (pred_dir, direction, arm label); zero-shot direction comes from the filename task
ARMS = [
    ("results/finetune/predictions", "en-xh", "ft-3ep"),
    ("results_mt_5epoch/predictions", "en-xh", "ft-5ep"),
    ("results_mt_xhen_5epoch/predictions", "xh-en", "ft-5ep-xhen"),
    ("results/zero_shot/predictions", None, "zeroshot"),
]
MT_TASKS = {"mt": "en-xh", "mt-xhen": "xh-en"}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="AfriCOMET over saved MT predictions.")
    parser.add_argument("--flores-dir", type=str, default="data/finetune/mt")
    parser.add_argument("--comet-model", type=str, default="masakhane/africomet-mtl")
    parser.add_argument("--out-dir", type=str, default="results/africomet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Score only the first N pending files (smoke test).")
    return parser.parse_args()


def parse_name(stem: str) -> dict | None:
    """Finetuned: model_step_task_seed_selection (selection itself has one
    underscore); files predating the two-selection change have no selection
    suffix. Zero-shot: model_step_task. Returns None for non-MT files."""
    p = stem.split("_")
    if len(p) == 6:
        model, step, task, seed = p[0], p[1], p[2], p[3]
        selection = f"{p[4]}_{p[5]}"
    elif len(p) == 4:
        model, step, task, seed = p
        selection = None
    elif len(p) == 3:
        model, step, task = p
        seed, selection = None, None
    else:
        return None
    if task not in MT_TASKS:
        return None
    return {"model": model, "step": int(step), "task": task,
            "seed": int(seed) if seed else None, "selection": selection,
            "direction": MT_TASKS[task]}


def read_preds(path: Path) -> list[str]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["i"])
    return [r["pred"] for r in rows]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"

    done = set()
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["file"]))

    from datasets import load_from_disk
    devtest = load_from_disk(args.flores_dir)["devtest"]
    eng = [s.strip() for s in devtest["source"]]
    xho = [t.strip() for t in devtest["target"]]
    sides = {"en-xh": (eng, xho), "xh-en": (xho, eng)}

    pending = []
    for pred_dir, arm_direction, arm in ARMS:
        directory = Path(pred_dir)
        if not directory.is_dir():
            print(f"skip missing dir {pred_dir}")
            continue
        for f in sorted(directory.glob("*.jsonl")):
            meta = parse_name(f.stem)
            if meta is None or (arm, f.name) in done:
                continue
            if arm_direction is not None and meta["direction"] != arm_direction:
                continue
            pending.append((arm, f, meta))
    if args.limit_files:
        pending = pending[:args.limit_files]
    print(f"{len(pending)} prediction files to score ({len(done)} already done)")
    if not pending:
        return

    import torch
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(args.comet_model)
    model = load_from_checkpoint(ckpt)
    gpus = 1 if torch.cuda.is_available() else 0

    cache: dict[tuple, tuple] = {}   # (direction, preds-hash) -> (mean, scores)
    for k, (arm, f, meta) in enumerate(pending, start=1):
        preds = read_preds(f)
        src, ref = sides[meta["direction"]]
        n = min(len(preds), len(src))
        if len(preds) != len(src):
            print(f"  WARNING {f.name}: {len(preds)} preds vs {len(src)} pairs - scoring first {n}")
        key = (meta["direction"], hashlib.md5(json.dumps(preds).encode()).hexdigest())

        if key in cache:
            mean, scores = cache[key]
        else:
            data = [{"src": src[i], "mt": preds[i], "ref": ref[i]} for i in range(n)]
            out = model.predict(data, batch_size=args.batch_size, gpus=gpus, progress_bar=False)
            mean, scores = float(out.system_score), [float(s) for s in out.scores]
            cache[key] = (mean, scores)

        with open(seg_dir / f"{arm}__{f.name}", "w") as g:
            for i, s in enumerate(scores):
                g.write(json.dumps({"i": i, "africomet": s}) + "\n")
        row = {"arm": arm, "file": f.name, **meta, "n": n,
               "africomet_mean": mean, "comet_model": args.comet_model}
        with open(scores_path, "a") as g:
            g.write(json.dumps(row) + "\n")
        print(f"[{k}/{len(pending)}] {arm} {f.name}  mean={mean:.4f}")

    print(f"rows -> {scores_path}")


if __name__ == "__main__":
    main()
