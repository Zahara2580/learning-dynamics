"""
Prove the CPT checkpoints are real, distinct, ordered weight states, and
that from_pretrained loads them faithfully.

Four checks, CPU-only, no GPU needed:
  1. DISTANCE FROM BASE  per checkpoint, relative Frobenius diff of a
     fixed tensor sample vs the un-adapted base model. CPT trained iff
     these are non-zero; the warmup schedule implies they grow with step.
  2. CONSECUTIVE DISTINCTNESS  diff between adjacent checkpoints; any
     zero means two "different" checkpoints hold identical weights.
  3. LOAD FIDELITY  AutoModelForSeq2SeqLM.from_pretrained on one
     checkpoint, tensor compared bit-for-bit against the raw safetensors
     file - proves the finetuning harness's load path maps disk to
     memory unchanged.
  4. CROSS-MODEL  byt5 vs nguni-byt5 at the same step, and base vs base:
     rules out one model's checkpoints being copies of the other's.

Comparisons run in bf16, the dtype the checkpoints are stored in, so
diffs are exact file-level facts rather than cast artefacts.

Usage:
    uv run python3 -m scripts.diagnostics.verify_checkpoint_weights --model byt5
    uv run python3 -m scripts.diagnostics.verify_checkpoint_weights --model all --cross
"""

import argparse
import re
from argparse import Namespace
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints"),
}

# Representative sample: embeddings, first/last encoder and decoder
# attention, and the output head. Missing names are skipped per model.
TENSOR_PATTERNS = [
    "shared.weight",
    "encoder.block.0.layer.0.SelfAttention.q.weight",
    "encoder.block.{last_enc}.layer.0.SelfAttention.o.weight",
    "decoder.block.0.layer.0.SelfAttention.q.weight",
    "decoder.block.{last_dec}.layer.2.DenseReluDense.wo.weight",
    "lm_head.weight",
]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Weight-level verification of CPT checkpoints.")
    parser.add_argument("--model", type=str, default="all",
                        choices=[*MODELS, "all"])
    parser.add_argument("--cross", action="store_true",
                        help="Also diff byt5 vs nguni-byt5 at matching steps.")
    parser.add_argument("--fidelity-step", type=int, default=1000,
                        help="Checkpoint used for the from_pretrained fidelity check.")
    parser.add_argument("--base-override", type=str, default=None,
                        help="Base model path/id override (for smoke tests).")
    parser.add_argument("--checkpoints-override", type=str, default=None,
                        help="Checkpoints dir override (for smoke tests).")
    return parser.parse_args()


def checkpoint_dirs(root: Path) -> list[tuple[int, Path]]:
    out = []
    for p in root.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def resolve_names(available: set[str]) -> list[str]:
    """Fill {last_enc}/{last_dec} from the largest block indices present."""
    def last_block(prefix: str) -> int:
        idxs = [int(m.group(1)) for k in available
                if (m := re.match(rf"{prefix}\.block\.(\d+)\.", k))]
        return max(idxs) if idxs else 0

    le, ld = last_block("encoder"), last_block("decoder")
    names = [p.format(last_enc=le, last_dec=ld) for p in TENSOR_PATTERNS]
    return [n for n in names if n in available]


def load_tensors(ckpt_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    file = ckpt_dir / "model.safetensors"
    out = {}
    with safe_open(str(file), framework="pt") as f:
        keys = set(f.keys())
        for n in names:
            if n in keys:
                out[n] = f.get_tensor(n)
    return out


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius distance, computed in fp32 for numerical sanity."""
    a32, b32 = a.float(), b.float()
    denom = b32.norm().item() or 1.0
    return ((a32 - b32).norm().item()) / denom


def mean_rel_diff(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> float:
    common = [n for n in x if n in y]
    return sum(rel_diff(x[n], y[n]) for n in common) / max(1, len(common))


def spearman(a: list[float], b: list[float]) -> float:
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def base_state(base_id: str, names_hint: list[str]) -> dict[str, torch.Tensor]:
    """Base model tensors in bf16 (the checkpoints' storage dtype)."""
    model = AutoModelForSeq2SeqLM.from_pretrained(base_id, dtype=torch.bfloat16)
    sd = model.state_dict()
    del model
    available = set(sd.keys())
    names = resolve_names(available)
    return {n: sd[n].clone() for n in names}


def verify_model(model_name: str, base_id: str, ckpt_root: Path, fidelity_step: int) -> dict:
    print("=" * 88)
    print(f"{model_name}: base={base_id}")
    print(f"          checkpoints={ckpt_root}")
    print("=" * 88)

    ckpts = checkpoint_dirs(ckpt_root)
    if not ckpts:
        print("  !! no checkpoints found")
        return {}

    base = base_state(base_id, TENSOR_PATTERNS)
    names = list(base.keys())
    print(f"  sampling {len(names)} tensors: {names}")

    rows, prev = [], None
    for step, d in ckpts:
        cur = load_tensors(d, names)
        vs_base = mean_rel_diff(cur, base)
        vs_prev = mean_rel_diff(cur, prev) if prev is not None else float("nan")
        rows.append((step, vs_base, vs_prev))
        prev = cur

    print(f"\n  {'step':>7} {'rel diff vs base':>17} {'rel diff vs prev':>17}")
    for step, vb, vp in rows:
        print(f"  {step:>7} {vb:>17.6f} {vp:>17.6f}" + ("" if vp == vp else "        (first)"))

    steps = [r[0] for r in rows]
    vs_base = [r[1] for r in rows]
    vs_prev = [r[2] for r in rows[1:]]
    rho = spearman(steps, vs_base)

    print(f"\n  VERDICTS")
    print(f"    all checkpoints differ from base : "
          f"{'PASS' if min(vs_base) > 0 else '*** FAIL - a checkpoint equals base ***'}"
          f"   (min {min(vs_base):.6f})")
    print(f"    all consecutive pairs distinct   : "
          f"{'PASS' if min(vs_prev) > 0 else '*** FAIL - duplicate checkpoints ***'}"
          f"   (min {min(vs_prev):.6f})")
    print(f"    distance grows with step         : rho={rho:+.3f}  "
          f"{'PASS' if rho > 0.8 else 'CHECK - not monotone (see table)'}")

    # Load-fidelity: harness-style load vs raw file bytes.
    fid = [d for s, d in ckpts if s == fidelity_step] or [ckpts[len(ckpts) // 2][1]]
    fdir = fid[0]
    model = AutoModelForSeq2SeqLM.from_pretrained(str(fdir), dtype=torch.bfloat16)
    probe = names[0]
    loaded = dict(model.state_dict())[probe]
    raw = load_tensors(fdir, [probe])[probe]
    exact = torch.equal(loaded, raw)
    print(f"    from_pretrained fidelity ({fdir.name}, {probe}): "
          f"{'PASS - bit-identical to file' if exact else '*** FAIL - loaded tensor differs from disk ***'}")
    del model

    return {"names": names, "rows": rows, "last": load_tensors(ckpts[-1][1], names),
            "base": base}


def main() -> None:
    args = parse_args()
    if args.base_override or args.checkpoints_override:
        results = {"smoke": verify_model(
            "smoke", args.base_override, Path(args.checkpoints_override), args.fidelity_step)}
        return

    wanted = list(MODELS) if args.model == "all" else [args.model]
    results = {}
    for m in wanted:
        base_id, root = MODELS[m]
        results[m] = verify_model(m, base_id, Path(root), args.fidelity_step)

    if args.cross and "byt5" in results and "nguni-byt5" in results:
        a, b = results["byt5"], results["nguni-byt5"]
        if a and b:
            print("=" * 88)
            print("CROSS-MODEL (rules out copied checkpoints)")
            print("=" * 88)
            print(f"  byt5 base vs nguni base          : {mean_rel_diff(a['base'], b['base']):.6f} "
                  f"(nguni adaptation - expect >> 0)")
            print(f"  byt5 ckpt-10000 vs nguni ckpt-10000: {mean_rel_diff(a['last'], b['last']):.6f} "
                  f"(expect >> 0)")


if __name__ == "__main__":
    main()
