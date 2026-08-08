"""
Feed checkpoints a PROPERLY FORMATTED span-corruption input - lines from
the model's own CPT dev set, exactly as CPT saw them - and print what
they generate next to the true target.

This separates two hypotheses the D2T zero-shot table cannot:
  A) format prior: the model denoises fine in its native format; the D2T
     degeneracy is pure input mismatch (feed it sentinel-free tag soup,
     get degenerate output).
  B) damage: generation is collapsed even on in-format input, so CPT
     (or the masking pipeline) broke the model.

Reads the token-id .source/.target files directly - no tokenisation, no
prompt, byte-exact CPT format.

Usage (sintx, CPU is fine for a handful of examples):
    uv run python3 -m scripts.diagnostics.informat_generation --model byt5
    uv run python3 -m scripts.diagnostics.informat_generation --model nguni-byt5 --steps 0 10000
"""

import argparse
import re
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints",
           "/scratch/rmdrak003/data/lafand/t5"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints",
             "/scratch/rmdrak003/data/lafand/byt5"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints",
                   "/scratch/rmdrak003/data/lafand/nguni-byt5"),
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Generate on in-format CPT dev inputs.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS))
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1000, 5000, 10000],
                        help="Checkpoints to probe (0 = un-adapted model).")
    parser.add_argument("--n", type=int, default=4, help="Dev examples to show.")
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--base-override", type=str, default=None)
    parser.add_argument("--checkpoints-override", type=str, default=None)
    parser.add_argument("--data-override", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_id, ckpt_root, data_dir = MODELS[args.model]
    if args.base_override:
        base_id = args.base_override
    root = Path(args.checkpoints_override or ckpt_root)
    data = Path(args.data_override or data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    src_lines = data.joinpath("dev.source").read_text().splitlines()[:args.n]
    tgt_lines = data.joinpath("dev.target").read_text().splitlines()[:args.n]
    sources = [[int(t) for t in line.split()][:512] for line in src_lines]
    targets = [[int(t) for t in line.split()] for line in tgt_lines]

    ckpts = {int(m.group(1)): str(p) for p in root.glob("checkpoint-*")
             if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir()}
    runs = [(s, base_id if s == 0 else ckpts[s]) for s in args.steps if s == 0 or s in ckpts]

    print(f"model={args.model}  device={device}  in-format CPT dev inputs from {data}")
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        print(f"\n{'#' * 90}\nDEV EXAMPLE {i}")
        print(f"  INPUT  : {tokenizer.decode(src, skip_special_tokens=False)[:220]!r}")
        print(f"  TARGET : {tokenizer.decode(tgt, skip_special_tokens=False)[:220]!r}")
        for step, path in runs:
            model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device)
            model.eval()
            with torch.no_grad():
                ids = torch.tensor([src]).to(device)
                gen = model.generate(ids, max_new_tokens=args.max_new,
                                     num_beams=1, do_sample=False)
            text = tokenizer.decode(gen[0], skip_special_tokens=False)
            print(f"  step {step:>6}: {text[:220]!r}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
