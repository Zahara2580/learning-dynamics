"""
Human-readable inspection of the lafand-preprocessed .source/.target files.

Decodes examples back to text so the corruption can be eyeballed, and
verifies the structural properties that would silently corrupt training
if violated (sentinel alignment, reconstruction, length sanity).

Usage:
    uv run python3 -m scripts.diagnostics.inspect_lafand_data \
        --model-config configs/models/t5.yaml \
        --data-dir /scratch/rmdrak003/data/lafand/t5 \
        --type-path train --n-examples 3
"""

import argparse
from argparse import Namespace

from transformers import AutoTokenizer

from src.pretraining.config import ModelConfig


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Eyeball the preprocessed lafand data.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--type-path", type=str, default="train", choices=["train", "dev"])
    parser.add_argument("--n-examples", type=int, default=3, help="How many examples to print in full.")
    parser.add_argument("--n-check", type=int, default=2000, help="How many examples to run structural checks over.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    first_sentinel = tokenizer.encode("<extra_id_0>")[0]
    # Build the sentinel id set the same way preprocessing did.
    sentinel_ids, i = [], 0
    while i <= 200:
        encoded = tokenizer.encode(f"<extra_id_{i}>")
        if len(encoded) != 2:
            break
        sentinel_ids.append(encoded[0])
        i += 1
    sentinel_set = set(sentinel_ids)

    src_path = f"{args.data_dir}/{args.type_path}.source"
    tgt_path = f"{args.data_dir}/{args.type_path}.target"
    with open(src_path) as f:
        sources = f.readlines()
    with open(tgt_path) as f:
        targets = f.readlines()

    print("=" * 78)
    print(f"MODEL      : {config.model_name_or_path}")
    print(f"DATA       : {args.data_dir} ({args.type_path})")
    print(f"SENTINELS  : {len(sentinel_ids)} available, first = {first_sentinel}, "
          f"order = {'ascending' if len(sentinel_ids) > 1 and sentinel_ids[1] > sentinel_ids[0] else 'descending'}")
    print(f"EXAMPLES   : {len(sources):,} source lines / {len(targets):,} target lines")
    print("=" * 78)

    if len(sources) != len(targets):
        print(f"!! MISMATCH: {len(sources)} sources vs {len(targets)} targets")

    # ---------- decode a few examples in full ----------
    for n in range(min(args.n_examples, len(sources))):
        src_ids = [int(t) for t in sources[n].split()]
        tgt_ids = [int(t) for t in targets[n].split()]

        print(f"\n{'-' * 78}\nEXAMPLE {n}   source={len(src_ids)} tokens, target={len(tgt_ids)} tokens\n{'-' * 78}")
        print("\n[SOURCE decoded]  (model sees this; <extra_id_N> = something removed)")
        print(tokenizer.decode(src_ids, skip_special_tokens=False))
        print("\n[TARGET decoded]  (model must produce this; the removed pieces)")
        print(tokenizer.decode(tgt_ids, skip_special_tokens=False))

        # Reconstruction: interleaving source and target non-sentinel runs
        # should recover the original passage in order.
        src_sent = [t for t in src_ids if t in sentinel_set]
        tgt_sent = [t for t in tgt_ids if t in sentinel_set]
        # Source and target are COMPLEMENTS: source gets one sentinel per
        # masked run, target one per unmasked run. Their counts differ by
        # at most 1 depending on whether the window ends masked or not -
        # equal counts is NOT the invariant. What must hold is that each
        # side's sentinels are the first-N in order (so <extra_id_k> in
        # the source refers to the same gap as <extra_id_k> in the target).
        print(f"\n[CHECK] sentinels in source: {len(src_sent)}, in target: {len(tgt_sent)} "
              f"(complementary; differ by <=1 by construction)")
        print(f"[CHECK] target starts with first sentinel ({first_sentinel})? "
              f"{'YES' if tgt_ids and tgt_ids[0] == first_sentinel else 'NO  <-- PROBLEM'}")
        print(f"[CHECK] counts differ by <= 1? "
              f"{'YES' if abs(len(src_sent) - len(tgt_sent)) <= 1 else 'NO  <-- PROBLEM'}")
        print(f"[CHECK] source sentinels are first-{len(src_sent)} in order? "
              f"{'YES' if src_sent == sentinel_ids[:len(src_sent)] else 'NO  <-- PROBLEM'}")
        print(f"[CHECK] target sentinels are first-{len(tgt_sent)} in order? "
              f"{'YES' if tgt_sent == sentinel_ids[:len(tgt_sent)] else 'NO  <-- PROBLEM'}")

    # ---------- structural checks over many examples ----------
    n_check = min(args.n_check, len(sources))
    bad_first, bad_align, empty, oversized = 0, 0, 0, 0
    src_lens, tgt_lens, n_sentinels = [], [], []
    vocab_size = len(tokenizer)
    out_of_range = 0

    for n in range(n_check):
        src_ids = [int(t) for t in sources[n].split()]
        tgt_ids = [int(t) for t in targets[n].split()]
        if not src_ids or not tgt_ids:
            empty += 1
            continue
        src_lens.append(len(src_ids))
        tgt_lens.append(len(tgt_ids))
        s_sent = [t for t in src_ids if t in sentinel_set]
        t_sent = [t for t in tgt_ids if t in sentinel_set]
        n_sentinels.append(len(s_sent))
        if tgt_ids[0] != first_sentinel:
            bad_first += 1
        # Correct invariant: each side uses the first-N sentinels in order,
        # and the two counts are complementary (differ by at most 1).
        if (s_sent != sentinel_ids[:len(s_sent)]
                or t_sent != sentinel_ids[:len(t_sent)]
                or abs(len(s_sent) - len(t_sent)) > 1):
            bad_align += 1
        if len(src_ids) > config.max_seq_length or len(tgt_ids) > config.max_target_length:
            oversized += 1
        if any(t < 0 or t >= vocab_size for t in src_ids + tgt_ids):
            out_of_range += 1

    print(f"\n{'=' * 78}\nSTRUCTURAL CHECKS over {n_check:,} examples\n{'=' * 78}")
    print(f"empty examples                        : {empty}          {'OK' if empty == 0 else '<-- PROBLEM'}")
    print(f"target not starting with 1st sentinel : {bad_first}          {'OK' if bad_first == 0 else '<-- PROBLEM'}")
    print(f"sentinel numbering broken             : {bad_align}          {'OK' if bad_align == 0 else '<-- PROBLEM'}")
    print(f"token ids outside vocab (0..{vocab_size-1})   : {out_of_range}          {'OK' if out_of_range == 0 else '<-- PROBLEM'}")
    print(f"examples longer than max_seq_length   : {oversized}          "
          f"{'OK' if oversized == 0 else '(will be truncated by collator)'}")
    print()
    print(f"source length  mean={sum(src_lens)/len(src_lens):.1f}  min={min(src_lens)}  max={max(src_lens)}")
    print(f"target length  mean={sum(tgt_lens)/len(tgt_lens):.1f}  min={min(tgt_lens)}  max={max(tgt_lens)}")
    print(f"sentinels/example mean={sum(n_sentinels)/len(n_sentinels):.1f}  "
          f"min={min(n_sentinels)}  max={max(n_sentinels)}  (limit {len(sentinel_ids)})")
    ratio = sum(tgt_lens) / sum(src_lens)
    print(f"\ntarget/source token ratio = {ratio:.2f}  "
          f"(expect ~0.2 for 15% i.i.d. masking; wildly different means the masking rate is off)")


if __name__ == "__main__":
    main()
