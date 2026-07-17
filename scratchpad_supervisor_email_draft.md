# Draft email to supervisor

Subject: byt5/nguni-byt5 sentinel bug found + confirmed fix — need your input on nguni-byt5

Hi [supervisor name],

We found and fixed a real bug in our CPT collator affecting byt5, and along the way discovered something about nguni-byt5 that needs your input since you built that model.

## Background: what a "sentinel" is

T5-style pretraining works by hiding random spans of the input and replacing each hidden span with a single placeholder token (a "sentinel"), which the model then has to predict. Standard T5 has ~100 of these reserved as literal vocabulary entries (`<extra_id_0>`, `<extra_id_1>`, ...) at the very end of its vocabulary, and it was trained to recognise them as "something is hidden here."

Our collator builds sentinel IDs by counting down from a vocabulary size: `sentinel_id = vocab_size - 1`, `vocab_size - 2`, etc. For standard T5 this vocab_size is just `len(tokenizer)`, which correctly lands on those trained `<extra_id_*>` slots.

## The bug

ByT5's *actual* trained vocabulary is small — 256 raw byte values plus a couple of special tokens (pad/eos/unk), so really only ~259 IDs were ever seen during pretraining. It works on raw UTF-8 bytes, not subwords.

However, its Hugging Face tokenizer wrapper reports `len(tokenizer) == 384`, because — mirroring how the standard T5 tokenizer is built — it appends 125 unused `extra_id` slots on top of the real 259-ID vocabulary (IDs 259–383). Those 125 slots exist in the tokenizer object, but ByT5 was never pretrained on them: they're essentially untouched, randomly-initialized embedding rows.

Our collator builds sentinel IDs by counting down from a vocabulary size (`sentinel_id = vocab_size - 1`, `vocab_size - 2`, ...). For standard T5, using `len(tokenizer)` as that vocab_size is correct, because T5's `extra_id` slots genuinely were trained as sentinels. We used the same `len(tokenizer)` approach for byt5 (giving `vocab_size = 384`), which put every sentinel in the untrained 259–383 range.

The problem: the ByT5 paper doesn't add new sentinel tokens at all — it just reuses its own final 100 real byte IDs (~159–258) as sentinels. So the correct sentinel base for byt5 is 259 (the edge of its *real* trained vocabulary), not 384 (the tokenizer's padded-out reported length).

We were feeding byt5 masked-span markers from the untrained 259–383 range. Every time the model saw one, it was looking up a meaningless, randomly-initialized vector instead of "a span is hidden here."

## How we found it, with numbers

We ran a step-0 forward pass (no training yet) and looked at the loss:
- Random-guess baseline for a 384-token vocab is ln(384) ≈ 5.95.
- With our original sentinel setup: loss ≈ 8.09 — *worse than random*.
- Switching only the sentinel base to the paper's convention (259 instead of 384): loss dropped to ≈ 4.88.
- Also switching the mean corrupted-span length to the paper's value (byt5 uses 20 bytes per span, not T5's 3 subword tokens — this matters because it changes how "easy" the task is, not just where the sentinels live): loss dropped further to ≈ 0.81, which is a normal, healthy number for a pretrained model that hasn't been damaged.

We also independently confirmed the data itself is fine (decoded a chunk and it's clean, correct isiXhosa text) and that this isn't a bf16-vs-fp32 precision issue (we reran the same comparison in full fp32 and got the same bad loss, so it's not a rounding artifact — it's specifically the wrong sentinel IDs/span length).

## Where the "597" number comes from

Because span corruption *shrinks* the sequence (a multi-token span becomes a single sentinel token), we have to feed the model longer raw chunks than the final length we want, so that after corruption it lands exactly on our target length (512).

For T5/nguni-byt5 (mean span length 3), the chunk length needed to land on 512 after corruption is 568.
For byt5 with the corrected mean span length of 20, that pre-corruption chunk length becomes 597 instead. This is just arithmetic our code already does automatically (`compute_input_and_target_lengths`), not something we chose by hand — but it does mean byt5's preprocessed data needs to be regenerated at the new chunk length, since the old 568-length chunks don't match anymore.

## The fix we applied

- byt5 now uses sentinel base 259 (not 384) and mean span length 20 (not 3), matching the ByT5 paper.
- t5 needs no change — we tested it with the same method and its loss is a healthy ≈2.7 with its current (correct) settings.
- Re-preprocessed byt5's train/validation data at the new chunk length.

## Where we need your input: nguni-byt5

Since nguni-byt5 was itself continued-pretrained from byt5-large by your MAFT work, we ran the same test on it, checking all 4 combinations of {sentinel base 384 or 259} × {span length 3 or 20}:

| Sentinel base | Span length | Loss |
|---|---|---|
| 384 (current/T5-style) | 3 | **4.45** |
| 384 (current/T5-style) | 20 | 4.49 |
| 259 (ByT5 paper) | 3 | 8.82 |
| 259 (ByT5 paper) | 20 | 5.67 |

The result is the *opposite* of byt5: nguni-byt5 clearly prefers the T5-style convention (sentinel base 384, span 3) — the ByT5 paper's convention actively hurts it (8.82, worse than random). Our read is that your MAFT training must have used the standard T5-style collator (sentinels at the top of the vocab, span length 3) rather than byt5's original convention, so nguni-byt5's embedding rows 259–383 *did* get trained as real sentinels during your continued pretraining, and its final byte IDs are genuinely still "real bytes" to it, not free sentinel slots.

**Question for you:** can you confirm whether nguni-byt5's MAFT training used the standard HF/T5-style collator (top-of-vocab sentinels, mean span length 3), as our numbers above suggest? If so, we'll leave nguni-byt5's config exactly as-is (no change needed there) and only ship the byt5 fix. If you did something different, let us know and we'll re-test.

One design question this raises: byt5 will now train on 20-byte spans while nguni-byt5 trains on 3-token spans — a genuinely different corruption task per model, matching what each was actually pretrained on. We think that's the right call (training byt5 on the wrong objective would confound the comparison far worse), but flagging it since it means the three models' CPT isn't running literally identical objectives — let us know if you'd rather we standardize instead.

Happy to walk through any of this on a call if easier.

Thanks,
[your name]
