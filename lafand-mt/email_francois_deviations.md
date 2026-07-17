# Draft email to Francois — pipeline questions

Subject: AfriByT5 pipeline replicated — 6 quick decisions before I launch the 7 runs

Hi Francois

I've replicated the mt5_byt5_pre_training pipeline from the lafand-mt repo inside my training setup (same masking code, same .source/.target data format, same collator logic), validated it end-to-end, and short GPU sanity runs are queued. Before launching the 7 batch-size runs, I need your call on a few points where the repo leaves a choice open, or where I had to adapt. My proposed default is in brackets for each — a one-word answer per item is perfect, or we can do a quick call.

**1. The general standard.** How faithful must I be to the repo? My working rule so far: replicate the training objective exactly (masking, mask-token assignment, target construction, padding), adapt only infrastructure (SLURM, checkpointing, logging), fix outright bugs, and document every deviation in the code. [Default: keep this rule.]

**2. Length-grouped batching — biggest schedule impact.** The repo's util.py includes the SortishSampler (it comes from HuggingFace's seq2seq utilities), which batches similar-length lines together to avoid padding waste. Your training script left it disabled. Our WURA lines vary a lot in length, so with random batching most batch positions are padding — I measured it on real batches, and enabling the modern equivalent would give roughly 2.5-3x more throughput, cutting the byte models from ~7 days to ~3 at the current queue congestion. Each optimizer step still averages 128 micro-batches, so gradient diversity per update is preserved; only the composition of individual micro-batches becomes length-grouped. [Default: enable it, given the deadline — but this is the one I most need your sign-off on, since your run had it off.]

**3. Label padding.** The repo pads target sequences with the pad token, which then contributes to the loss (~40% of target positions with our data). Modern practice excludes padding from the loss (-100). [Default: keep the repo's behavior for faithfulness; I'm A/B-testing both on short runs regardless and comparing eval curves.]

**4. Line granularity.** The repo consumes pre-split line files, but the splitting itself happened upstream and isn't in the repo. I split WURA into paragraph-level lines (headline + paragraphs, ~90-150 tokens on average). Sentence-level splitting is the alternative (shorter, more uniform, but less context and needs an isiXhosa sentence segmenter). [Default: paragraphs.]

**5. Sequence lengths.** Your distribute_train.sh used max_source/target_length 256; my memory budgeting and proposal were built around 512, and the code takes it as an argument. [Default: 512/512.]

**6. Hyperparameters.** I'm using your published Nguni-ByT5 recipe (10k steps, 5000 warmup, effective batch 1024) rather than the repo script's defaults (4 epochs, 10000 warmup, epoch-based). [Default: your recipe.]

For transparency, the adaptations I've already made, all documented in the code: the hardcoded first-mask-token ID 258 is now computed from the tokenizer per the repo README's own instruction (it resolves to 259 for the byte models on current library versions — the same value your Nguni-ByT5 training would have used); empty/near-empty lines are filtered (they crash the repo's own data loader; measured cost 0.12% of the corpus characters); lines are capped at 512 tokens before masking (much longer lines need more mask tokens than the tokenizer defines and get silently corrupted in the original code); the random seed is fixed for reproducibility; and I'm on the current transformers version rather than the pinned 4.10, which no longer installs on our Python.

One last thing: the l40sfree queue is heavily congested and it's now the main schedule risk for the two weeks. If you have access to a priority account I could use for the training window, it would help a great deal.

Thanks!
Rakeen
