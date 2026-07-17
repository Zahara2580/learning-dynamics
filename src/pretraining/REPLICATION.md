# Replication audit: lafand-mt `mt5_byt5_pre_training` -> this pipeline

Source: https://github.com/masakhane-io/lafand-mt (folder `mt5_byt5_pre_training`,
files `process.py`, `util.py`, `main.py`, `distribute_train.sh`). This is the
codebase used for AfriByT5, which Nguni-ByT5's continued pretraining was based
on (confirmed by supervisor).

## Replicated verbatim (the training objective)

| Original | Here | Notes |
|---|---|---|
| `racha_detection` (process.py) | `lafand_preprocess.racha_detection` | identical logic |
| `masking` (one sentinel per run, -100 drop) | `lafand_preprocess.masking` | identical; sentinel ids precomputed once (same values, not re-encoded per run) |
| `add_noise` (15% i.i.d. position sampling, complement target) | `lafand_preprocess.add_noise` | identical; tokenization hoisted out of the retry loop (same result, no re-tokenizing per retry) |
| `while(target[0] != 258)` alignment resampling | same loop, dynamic first-sentinel + 1000-attempt safety cap | cap statistically never fires (p_success = 0.85/attempt) |
| Offline corruption written to `train.source/.target` id files | same format | one fixed corruption per line, identical masks every epoch |
| `Seq2SeqDataset` (linecache reader, char lens, non-empty assert, `n_obs`) | `lafand_data.LafandSeq2SeqDataset` | pickle len-file path dropped (unused); `prefix` dropped (resolved to "" for these models) |
| `Seq2SeqDataCollator`, `generation_id` branch | `lafand_data.LafandSeq2SeqCollator` | identical parse/truncate/pad/attention-mask order; label padding behavior selectable (below) |
| No decoder_input_ids built (T5 shifts labels internally) | same | |
| `Seq2SeqTrainer` + offline dev files as eval set | same | |

## Deviations (each deliberate, each documented at the code site)

| # | Deviation | Why | Cost/effect |
|---|---|---|---|
| 1 | First mask-token id computed from tokenizer instead of hardcoded 258 | the repo README's own instruction ("change the number to the first token id"); resolves to 259 (byte models) / 32099 (t5) on current transformers - 259-ascending matches what Nguni-ByT5 itself trained with (verified by embedding forensics) | none |
| 2 | Lines under 10 chars filtered at export | empty lines crash the repo's own dataset assert; <10-char lines produce zero masked positions (degenerate examples) | 0.12% of corpus characters |
| 3 | Lines pre-truncated to 512 tokens before masking | lines long enough to need more mask tokens than exist (100/125) get silently corrupted by the original code (encodes the literal `<extra_id_125>` string) | 5.4% of train lines truncated |
| 4 | `random.seed(42)` | reproducibility (original unseeded) | none |
| 5 | Label padding: -100 (excluded from loss) instead of pad_token_id, unconditionally | measured A/B (t5 eval 3.53 vs 9.03); supervisor confirmed the intended behavior is padding-ignored ("the HF loss might ignore pad tokens" - true only for -100, which we therefore use); the faithful mode was removed entirely after the decision | changes loss scale, not examples |
| 6 | transformers 5.12.1 instead of pinned 4.10 | 4.10 doesn't install on py3.12; API-sensitive parts not ported; sentinel resolution matches nguni's actual training env | `remove_unused_columns=False` set to restore 4.10 collator behavior |
| 7 | WURA passage-level examples, windowed into 512-token chunks | supervisor directive ("process the passages as is... leave splitting into equal-length sequences up to Huggingface"); passages average ~1,580 chars, so without windowing ~87% would be truncated (discarding ~2/3 of long passages) | one passage -> ~2.3-3.6 window examples; no text discarded |
| 8 | Hyperparameters: 10k steps / 5000 warmup / effective batch 1024 | the published Nguni-ByT5 recipe, not the repo script's AfroMAFT defaults (4 epochs / 10000 warmup / eff. 2048 / seq 256) | supervisor's own recipe; seq len 512/512 per thesis design |
| 9 | Training infra: single-GPU accelerate bf16, two-phase checkpoint schedule, resume/ split, wandb + metrics.jsonl, eval every 200 steps on 2000-dev subsample | fairscale `sharded_ddp` is dead; the repo effectively disabled logging (`logging_steps=1e12`) and evaluated every 10k steps | additive; does not change the objective |
| 10 | Tokenizer saved into each weights-only checkpoint | makes checkpoints self-contained for finetuning | none |

## Not ported (unused by the repo's own pretraining run)

`prepare_seq2seq_batch` text path (removed API; their run used `generation_id`),
DistributedSortishSampler, fairseq dynamic batching, freeze utils (revisit for
the finetuning phase), `trim_batch`, pickle helpers, `check_output_dir`,
`evaluate.py`/predict modes (MT inference tooling), the 19-language loop and
its undefined-`i` file-writing bug.

## Enabled with supervisor sign-off

SortishSampler (ported from the repo's own util.py into `lafand_data.py`,
`--sortish-sampler` flag): length-grouped batching, measured 2.4x padded-token
reduction on real batches. Present in the repo but disabled in their launch
script; supervisor approved enabling it ("Yes, that should be okay").
