# learning-dynamics

Honours project studying continued pretraining (CPT) of T5, ByT5 and
Nguni-ByT5 on isiXhosa, with monolingual and bilingual experiments.

## Training and evaluation

- `src/data_processing/`: dataset downloads and corpus export/mixing.
- `src/pretraining/`: span corruption preprocessing, CPT training,
  schedules and checkpoint resume.
  Entry points are `continued_pretrain.py` for training and `preprocess.py`
  for span corruption; `data.py` contains dataset, sampler and collator classes.
- `src/finetuning/run_finetune.py`: MT and T2X fine-tuning and evaluation;
  `config.py`, `metrics.py` and `predictions_io.py` implement configuration
  and scoring. Dataset loaders are in `src/data_processing/preparation/`.
- `configs/`: model and fine-tuning configurations.
- `scripts/lafand_bilingual/` and `scripts/lafand_candidates/`: CPT launchers,
  including the batch-size runs.
- `scripts/finetune/` and `scripts/finetune_bilingual/`: fine-tuning and ablation launchers.
- `collab/`: Colab representation/probing experiments.
- `src/diagnostics/crosslingual_alignment.py` and `layerwise_alignment.py`:
  alignment measurements used by the plotting scripts.
- `src/diagnostics/t2x_slot_metrics.py`: subject/object copy scoring;
  `summarize_t2x_f1.py` recreates the F1 figures from saved scores. See
  [T2X F1 instructions](src/diagnostics/README_t2x.md).
- `src/utils/backup_checkpoints.py` and
  `src/diagnostics/verify_hf_backup.py`: Hugging Face checkpoint backup and
  verification. The Python backup utility remains; its shell launcher was removed.

Install dependencies with `uv sync --frozen`. Launchers contain the HPC paths
and resource settings used for the experiments.

## Dataset preparation

Run these commands from the repository root:

```sh
uv run python -m src.data_processing.downloads.wura --language xho
uv run python -m src.data_processing.downloads.wura --language eng
uv run python -m src.data_processing.downloads.flores
uv run python -m src.data_processing.downloads.t2x
```

WURA is saved under `data/corpus/{language}-passage`. FLORES saves `dev`
and `devtest` to `data/finetune/mt`; T2X saves `train`, `valid` and `test`
triples and references to `data/finetune/d2t`. Use `--output-dir` to change
these roots.

MT training automatically loads `allenai/wmt22_african` (`eng-xho`, `train`)
through `src.data_processing.downloads.wmt22`. To populate the Hugging Face
cache beforehand, run `uv run python -m src.data_processing.downloads.wmt22`.
LASER ranking, deduplication, training subset size and translation direction
are handled by `src/data_processing/preparation/data_mt.py`.

Corpus preparation commands are
`python -m src.data_processing.preparation.export_wura_lines` and
`python -m src.data_processing.preparation.merge_shuffle_lines`; use `--help`
for their input and output options. Tokenisation and span corruption remain
in `src/pretraining/preprocess.py`.

The fine-tuning data loaders live in `src/data_processing/preparation/`:

| File | Purpose |
|---|---|
| `data_mt.py` | Select MT training pairs, load FLORES splits and apply translation direction. |
| `data_t2x.py` | Load T2X triples and references; construct training pairs for fine-tuning and supply references for slot scoring. |

## Fine-tuning files

`src/finetuning/` contains the training pipeline and plotting code:

| File | Purpose |
|---|---|
| `run_finetune.py` | Fine-tune each CPT checkpoint, generate predictions and save evaluation scores. |
| `config.py` | Load and validate fine-tuning settings. |
| `metrics.py` | Compute BLEU, chrF, chrF++ and TER during evaluation, including multiple references. |
| `predictions_io.py` | Save/load indexed predictions without breaking Unicode or embedded newlines. |
| `plot.py` | Draw MT/D2T grids, alignment and bilingual combined figures; dispatch other figure families. |
| `figstyle.py` | Shared colours, fonts, sizing and legends, also used by CPT and POS plotters. |
| `__init__.py` | Python package marker. |

Standalone table/report generators and paired significance testing were removed.
Alignment, bilingual and common plotting helpers are consolidated into `plot.py`.

## Retained figure scripts

The keep set is the former `FINAL_SUBMISSION` figures plus the T2X F1 graphs.
This maps figure families to their implementations; filenames alone do not
recover every original command-line/style setting.

All figure families are available through `python -m src.finetuning.plot COMMAND`.

| Figures | Command |
|---|---|
| `all_tasks_2x3.png` | `full` |
| `ablations_chrf.png`, `ablations_bleu.png` | `ablations` |
| `ablations_d2t.png`, `ablations_mt.png` | `task` |
| `bilingual_probe_alignment_3x3.png` | `bilingual` |
| Alignment layer panels | `alignment` |
| POS layer accuracy, pooling comparisons/deltas and per-class F1 | `probes` |
| CPT validation-loss curves | `cpt` |
| T2X subject/object F1 tables and graphs | `f1` |

The full-results, ablation and custom task grids share one renderer in
`src/finetuning/plot.py`, which also contains alignment and combined bilingual
plotting, result loading and axis helpers. `figstyle.py` holds shared visual
styles. Probing, CPT-loss and F1 plotting remain in their respective directories
and are accessible through the same command.

Use `uv run python -m src.finetuning.plot COMMAND --help` for inputs and options.
For example:

```sh
uv run python -m src.finetuning.plot full --sparse \
  --mt ablation_results/mt_full.jsonl \
  --mt-xhen ablation_results/mt_xhen_full.jsonl \
  --d2t ablation_results/d2t_full.jsonl \
  --out FINAL_SUBMISSION/all_tasks_2x3.png

uv run python -m src.finetuning.plot f1
```

Generated images and detailed probe JSON files were removed during cleanup.
For POS plots backed by retained CSV summaries, pass `--scores PATH --only`
with the desired figures (see `--help`). The per-class F1 plot needs the
original nested `metrics.json` results restored via `--run-dir`; aggregate
CSV accuracy cannot reconstruct per-class F1. Plotting code remains intact.

## POS probing and crosslingual alignment

`src/pos_probing/run.py` contains the supplied linear POS experiment: pinned
MasakhaPOS splits, frozen encoders, mean/first/last word pooling, training-only
feature scaling and validation-loss checkpoint selection. The effective default
from the supplied notebook is the three bilingual models; use `--models t5 byt5
nguni-byt5` for monolingual runs. Results include probe weights, metrics,
predictions and settings. Attribution and the upstream licence are in the same
folder.

`src/crosslingual_alignment/run.py` measures FLORES devtest alignment at each
encoder layer: paired cosine similarity, its gap from all-pairs similarity,
and retrieval P@1 in both directions. It retains float32 extraction,
512-token truncation and attention-mask mean pooling from the supplied script.
The default is the three monolingual models. Layer records are saved before
marking a checkpoint complete, so an interrupted step can be recomputed.

Each folder has a separate `plot.py`; plotting reads saved outputs and does not
load models. The commented installation commands at the top retain the supplied
Colab dependencies. POS pins Transformers 4.57.1 in a separate Colab runtime;
these comments do not change the repository lockfile. Restart the runtime after
installing. Use `--login` when interactive Hugging Face authentication is needed,
or use an existing login. Drive mounting is handled outside these scripts;
`--output-dir` can point to a mounted Drive directory.

```sh
python -m src.pos_probing.run --output-dir results/pos_probing
python -m src.pos_probing.plot --run-dir results/pos_probing/RUN_ID
python -m src.crosslingual_alignment.run --output-dir results/crosslingual_alignment
python -m src.crosslingual_alignment.plot --input-dir results/crosslingual_alignment
```

The POS plot command exports `scores.csv` and draws layer profiles, checkpoint
trajectories and pooling comparisons. Alignment plots show final-layer
trajectories and layer profiles. Existing figure entry points and Colab scripts
remain available for their existing workflows.
