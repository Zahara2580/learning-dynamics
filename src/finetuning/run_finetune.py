"""
Finetune CPT checkpoints on a downstream task and score them.

Walks a model's checkpoints in increasing step order; for each,
finetunes a fresh copy, generates on the test set, appends one metrics
row, writes the predictions, and deletes the weights (~5GB x ~120 runs
would exceed the scratch quota; predictions allow any metric to be
recomputed later).

Completed runs are keyed (model, ckpt_step, task, seed) in results.jsonl
and skipped, so resubmitting after a killed job is safe.

Usage:
    uv run python3 -m src.finetuning.run_finetune \
        --config configs/finetune/t2x.yaml --model t5 --pilot
"""

import argparse
import collections
import gc
import json
import logging
import math
import os
import re
import shutil
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
)

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import (
    FLORES_TEST_SPLIT,
    FLORES_VALIDATION_SPLIT,
    build_mt_sources,
    load_flores_split,
    load_mt_train,
)
from src.finetuning.data_t2x import build_training_pairs, load_t2x_split
from src.finetuning.metrics import score_corpus
from src.finetuning.predictions_io import write_predictions
from src.pretraining.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Step 0 denotes the un-CPT'd base model pulled from HuggingFace. It runs
# through exactly the same finetune-and-score path as every checkpoint,
# so it is the anchor the whole curve is measured against.
BASE_MODEL_STEP = 0

# Which finetuned model gets evaluated: the best-validation epoch and the
# final epoch. Both come from one training run.
SELECTIONS = ("best_epoch", "last_epoch")


class Seq2SeqDataset(TorchDataset):
    """Tokenised (source, target) pairs for Seq2SeqTrainer."""

    def __init__(self, sources, targets, tokenizer, max_source_length, max_target_length):
        self.encodings = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        self.labels = labels["input_ids"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Finetune CPT checkpoints and score them.")
    parser.add_argument("--config", type=str, required=True, help="Finetune config YAML.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name, matching configs/models/{model}.yaml (t5, byt5, nguni-byt5).")
    parser.add_argument("--model-config-dir", type=str, default="configs/models",
                        help="Directory holding the CPT model YAMLs.")
    parser.add_argument("--checkpoints-dir", type=str, default=None,
                        help="Directory containing checkpoint-* subdirectories. "
                             "Defaults to auto-discovery under the CPT output_dir.")
    parser.add_argument("--base-model", type=str, default=None,
                        help="HF id for step 0. Defaults to the CPT config's model_name_or_path.")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Explicit checkpoint steps to run. Default: all discovered, plus step 0.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run only step 0 and the final checkpoint, to shake out "
                             "the pipeline and measure per-checkpoint wall-clock.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Only vary this for the explicit variance probe.")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip step 0 (the un-CPT'd base model).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would run, touch no GPU.")
    parser.add_argument("--wandb", action="store_true",
                        help="Stream live train/eval curves to W&B, one run per "
                             "checkpoint, grouped by model-task.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Override the W&B project name from the config.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results_dir, e.g. to keep a variance probe "
                             "separate from the main sweep.")
    return parser.parse_args()


def discover_checkpoints(checkpoints_dir: Path) -> list[tuple[int, str]]:
    """
    Find checkpoint directories and return them in increasing step order.

    :param checkpoints_dir: Directory containing checkpoint-* subdirs.
    :return: List of (step, path) sorted by step.
    """
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"checkpoints dir does not exist: {checkpoints_dir}")

    found = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), str(path)))

    if not found:
        raise FileNotFoundError(f"no checkpoint-* directories under {checkpoints_dir}")

    return sorted(found, key=lambda pair: pair[0])


def resolve_checkpoints_dir(model_config: ModelConfig, override: str | None) -> Path:
    """
    Locate the archived checkpoints for a CPT run.

    The CPT scripts write to {output_dir}/{run_subdir}/checkpoints, where
    run_subdir records the winning batch size (lafand-bs8 for t5,
    lafand-bs4 for the byte models). Rather than hardcode that mapping,
    glob for it and fail loudly if it is ambiguous.

    :param model_config: The CPT ModelConfig, for its output_dir.
    :param override: Explicit --checkpoints-dir, if given.
    :return: Directory containing checkpoint-* subdirectories.
    """
    if override:
        return Path(override)

    root = Path(model_config.output_dir)
    candidates = sorted(p for p in root.glob("*/checkpoints") if p.is_dir())

    if not candidates:
        raise FileNotFoundError(
            f"no */checkpoints directory under {root}; pass --checkpoints-dir explicitly"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous checkpoint directories under {root}: {[str(c) for c in candidates]}. "
            f"Pass --checkpoints-dir to choose (leftover sweep arms should be deleted)."
        )
    return candidates[0]


def load_completed_keys(results_path: Path) -> set[tuple]:
    """
    Read the keys of runs already recorded, for resume.

    Malformed trailing lines (a job killed mid-write) are skipped with a
    warning rather than crashing the sweep.

    :param results_path: Path to results.jsonl.
    :return: Set of (model, ckpt_step, task, seed, selection) tuples.
    """
    if not results_path.exists():
        return set()

    keys = set()
    with open(results_path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add((row["model"], row["ckpt_step"], row["task"], row["seed"],
                          row.get("selection", "best_epoch")))
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"skipping malformed results line {lineno} in {results_path}")
    return keys


def append_result(results_path: Path, row: dict) -> None:
    """
    Append one result row, flushed and fsynced.

    The sweep can be killed by SLURM at any moment; a buffered write
    would lose a finished run whose weights have already been deleted.

    :param results_path: Path to results.jsonl.
    :param row: The result row to append.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_task_data(cfg: FinetuneConfig) -> dict:
    """
    Load train/validation/test data for the configured task.

    Both tasks return the same shapes so the harness below is task
    agnostic: sources are plain strings, references are lists of lists
    (T2X test sentences have 1-8 (most have 1-3) references, FLORES
    always has 1).

    :param cfg: Finetune configuration.
    :return: Dict with train/val sources+targets and test sources+references.
    """
    if cfg.task == "d2t":
        data_dir = Path(cfg.data_dir)
        train_inputs, train_refs = load_t2x_split(data_dir, "train")
        val_inputs, val_refs = load_t2x_split(data_dir, "valid")
        test_inputs, test_refs = load_t2x_split(data_dir, "test")

        train_sources, train_targets = build_training_pairs(
            train_inputs, train_refs, cfg.source_prefix)
        # Validation loss is teacher-forced, so it needs exactly one
        # target; valid.text may pack multiple, so take the first.
        val_sources, val_targets = build_training_pairs(
            val_inputs, val_refs, cfg.source_prefix)
        test_sources = [cfg.source_prefix + inp for inp in test_inputs]

        return {
            "train_sources": train_sources, "train_targets": train_targets,
            "val_sources": val_sources, "val_targets": val_targets,
            "test_sources": test_sources, "test_references": test_refs,
        }

    # MT: FLORES dev/devtest for validation/test, training corpus TBD.
    raw_train_sources, raw_train_targets = load_mt_train(cfg)
    val_raw, val_refs = load_flores_split(cfg.data_dir, FLORES_VALIDATION_SPLIT)
    test_raw, test_refs = load_flores_split(cfg.data_dir, FLORES_TEST_SPLIT)

    return {
        "train_sources": build_mt_sources(raw_train_sources, cfg.source_prefix, cfg.direction_prefix),
        "train_targets": raw_train_targets,
        "val_sources": build_mt_sources(val_raw, cfg.source_prefix, cfg.direction_prefix),
        "val_targets": [r[0] for r in val_refs],
        "test_sources": build_mt_sources(test_raw, cfg.source_prefix, cfg.direction_prefix),
        "test_references": test_refs,
    }


def build_optimizer_and_scheduler(model, cfg: FinetuneConfig, num_training_steps: int):
    """
    Build Adafactor with the locked settings, plus the task's schedule.

    scale_parameter/relative_step/warmup_init are all disabled so that
    Adafactor honours the explicit learning rate from the paper rather
    than computing its own internal schedule.

    :param model: The model whose parameters are optimised.
    :param cfg: Finetune configuration.
    :param num_training_steps: Total optimiser steps, for linear decay.
    :return: (optimizer, scheduler).
    """
    optimizer = Adafactor(
        model.parameters(),
        lr=cfg.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    warmup = cfg.warmup_steps or round(cfg.warmup_ratio * num_training_steps)
    if cfg.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=num_training_steps)
    elif warmup:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    else:
        scheduler = get_constant_schedule(optimizer)

    logger.info(f"schedule: {cfg.lr_scheduler_type}, {warmup} warmup of "
                f"{num_training_steps} steps, peak lr {cfg.learning_rate}")
    return optimizer, scheduler


@torch.no_grad()
def generate_predictions(model, tokenizer, sources: list[str], cfg: FinetuneConfig) -> list[str]:
    """
    Generate on the test set with the frozen generation settings.

    Sources are processed in input order, not sorted by length: sorting
    would change how examples are batched and padded, which can perturb
    beam search outputs, and reproducibility matters more here than the
    throughput a length-sorted loop would buy.

    :param model: The finetuned model.
    :param tokenizer: The model's tokenizer.
    :param sources: Prefixed source strings.
    :param cfg: Finetune configuration (generation params come from here only).
    :return: Decoded prediction strings, one per source.
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for start in range(0, len(sources), cfg.eval_batch_size):
        batch = sources[start:start + cfg.eval_batch_size]
        encoded = tokenizer(
            batch,
            max_length=cfg.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated = model.generate(
            **encoded,
            num_beams=cfg.num_beams,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            # Pin the decoding companions so every checkpoint decodes
            # identically, regardless of what its generation_config.json
            # inherited (step 0 is an old Hub config, CPT checkpoints a
            # newer one). Vanilla beam search: no length bias, no n-gram
            # blocking, no repetition penalty.
            length_penalty=1.0,
            early_stopping=False,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
        )
        predictions.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )

        if start % (cfg.eval_batch_size * 20) == 0:
            logger.info(f"    generated {min(start + len(batch), len(sources))}/{len(sources)}")

    return predictions


def run_one_checkpoint(
    step: int, checkpoint_path: str, base_model: str,
    cfg: FinetuneConfig, data: dict, model_name: str, seed: int,
    use_wandb: bool = False, wandb_project: str = "", wandb_job_type: str = "sweep",
) -> list[dict]:
    """
    Finetune one checkpoint, score it, and delete its weights.

    The tokenizer always comes from the base model, never the
    checkpoint: CPT never altered the vocabulary, and the archived
    checkpoints are weights-only.

    When use_wandb is set, one grouped W&B run streams the live train and
    eval-loss curves; the final test metrics land in that run's summary.

    :param step: CPT step this checkpoint came from (0 = base model).
    :param checkpoint_path: Path or HF id to load weights from.
    :param base_model: HF id of the un-CPT'd base model.
    :param cfg: Finetune configuration.
    :param data: Loaded task data.
    :param model_name: Model identifier for the results row.
    :param seed: Random seed.
    :param use_wandb: Stream curves to W&B, one run per checkpoint.
    :param wandb_project: W&B project name.
    :param wandb_job_type: "pilot" or "sweep", for filtering.
    :return: One results row per selection in SELECTIONS.
    """
    set_seed(seed)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # fp32 explicitly: CPT saved bf16 checkpoints while the Hub base model is
    # fp32, so inheriting the checkpoint's dtype would finetune base with fp32
    # master weights and every CPT checkpoint with bf16 ones - a systematic
    # base-vs-checkpoint advantage. bf16=True still gives mixed-precision
    # compute; only the master weights are pinned.
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, dtype=torch.float32)
    logger.info(
        f"loaded {'BASE model' if step == BASE_MODEL_STEP else f'checkpoint-{step}'} "
        f"weights from {checkpoint_path} (dtype {next(model.parameters()).dtype})")

    train_dataset = Seq2SeqDataset(
        data["train_sources"], data["train_targets"], tokenizer,
        cfg.max_source_length, cfg.max_target_length)
    val_dataset = Seq2SeqDataset(
        data["val_sources"], data["val_targets"], tokenizer,
        cfg.max_source_length, cfg.max_target_length)

    work_dir = Path(cfg.work_dir) / f"{model_name}_{step}_{cfg.task}_{seed}"
    if work_dir.exists():
        # Leftovers from a job killed mid-run; the results row was never
        # written, so this run is redone from scratch.
        shutil.rmtree(work_dir)

    # Optimiser steps consume batch_size * accumulation examples. ceil, not
    # floor: the dataloader does not drop the last partial batch, so
    # flooring would make the linear schedule reach zero before the final
    # optimiser steps and silently stop training them.
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / cfg.effective_batch_size))
    total_steps = steps_per_epoch * cfg.num_epochs

    # One run per checkpoint, grouped by model-task so the ~120 runs
    # collapse to 6 groups; named by zero-padded step so they sort within.
    run_name = f"{model_name}-{cfg.task}-s{step:05d}" + (f"-seed{seed}" if seed != 42 else "")
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            group=f"{model_name}-{cfg.task}",
            job_type=wandb_job_type,
            name=run_name,
            tags=[model_name, cfg.task, f"seed{seed}", wandb_job_type],
            config={
                "model": model_name, "ckpt_step": step, "task": cfg.task, "seed": seed,
                "learning_rate": cfg.learning_rate, "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "effective_batch_size": cfg.effective_batch_size,
                "num_epochs": cfg.num_epochs, "lr_scheduler_type": cfg.lr_scheduler_type,
                "num_beams": cfg.num_beams, "max_new_tokens": cfg.max_new_tokens,
                "n_train_pairs": cfg.n_train_pairs, "config_hash": cfg.hash(),
            },
            reinit=True,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["wandb"] if use_wandb else [],
        run_name=run_name,
        seed=seed,
        bf16=torch.cuda.is_available(),
        # Generation is done manually after training so that the frozen
        # settings live in one place and multi-reference scoring stays
        # outside the Trainer's single-reference assumptions.
        predict_with_generate=False,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        optimizers=(optimizer, scheduler),
    )

    train_output = trainer.train()

    # Per-epoch validation losses, in order, plus which epoch won.
    val_losses = [
        entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry
    ]
    best_epoch = (val_losses.index(min(val_losses)) + 1) if val_losses else None

    # Training loss averaged per epoch, from the same history. Kept so the
    # train-vs-eval gap (overfitting, instability) is visible after the fact
    # without re-reading SLURM logs.
    per_epoch = collections.defaultdict(list)
    for entry in trainer.state.log_history:
        if "loss" in entry and "epoch" in entry:
            per_epoch[math.ceil(entry["epoch"])].append(entry["loss"])
    train_losses = [sum(v) / len(v) for _, v in sorted(per_epoch.items())]

    # save_total_limit=2 with load_best_model_at_end keeps exactly the best
    # (HF protects it from rotation) and the newest, so both selections are
    # available without keeping every epoch on disk.
    best_dir = trainer.state.best_model_checkpoint
    saved = sorted((p for p in work_dir.glob("checkpoint-*") if p.is_dir()),
                   key=lambda p: int(p.name.split("-")[1]))
    last_dir = str(saved[-1]) if saved else None
    best_is_last = (not best_dir) or (not last_dir) or Path(best_dir).name == Path(last_dir).name
    device = next(trainer.model.parameters()).device

    common = {
        "model": model_name, "ckpt_step": step, "task": cfg.task, "seed": seed,
        "val_loss_per_epoch": val_losses, "train_loss_per_epoch": train_losses,
        "best_epoch": best_epoch, "n_epochs": cfg.num_epochs,
        "best_is_last": best_is_last,
        "train_runtime_s": round(train_output.metrics.get("train_runtime", 0.0), 1),
        "n_train_examples": len(train_dataset),
        "checkpoint_path": checkpoint_path,
        "config_hash": cfg.hash(),
    }
    pinned = {"num_beams": cfg.num_beams, "max_new_tokens": cfg.max_new_tokens,
              "do_sample": False, "length_penalty": 1.0, "early_stopping": False,
              "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}
    pred_dir = Path(cfg.results_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def build_row(selection: str, predictions: list[str], gen_cfg, gen_runtime: float) -> dict:
        """Score one set of predictions and assemble its results row."""
        scored = score_corpus(predictions, data["test_references"])
        row = dict(common)
        row.update({
            "selection": selection,
            "metrics": scored["metrics"],
            "diagnostics": scored["diagnostics"],
            "sacrebleu_signatures": scored["sacrebleu_signatures"],
            "generation": {
                "pinned": pinned,
                "checkpoint_defaults": {k: getattr(gen_cfg, k, None) for k in
                                        ["length_penalty", "early_stopping",
                                         "no_repeat_ngram_size", "repetition_penalty",
                                         "num_beams", "max_length"]},
            },
            "generate_runtime_s": round(gen_runtime, 1),
            "total_runtime_s": round(time.time() - started, 1),
            "n_test_examples": len(predictions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        write_predictions(
            pred_dir / f"{model_name}_{step}_{cfg.task}_{seed}_{selection}.jsonl", predictions)
        logger.info(f"  {selection}: BLEU {scored['metrics']['bleu']:.2f} "
                    f"chrF {scored['metrics']['chrf']:.2f}")
        return row

    # load_best_model_at_end=True means the in-memory model is already the
    # best-validation epoch.
    gen_started = time.time()
    best_preds = generate_predictions(trainer.model, tokenizer, data["test_sources"], cfg)
    rows = [build_row("best_epoch", best_preds, trainer.model.generation_config,
                      time.time() - gen_started)]

    if best_is_last:
        # Same weights: reuse the predictions rather than decoding an
        # identical model twice.
        rows.append(build_row("last_epoch", best_preds, trainer.model.generation_config, 0.0))
        logger.info("  last_epoch == best_epoch (validation minimum was the final epoch)")
        del trainer, model, optimizer, scheduler
    else:
        # Free the trainer first: an fp32 byte model plus its optimiser state
        # and a second copy would not fit alongside each other.
        del trainer, model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        last_model = AutoModelForSeq2SeqLM.from_pretrained(
            last_dir, dtype=torch.float32).to(device)
        last_model.eval()
        gen_started = time.time()
        last_preds = generate_predictions(last_model, tokenizer, data["test_sources"], cfg)
        rows.append(build_row("last_epoch", last_preds, last_model.generation_config,
                              time.time() - gen_started))
        del last_model

    if use_wandb:
        import wandb
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_is_last"] = best_is_last
        for row in rows:
            for k, v in row["metrics"].items():
                wandb.run.summary[f"{row['selection']}/{k}"] = v
        wandb.finish()

    # Weights are transient. ~5GB per run. gc.collect() matters here: 20
    # checkpoints load sequentially in one process, and an fp32 byte model is
    # ~5GB of host RAM that Python is slow to hand back.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(work_dir, ignore_errors=True)

    return rows


def main() -> None:
    """Walk the checkpoints, finetuning and scoring each in turn."""
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)
    if args.results_dir:
        # Excluded from cfg.hash(), so the protocol is unchanged - only
        # where the rows land.
        cfg.results_dir = args.results_dir
        logger.info(f"results going to {cfg.results_dir}")

    model_config = ModelConfig.from_yaml(Path(args.model_config_dir) / f"{args.model}.yaml")
    base_model = args.base_model or model_config.model_name_or_path
    checkpoints_dir = resolve_checkpoints_dir(model_config, args.checkpoints_dir)

    discovered = discover_checkpoints(checkpoints_dir)
    logger.info(f"found {len(discovered)} checkpoints under {checkpoints_dir}")

    # Step 0 is the un-CPT'd base model, prepended so the sweep always
    # runs in increasing order of CPT exposure.
    runs: list[tuple[int, str]] = []
    if not args.skip_base:
        runs.append((BASE_MODEL_STEP, base_model))
    runs.extend(discovered)

    if args.pilot:
        # Cheapest end-to-end signal: does the curve have two endpoints,
        # and how long does one point take?
        runs = [runs[0], runs[-1]]
        logger.info(f"PILOT: running only steps {[s for s, _ in runs]}")
    elif args.steps is not None:
        wanted = set(args.steps)
        runs = [(s, p) for s, p in runs if s in wanted]
        logger.info(f"running explicit steps {[s for s, _ in runs]}")

    results_path = Path(cfg.results_dir) / "results.jsonl"
    completed = load_completed_keys(results_path)

    # both selections must be present for a checkpoint to count as done
    pending = [(s, p) for s, p in runs
               if not all((args.model, s, cfg.task, args.seed, sel) in completed
                          for sel in SELECTIONS)]
    logger.info(
        f"{len(runs)} requested, {len(runs) - len(pending)} already done, {len(pending)} to run"
    )

    if args.dry_run:
        for step, path in pending:
            logger.info(f"  would run step {step}: {path}")
        return

    wandb_project = args.wandb_project or cfg.wandb_project
    wandb_job_type = "pilot" if args.pilot else "sweep"
    if args.wandb:
        os.environ.setdefault("WANDB_WATCH", "false")      # no gradient logging on 1.2B params
        os.environ.setdefault("WANDB_LOG_MODEL", "false")  # weights are deleted, never uploaded
        logger.info(f"W&B on: project={wandb_project}, group=<model>-<task>, "
                    f"job_type={wandb_job_type}")

    # Loaded once: identical data for every checkpoint is the point of
    # the protocol, and re-reading it per run would only add I/O.
    data = load_task_data(cfg)
    logger.info(
        f"data: {len(data['train_sources'])} train, {len(data['val_sources'])} val, "
        f"{len(data['test_sources'])} test"
    )

    for index, (step, path) in enumerate(pending, start=1):
        logger.info(f"=== [{index}/{len(pending)}] {args.model} step {step} ({cfg.task}) ===")
        started = time.time()
        rows = run_one_checkpoint(
            step, path, base_model, cfg, data, args.model, args.seed,
            use_wandb=args.wandb, wandb_project=wandb_project, wandb_job_type=wandb_job_type)
        for row in rows:
            append_result(results_path, row)
        logger.info(f"=== step {step} done in {time.time() - started:.0f}s, "
                    f"{len(rows)} rows ===")

    logger.info(f"sweep complete: {results_path}")


if __name__ == "__main__":
    main()
