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
    :return: Set of (model, ckpt_step, task, seed) tuples.
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
                keys.add((row["model"], row["ckpt_step"], row["task"], row["seed"]))
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
    if cfg.task == "t2x":
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

    if cfg.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    else:
        scheduler = get_constant_schedule(optimizer)

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
) -> dict:
    """
    Finetune one checkpoint, score it, and delete its weights.

    The tokenizer always comes from the base model, never the
    checkpoint: CPT never altered the vocabulary, and the archived
    checkpoints are weights-only.

    :param step: CPT step this checkpoint came from (0 = base model).
    :param checkpoint_path: Path or HF id to load weights from.
    :param base_model: HF id of the un-CPT'd base model.
    :param cfg: Finetune configuration.
    :param data: Loaded task data.
    :param model_name: Model identifier for the results row.
    :param seed: Random seed.
    :return: The result row to append to results.jsonl.
    """
    set_seed(seed)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

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

    # ceil, not floor: the dataloader does not drop the last partial
    # batch, so flooring would make the MT linear schedule reach zero
    # before the final optimiser steps and silently stop training them.
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.num_epochs

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
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

    gen_started = time.time()
    predictions = generate_predictions(trainer.model, tokenizer, data["test_sources"], cfg)
    gen_runtime = time.time() - gen_started

    scored = score_corpus(predictions, data["test_references"])

    row = {
        "model": model_name,
        "ckpt_step": step,
        "task": cfg.task,
        "seed": seed,
        "metrics": scored["metrics"],
        "diagnostics": scored["diagnostics"],
        "sacrebleu_signatures": scored["sacrebleu_signatures"],
        "val_loss_per_epoch": val_losses,
        "best_epoch": best_epoch,
        "train_runtime_s": round(train_output.metrics.get("train_runtime", 0.0), 1),
        "generate_runtime_s": round(gen_runtime, 1),
        "total_runtime_s": round(time.time() - started, 1),
        "n_train_examples": len(train_dataset),
        "n_test_examples": len(predictions),
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash(),
    }

    # Predictions are permanent: every metric in the thesis can be
    # recomputed from these without re-running a single GPU hour. JSONL,
    # not newline-joined text - byte models can emit newlines.
    pred_dir = Path(cfg.results_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{model_name}_{step}_{cfg.task}_{seed}.jsonl"
    write_predictions(pred_path, predictions)

    # Weights are transient. ~5GB per run, ~120 runs.
    del trainer, model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(work_dir, ignore_errors=True)

    return row


def main() -> None:
    """Walk the checkpoints, finetuning and scoring each in turn."""
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)

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

    pending = [(s, p) for s, p in runs
               if (args.model, s, cfg.task, args.seed) not in completed]
    logger.info(
        f"{len(runs)} requested, {len(runs) - len(pending)} already done, {len(pending)} to run"
    )

    if args.dry_run:
        for step, path in pending:
            logger.info(f"  would run step {step}: {path}")
        return

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
        row = run_one_checkpoint(
            step, path, base_model, cfg, data, args.model, args.seed)
        append_result(results_path, row)
        logger.info(
            f"=== step {step} done in {time.time() - started:.0f}s | "
            f"BLEU {row['metrics']['bleu']:.2f} chrF {row['metrics']['chrf']:.2f} "
            f"chrF++ {row['metrics']['chrf_pp']:.2f} ==="
        )

    logger.info(f"sweep complete: {results_path}")


if __name__ == "__main__":
    main()
