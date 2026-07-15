"""
Continued pretraining (CPT) for T5-family models (T5, ByT5, Nguni-ByT5)
using the lab-mandated lafand-mt (AfriByT5) pipeline: offline i.i.d.
span corruption preprocessed to {data_dir}/train.source|target and
dev.source|target files (see lafand_preprocess.py), read back by
LafandSeq2SeqDataset and batched by LafandSeq2SeqCollator.

The training harness (checkpoint schedule, resume logic, wandb, metrics
logging, dtype handling) is carried over unchanged from the previous
trainer; only the data pipeline differs.

Usage:
    uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand --model-config configs/models/t5.yaml --data-dir /scratch/rmdrak003/data/lafand/t5
    uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand --model-config configs/models/t5.yaml --data-dir /scratch/rmdrak003/data/lafand/t5 --resume
"""

import argparse
import json
import logging
import os
from argparse import Namespace
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from src.lafand_pretraining.lafand_data import LafandSeq2SeqCollator, LafandSeq2SeqDataset
from src.pretraining.config import ModelConfig
from src.pretraining.resume import find_latest_checkpoint
from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CustomCheckpointCallback(TrainerCallback):
    """
    Saves weights-only snapshots at the exact steps given by our
    two-phase schedule, since Trainer's built-in save_steps only
    supports a single fixed interval, not an irregular list of steps.

    These are separate from Trainer's own save_strategy="steps"
    checkpoints (which include optimizer/scheduler state for
    resumption): fine-tuning only ever needs model weights, so saving
    full state at all ~19 schedule steps would waste a lot of disk.
    """

    def __init__(self, checkpoint_steps: list[int], save_dir: Path):
        self.checkpoint_steps = set(checkpoint_steps)
        self.save_dir = save_dir

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step in self.checkpoint_steps and state.is_world_process_zero:
            model = kwargs.get("model")
            if model is not None:
                checkpoint_path = self.save_dir / f"checkpoint-{state.global_step}"
                # save_pretrained on a state_dict cast to bf16, rather than
                # model.to(torch.bfloat16), so the live training model (and
                # its optimizer, which is tied to its param dtype) is never
                # mutated mid-training - only the on-disk snapshot is bf16.
                bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
                model.save_pretrained(checkpoint_path, state_dict=bf16_state_dict)
                logger.info(f"Saved weights-only schedule checkpoint (bf16): {checkpoint_path}")
        return control


class GPUMemoryLoggingCallback(TrainerCallback):
    """
    Logs peak GPU memory usage periodically, to check over a longer run
    whether memory plateaus after the first few steps (expected) or keeps
    growing (would indicate a leak).
    """

    def __init__(self, log_every: int):
        self.log_every = log_every

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.is_world_process_zero and torch.cuda.is_available() and state.global_step % self.log_every == 0:
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"Step {state.global_step}: peak GPU memory allocated = {peak_gb:.2f} GB")
        return control


class MetricsLoggingCallback(TrainerCallback):
    """
    Appends every Trainer.log() call to a JSONL file, giving a persisted
    record to build loss curves from later. NOTE: the logged train loss
    is the SUM over the gradient-accumulation window (HF skips its
    normalization for T5-family models) - divide by the accumulation
    steps when plotting; eval_loss needs no correction.
    """

    def __init__(self, metrics_path: Path):
        self.metrics_path = metrics_path

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ) -> TrainerControl:
        if logs is None:
            return control
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return control


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Run lafand-pipeline CPT for a T5-family model.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory holding train.source/train.target and dev.source/dev.target "
             "(from lafand_preprocess.py).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint, if one exists.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the per_device_batch_size from the config.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Override the gradient_accumulation_steps from the config.")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="Interval (in steps) at which Trainer saves a full-state resume checkpoint.")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable all checkpoint saving, for throwaway trials that only "
                             "care about the loss/eval curves.")
    parser.add_argument("--log-memory-every", type=int, default=10,
                        help="Interval (in steps) at which peak GPU memory usage is logged.")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Interval (in steps) at which validation loss is computed.")
    parser.add_argument("--n-eval-obs", type=int, default=None,
                        help="Subsample the dev set to this many examples (the full paragraph-level "
                             "dev set is large; evaluating all of it every eval-steps is slow).")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override the warmup_steps from the config, for short test runs.")
    parser.add_argument("--metrics-filename", type=str, default="metrics.jsonl",
                        help="Filename (under config.output_dir) for Trainer.log() records. Override "
                             "when running concurrent trials against the same output_dir.")
    parser.add_argument("--run-subdir", type=str, default=None,
                        help="Subdirectory (under config.output_dir) isolating this run's "
                             "checkpoints/ and resume/ dirs, for concurrent trials.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Override the wandb_run_name from the config.")
    parser.add_argument("--model-dtype", type=str, choices=["bf16", "fp32"], default="bf16",
                        help="Dtype to load and train the model in.")
    parser.add_argument(
        "--ignore-pad-in-labels",
        action="store_true",
        help="Pad labels with -100 (excluded from loss) instead of the faithful lafand "
             "behavior of padding with pad_token_id (which contributes to the loss, as it "
             "did for nguni-byt5's own training). Off by default = faithful.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="If set, stop when eval_loss hasn't improved for this many evaluations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.model_config)
    use_wandb = bool(config.wandb_project) and bool(os.environ.get("WANDB_API_KEY"))
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else config.wandb_run_name
    if use_wandb:
        logger.info(f"Logging to wandb project: {config.wandb_project}, run: {wandb_run_name}")
    else:
        logger.info("wandb not configured (missing wandb_project in config or WANDB_API_KEY in .env) - skipping.")
    logger.info(f"Model: {config.model_name_or_path}")

    batch_size = args.batch_size if args.batch_size is not None else config.per_device_batch_size
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else config.gradient_accumulation_steps
    )
    logger.info(
        f"Per-device batch size: {batch_size}, "
        f"gradient accumulation steps: {gradient_accumulation_steps}, "
        f"effective batch size: {batch_size * gradient_accumulation_steps}"
    )

    use_bf16 = args.model_dtype == "bf16"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    actual_dtype = next(model.parameters()).dtype
    logger.info(f"Requested model dtype: {args.model_dtype}, actual loaded dtype: {actual_dtype}")
    if use_bf16 and actual_dtype != torch.bfloat16:
        raise ValueError(f"Requested bf16 but model loaded as {actual_dtype} - dtype mismatch.")
    if not use_bf16 and actual_dtype != torch.float32:
        raise ValueError(f"Requested fp32 but model loaded as {actual_dtype} - dtype mismatch.")

    logger.info(f"Loading lafand-preprocessed data from {args.data_dir}...")
    dataset = LafandSeq2SeqDataset(args.data_dir, type_path="train")
    eval_dataset = LafandSeq2SeqDataset(args.data_dir, type_path="dev", n_obs=args.n_eval_obs)
    logger.info(f"Loaded {len(dataset):,} train / {len(eval_dataset):,} dev examples.")

    collator = LafandSeq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=config.max_seq_length,
        max_target_length=config.max_target_length,
        ignore_pad_in_labels=args.ignore_pad_in_labels,
    )
    logger.info(
        f"Collator: max_source_length={config.max_seq_length}, "
        f"max_target_length={config.max_target_length}, "
        f"label padding={'-100 (excluded from loss)' if args.ignore_pad_in_labels else 'pad_token_id (faithful lafand behavior)'}"
    )

    actual_max_steps = args.max_steps if args.max_steps is not None else config.total_steps
    actual_warmup_steps = args.warmup_steps if args.warmup_steps is not None else config.warmup_steps

    checkpoint_steps = compute_checkpoint_steps(actual_max_steps, CheckpointScheduleConfig())
    logger.info(f"Checkpoint schedule: {len(checkpoint_steps)} checkpoints at steps {checkpoint_steps}")

    run_root = Path(config.output_dir) / args.run_subdir if args.run_subdir else Path(config.output_dir)
    schedule_checkpoint_dir = run_root / "checkpoints"
    resume_checkpoint_dir = run_root / "resume"
    schedule_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(str(resume_checkpoint_dir))

    early_stopping_enabled = args.early_stopping_patience is not None
    if args.no_save and early_stopping_enabled:
        raise ValueError("--no-save and --early-stopping-patience are incompatible: "
                         "early stopping needs load_best_model_at_end, which requires saved checkpoints.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(resume_checkpoint_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        max_steps=actual_max_steps,
        warmup_steps=actual_warmup_steps,
        save_strategy="no" if args.no_save else "steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=1,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=early_stopping_enabled,
        metric_for_best_model="eval_loss" if early_stopping_enabled else None,
        report_to=["wandb"] if use_wandb else [],
        run_name=wandb_run_name if use_wandb else None,
    )

    metrics_path = Path(config.output_dir) / args.metrics_filename
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks: list[TrainerCallback] = [
        MetricsLoggingCallback(metrics_path),
        GPUMemoryLoggingCallback(log_every=args.log_memory_every),
    ]
    if not args.no_save:
        callbacks.append(CustomCheckpointCallback(checkpoint_steps, save_dir=schedule_checkpoint_dir))
    if early_stopping_enabled:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        logger.info(f"Early stopping enabled: patience={args.early_stopping_patience} evaluations on eval_loss")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
