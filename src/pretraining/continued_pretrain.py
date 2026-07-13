"""
Continued pretraining (CPT) script for T5-family models (T5, ByT5,
Nguni-ByT5), using T5-style span corruption via DataCollatorForT5MLM.

Before running a real (non-test) job, manually check available scratch
space with storage_check.py rather than relying on this script to abort
mid-run - by the time a job is running, queue time and GPU allocation
are already spent, so an automated abort here wastes the expensive
resources it's meant to protect.

Usage:
    uv run python3 -m src.pretraining.continued_pretrain --model-config configs/models/t5.yaml --input /path/to/preprocessed/wura/train --eval-input /path/to/preprocessed/wura/validation
    uv run python3 -m src.pretraining.continued_pretrain --model-config configs/models/t5.yaml --input /path/to/preprocessed/wura/train --eval-input /path/to/preprocessed/wura/validation --resume
"""

import argparse
import json
import logging
import os
from argparse import Namespace
from pathlib import Path

import torch
from datasets import load_from_disk
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

from src.pretraining.collator import DataCollatorForT5MLM, compute_input_and_target_lengths
from src.pretraining.config import ModelConfig
from src.pretraining.resume import find_latest_checkpoint
from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0


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
    Appends every Trainer.log() call to a JSONL file, since report_to=[]
    means logs would otherwise only be visible in stdout - this gives a
    persisted record to build loss curves from later.
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
    parser = argparse.ArgumentParser(description="Run continued pretraining (CPT) for a T5-family model.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed WURA chunks.")
    parser.add_argument(
        "--eval-input",
        type=str,
        required=True,
        help="Path to preprocessed WURA validation chunks (see preprocess_wura.py --split validation).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint, if one exists.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per_device_batch_size from the config. The config's "
             "per_device_batch_size (1024) was tuned for base-sized models and is too large "
             "for large-sized models to fit in GPU memory directly.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override the gradient_accumulation_steps from the config. Accumulates "
             "gradients over N steps before updating weights, to simulate a larger "
             "effective batch size without needing it all in memory at once. "
             "effective_batch_size = batch_size * this value.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Interval (in steps) at which Trainer saves a full-state resume "
             "checkpoint. Lower this for quick resume tests so a checkpoint "
             "appears without waiting for the default 500 steps.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable all checkpoint saving (both the weights-only schedule "
             "snapshots and the full-state resume checkpoints). For throwaway "
             "trials that only care about the loss/eval curves, not the model.",
    )
    parser.add_argument(
        "--log-memory-every",
        type=int,
        default=10,
        help="Interval (in steps) at which peak GPU memory usage is logged, "
             "to help spot memory leaks over a longer run.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Interval (in steps) at which validation loss is computed.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override the warmup_steps from the config, for short test runs "
             "where the config's production warmup schedule (e.g. 5000 steps) "
             "wouldn't fit within --max-steps.",
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="metrics.jsonl",
        help="Filename (under config.output_dir) to append Trainer.log() records to. "
             "Override this when running multiple trials against the same output_dir "
             "concurrently (e.g. a batch-size sweep), so they don't interleave into "
             "one shared metrics.jsonl.",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default=None,
        help="Subdirectory (under config.output_dir) to isolate this run's "
             "checkpoints/ and resume/ dirs under. Required when running multiple "
             "trials against the same output_dir concurrently (e.g. a batch-size "
             "sweep), so they never share a Trainer output_dir or checkpoint path.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override the wandb_run_name from the config. Useful for parallel "
             "trials against the same model config (e.g. batch size sweeps) that "
             "would otherwise all log to the same wandb run name.",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="Dtype to load and train the model in. fp32 exists as a diagnostic "
             "for suspected bf16 numerical issues (e.g. byt5 showing worse-than-"
             "random loss at load time); it disables Trainer-level bf16 autocast "
             "too, so pass --mixed_precision no to accelerate launch as well for "
             "a clean fp32 run.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="If set, stop training when eval_loss hasn't improved for this many "
             "consecutive evaluations (via transformers' EarlyStoppingCallback). "
             "Disabled by default, so quick test runs don't pay the extra cost of "
             "load_best_model_at_end.",
    )
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
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(
        f"Per-device batch size: {batch_size}, "
        f"gradient accumulation steps: {gradient_accumulation_steps}, "
        f"effective batch size: {effective_batch_size}"
    )
    use_bf16 = args.model_dtype == "bf16"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    # Read the dtype back off the actual loaded parameters, rather than just
    # trusting --model-dtype, so a from_pretrained quirk (e.g. a checkpoint
    # shard that ignores torch_dtype) would be caught here rather than
    # silently invalidating a dtype comparison experiment.
    actual_dtype = next(model.parameters()).dtype
    logger.info(f"Requested model dtype: {args.model_dtype}, actual loaded dtype: {actual_dtype}")
    if use_bf16 and actual_dtype != torch.bfloat16:
        raise ValueError(f"Requested bf16 but model loaded as {actual_dtype} - dtype mismatch.")
    if not use_bf16 and actual_dtype != torch.float32:
        raise ValueError(f"Requested fp32 but model loaded as {actual_dtype} - dtype mismatch.")

    logger.info(f"Loading preprocessed chunks from {args.input}...")
    dataset = load_from_disk(args.input)
    logger.info(f"Loaded {len(dataset):,} chunks.")

    logger.info(f"Loading preprocessed validation chunks from {args.eval_input}...")
    eval_dataset = load_from_disk(args.eval_input)
    logger.info(f"Loaded {len(eval_dataset):,} validation chunks.")

    _, target_length = compute_input_and_target_lengths(
        input_length=config.max_seq_length,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
    )
    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
        input_length=config.max_seq_length,
        target_length=target_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    # --max-steps overrides how many steps actually run (for quick tests)
    actual_max_steps = args.max_steps if args.max_steps is not None else config.total_steps
    actual_warmup_steps = args.warmup_steps if args.warmup_steps is not None else config.warmup_steps

    checkpoint_steps = compute_checkpoint_steps(actual_max_steps, CheckpointScheduleConfig())
    logger.info(f"Checkpoint schedule: {len(checkpoint_steps)} checkpoints at steps {checkpoint_steps}")

    # Weights-only schedule snapshots (for fine-tuning) live under
    # checkpoints/, separate from resume/'s full-state checkpoints
    # (optimizer + scheduler state), so find_latest_checkpoint never
    # picks a weights-only folder to resume training from. --run-subdir
    # further isolates these under a per-run directory, so concurrent
    # trials against the same config/output_dir (e.g. a batch-size sweep)
    # never share a Trainer output_dir or checkpoint path.
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

    # NOTE: the ~330 loss seen with grad_accum=128 was Trainer logging the
    # *summed* loss over the accumulation window, not the mean (330 / 128
    # is consistent with the ~2.5 loss seen at grad_accum=1) - gradients
    # themselves were confirmed fine via a separate diagnostic script.
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(resume_checkpoint_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
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
        # load_best_model_at_end/metric_for_best_model are only needed by
        # EarlyStoppingCallback - forcing them on unconditionally would add
        # overhead (reloading the best checkpoint at the end) to quick test
        # runs that don't use early stopping.
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