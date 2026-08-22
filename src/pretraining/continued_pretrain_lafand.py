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
    uv run python3 -m src.pretraining.continued_pretrain_lafand --model-config configs/models/t5.yaml --data-dir /scratch/rmdrak003/data/lafand/t5
    uv run python3 -m src.pretraining.continued_pretrain_lafand --model-config configs/models/t5.yaml --data-dir /scratch/rmdrak003/data/lafand/t5 --resume
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

from src.pretraining.lafand_data import LafandSeq2SeqCollator, LafandSeq2SeqDataset, SortishSampler
from src.pretraining.config import ModelConfig
from src.pretraining.resume import find_latest_checkpoint
from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Explicit rather than relying on the HF default (also 42): the phase-1
# logs never stated the seed, so it was not reconstructable from them.
RANDOM_SEED = 42


class LafandTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional length-grouped batching via the
    SortishSampler ported from the lafand repo's own util.py (their
    launch script had it disabled; --sortish-sampler enables it here).
    Each optimizer step still accumulates many micro-batches spanning
    multiple sorted windows, so gradient diversity per update is
    largely preserved - only within-micro-batch lengths are grouped."""

    def __init__(self, *args, use_sortish_sampler: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sortish_sampler = use_sortish_sampler

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.use_sortish_sampler and dataset is not None and hasattr(dataset, "src_lens"):
            return SortishSampler(dataset.src_lens, self.args.per_device_train_batch_size, shuffle=True)
        return super()._get_train_sampler(train_dataset)


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

    def __init__(self, checkpoint_steps: list[int], save_dir: Path, tokenizer=None):
        self.checkpoint_steps = set(checkpoint_steps)
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step in self.checkpoint_steps and state.is_world_process_zero:
            model = kwargs.get("model")
            if model is not None:
                checkpoint_path = self.save_dir / f"checkpoint-{state.global_step}"
                # Snapshot in the SAME dtype the model is training in. Cast a
                # copy, never model.to(dtype): mutating the live model would
                # also change its optimiser state dtype mid-run.
                #
                # SAVED fp32, hardcoded. A bf16 snapshot quantises to an
                # 8-bit mantissa, and at T5's embedding magnitude the gap
                # between representable values is 0.0625 - so any genuine drift
                # below that is ERASED at save time, and a correctly-trained
                # fp32 run would still look frozen on disk, indistinguishable
                # from the phase-1 bug. Costs 2x disk; the measurement is worth
                # more. Cast a COPY: model.to(dtype) would mutate the live model
                # and its optimiser state mid-training.
                save_dtype = torch.float32
                out_state_dict = {k: v.to(save_dtype) for k, v in model.state_dict().items()}
                assert next(iter(out_state_dict.values())).dtype == torch.float32
                model.save_pretrained(checkpoint_path, state_dict=out_state_dict)
                # Save the tokenizer too, so each checkpoint is a fully
                # self-contained model dir that finetuning can point at
                # directly (no need to know the base model name).
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved weights-only schedule checkpoint "
                            f"({save_dtype}): {checkpoint_path}")
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
    parser.add_argument("--eval-sets", type=str, nargs="+", default=None,
                        help="Evaluate named dev sets separately: 'xho eng' reads "
                             "dev_xho.* and dev_eng.* and logs eval_xho_loss / "
                             "eval_eng_loss. Default: the single combined dev.*.")
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
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable bf16 autocast. Autocast is ON by default on "
                             "GPU: it casts ACTIVATIONS only, never the master "
                             "weights, gradients or optimiser state, all of which "
                             "stay fp32. There is deliberately no flag to make the "
                             "PARAMETERS bf16 - that was the phase-1 bug "
                             "(notes/bf16_finding.md).")
    parser.add_argument(
        "--sortish-sampler",
        action="store_true",
        help="Enable the SortishSampler ported from the lafand repo's util.py "
             "(length-grouped batching; collapses pad-to-batch-max waste, "
             "measured ~2-3x throughput on our length distribution). Their "
             "launch script had it disabled - off by default pending "
             "supervisor sign-off.",
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

    # PARAMETERS ARE ALWAYS fp32. Not configurable.
    #
    # Phase 1 passed torch_dtype=torch.bfloat16 here, which made the master
    # weights bf16. AdamW then wrote every update into an 8-bit mantissa, and
    # at T5's embedding magnitude (|w| ~ 18) the gap between representable
    # bf16 values is 0.125 - so an update of order lr=1e-4 rounded straight
    # back to the original number. Measured on those checkpoints: 99.83% of
    # t5's embeddings and 86.81% of ALL its parameters were bit-identical to
    # the base model after 10,000 steps. See notes/bf16_finding.md.
    #
    # bf16 AUTOCAST is a different thing and stays on: it casts activations
    # only. Verified - under autocast the parameter, its gradient and the
    # AdamW state all remain fp32, so the update lands.
    use_amp = not args.no_amp and torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name_or_path, dtype=torch.float32,
    )
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != torch.float32:
        raise RuntimeError(
            f"Model loaded as {actual_dtype}, expected torch.float32. Refusing to "
            f"train: reduced-precision PARAMETERS silently discard optimiser "
            f"updates (notes/bf16_finding.md).")

    actual_max_steps = args.max_steps if args.max_steps is not None else config.total_steps
    actual_warmup_steps = args.warmup_steps if args.warmup_steps is not None else config.warmup_steps
    # ---------------------------------------------------------------- banner
    # Everything a reader of this log needs to reconstruct the run, printed
    # once, unmissably. Phase 1 left three questions unanswerable after the
    # fact - what precision the parameters were, which corpus export fed the
    # preprocessing, and what the effective batch actually was. All three are
    # here now.
    bar = "=" * 78
    logger.info(bar)
    logger.info("RUN CONFIGURATION")
    logger.info(bar)
    logger.info(f"  model              : {config.model_name_or_path}")
    logger.info(f"  model config       : {args.model_config}")
    logger.info(f"  PARAMETERS         : {actual_dtype}   "
                f"{'OK' if actual_dtype == torch.float32 else 'WRONG - STOP'}")
    logger.info(f"  optimiser state    : {actual_dtype} (AdamW allocates in the param dtype)")
    logger.info(f"  gradients          : {actual_dtype}")
    logger.info(f"  bf16 autocast      : {use_amp}  (activations only)")
    logger.info(f"  archived ckpt dtype: torch.float32 (hardcoded)")
    logger.info(f"  resume ckpt dtype  : {actual_dtype} (model + optimiser state, "
                f"written by Trainer in the live param dtype)")
    logger.info(f"  optimiser          : adamw_torch, lr {config.learning_rate}")
    logger.info(f"  schedule           : {actual_warmup_steps} warmup / "
                f"{actual_max_steps} total, linear")
    logger.info(f"  effective batch    : {batch_size} x {gradient_accumulation_steps} "
                f"= {batch_size * gradient_accumulation_steps}")
    logger.info(f"  max_seq_length     : {config.max_seq_length}")
    logger.info(f"  seed               : {RANDOM_SEED}")
    logger.info(f"  sortish sampler    : {args.sortish_sampler}")

    logger.info(bar)
    logger.info("DATA")
    logger.info(bar)
    data_dir = Path(args.data_dir)
    logger.info(f"  --data-dir         : {data_dir.resolve()}")
    if "bilingual" in str(data_dir):
        logger.info(f"  corpus             : BILINGUAL (isiXhosa + English)")
    else:
        logger.info(f"  corpus             : MONOLINGUAL isiXhosa")
    # Which line-file export produced these token ids? Written by
    # lafand_preprocess.py; absent for data preprocessed before that change,
    # in which case say so rather than leaving it ambiguous.
    for split in ("train", "dev"):
        prov = data_dir / f"{split}.provenance.json"
        if prov.exists():
            import json as _json
            rec = _json.loads(prov.read_text())
            logger.info(f"  {split}.source built from: {rec.get('input_text')}")
            logger.info(f"     {rec.get('input_lines'):,} input lines -> "
                        f"{rec.get('output_examples'):,} segments, "
                        f"seed {rec.get('seed')}, "
                        f"max_line_tokens {rec.get('max_line_tokens')}")
        else:
            logger.info(f"  {split}.provenance.json ABSENT - the producing corpus is "
                        f"not recorded in the data dir. Verified separately as "
                        f"lines-passage/ (scripts/diagnostics/which_corpus_fed_cpt.py).")
    logger.info(bar)

    logger.info(f"Loading lafand-preprocessed data from {args.data_dir}...")
    dataset = LafandSeq2SeqDataset(args.data_dir, type_path="train")
    if args.eval_sets:
        # dict eval: Trainer evaluates each set and logs eval_{name}_loss
        eval_dataset = {name: LafandSeq2SeqDataset(args.data_dir, type_path=f"dev_{name}",
                                                   n_obs=args.n_eval_obs)
                        for name in args.eval_sets}
        sizes = ", ".join(f"{k}={len(v):,}" for k, v in eval_dataset.items())
        logger.info(f"Loaded {len(dataset):,} train / dev sets: {sizes}")
    else:
        eval_dataset = LafandSeq2SeqDataset(args.data_dir, type_path="dev", n_obs=args.n_eval_obs)
        logger.info(f"Loaded {len(dataset):,} train / {len(eval_dataset):,} dev examples.")

    collator = LafandSeq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=config.max_seq_length,
        max_target_length=config.max_target_length,
    )
    logger.info(
        f"Collator: max_source_length={config.max_seq_length}, "
        f"max_target_length={config.max_target_length}, "
        f"label padding=-100 (excluded from loss)"
    )


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
    if early_stopping_enabled and args.eval_sets:
        raise ValueError("--early-stopping-patience selects on eval_loss, which does not "
                         "exist with --eval-sets (keys are eval_<name>_loss). Use one or the other.")
    if args.no_save and early_stopping_enabled:
        raise ValueError("--no-save and --early-stopping-patience are incompatible: "
                         "early stopping needs load_best_model_at_end, which requires saved checkpoints.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(resume_checkpoint_dir),
        # Our dataset yields src_texts/tgt_texts/id staging keys for the
        # collator, not model-forward args; modern Trainer wraps the
        # collator with RemoveColumnsCollator and strips them unless this
        # is off. (transformers 4.10, which lafand ran, only stripped
        # columns from datasets.Dataset objects - plain torch Datasets
        # passed through - so False here matches their actual behavior.)
        remove_unused_columns=False,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        max_steps=actual_max_steps,
        warmup_steps=actual_warmup_steps,
        save_strategy="no" if args.no_save else "steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=1,
        bf16=use_amp,          # ACTIVATIONS only; master weights stay fp32
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
        callbacks.append(
            CustomCheckpointCallback(checkpoint_steps, save_dir=schedule_checkpoint_dir, tokenizer=tokenizer)
        )
    if early_stopping_enabled:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        logger.info(f"Early stopping enabled: patience={args.early_stopping_patience} evaluations on eval_loss")

    if args.sortish_sampler:
        logger.info("SortishSampler ENABLED (length-grouped batching, from lafand util.py)")
    trainer = LafandTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        use_sortish_sampler=args.sortish_sampler,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
