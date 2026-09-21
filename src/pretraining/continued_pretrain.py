"""Continue pretraining t5, byt5 and nguni and save scheduled checkpoints."""

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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from src.pretraining.config import ModelConfig
from src.pretraining.data import (
    Seq2SeqCollator,
    Seq2SeqDataset,
    SortishSampler,
)
from src.pretraining.resume import find_latest_checkpoint
from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_SEED = 42


class Trainer(Seq2SeqTrainer):
    """Train with batches of similar-length examples using sortish sampling."""
    def _get_train_sampler(self, train_dataset=None):

        dataset = train_dataset if train_dataset is not None else self.train_dataset

        return SortishSampler(dataset.src_lens, self.args.per_device_train_batch_size, shuffle=True)


class CustomCheckpointCallback(TrainerCallback):
    """Save model weights and the tokenizer at scheduled training steps."""
    def __init__(self, checkpoint_steps: list[int], save_dir: Path, tokenizer=None):

        self.checkpoint_steps = set(checkpoint_steps)
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    def on_step_end(self,_args,state: TrainerState,control: TrainerControl,**kwargs,) -> TrainerControl:
        if state.global_step not in self.checkpoint_steps:
            return control
        if not state.is_world_process_zero:
            return control
        model = kwargs.get('model')
        if model is None:
            return control

        checkpoint_path = self.save_dir / f'checkpoint-{state.global_step}'
        out_state_dict = {k: v.to(torch.float32) for k, v in model.state_dict().items()}
        model.save_pretrained(checkpoint_path, state_dict=out_state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}')
        return control


class MetricsLoggingCallback(TrainerCallback):
    """Append training and evaluation metrics to a JSONL file."""
    def __init__(self, metrics_path: Path):
        self.metrics_path = metrics_path

    def on_log(self, _args,state: TrainerState,control: TrainerControl,logs=None,**_kwargs,) -> TrainerControl:
        if logs is None:
            return control
        record = {'step': state.global_step, 'epoch': state.epoch, **logs}
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        return control


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Run continued pretraining.')
    parser.add_argument('--model-config', type=str, required=True)
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Preprocessed training and validation directory.',
    )
    parser.add_argument('--resume', action='store_true', help='Resume the latest checkpoint.')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None, help='Override the batch size.')
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=None,
        help='Override gradient accumulation.',
    )
    parser.add_argument('--save-steps', type=int, default=200, help='Resume checkpoint interval.')
    parser.add_argument('--no-save', action='store_true', help='Disable checkpoint saving.')
    parser.add_argument('--eval-steps', type=int, default=200, help='Evaluation interval.')
    parser.add_argument(
        '--eval-sets',
        type=str,
        nargs='+',
        default=None,
        help='Named validation sets, e.g. xho eng; defaults to dev.',
    )
    parser.add_argument('--warmup-steps', type=int, default=None, help='Override warmup steps.')
    parser.add_argument(
        '--metrics-filename',
        type=str,
        default='metrics.jsonl',
        help='Metrics filename within output_dir.',
    )
    parser.add_argument(
        '--run-subdir',
        type=str,
        default=None,
        help='Run subdirectory within output_dir.',
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Override the W&B run name.',
    )
    return parser.parse_args()


def main() -> None:
    """Load the model and datasets, then start or resume continued pretraining"""
    args = parse_args()
    config = ModelConfig.from_yaml(args.model_config)
    use_wandb = bool(config.wandb_project) and bool(os.environ.get('WANDB_API_KEY'))
    wandb_run_name = (
        args.wandb_run_name
        if args.wandb_run_name is not None
        else config.wandb_run_name
    )
    if use_wandb:
        logger.info(f'Logging to wandb project: {config.wandb_project}, run: {wandb_run_name}')
    else:
        logger.info('W&B disabled')
    logger.info(f'Model: {config.model_name_or_path}')
    batch_size = args.batch_size if args.batch_size is not None else config.per_device_batch_size
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else config.gradient_accumulation_steps
    )
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Gradient accumulation: {gradient_accumulation_steps}')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, dtype=torch.float32)

    actual_max_steps = args.max_steps if args.max_steps is not None else config.total_steps
    actual_warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else config.warmup_steps
    )
    logger.info(f'Training steps: {actual_max_steps}')
    logger.info(f'Warmup steps: {actual_warmup_steps}')
    logger.info(f'Learning rate: {config.learning_rate}')
    
    logger.info(f'Seed: {RANDOM_SEED}')
    logger.info(f'Data: {args.data_dir}')

    dataset = Seq2SeqDataset(args.data_dir, type_path='train')
    logger.info(f'Training examples: {len(dataset)}')

    if args.eval_sets:
        eval_dataset = {
            name: Seq2SeqDataset(args.data_dir, type_path=f'dev_{name}')
            for name in args.eval_sets
        }
        for name, data in eval_dataset.items():
            logger.info(f'Validation {name}: {len(data)} examples')
    else:
        eval_dataset = Seq2SeqDataset(args.data_dir, type_path='dev')
        logger.info(f'Validation examples: {len(eval_dataset)}')
    collator = Seq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=config.max_seq_length,
        max_target_length=config.max_target_length,
    )

    logger.info(f'Max source length: {config.max_seq_length}')
    logger.info(f'Max target length: {config.max_target_length}')

    checkpoint_steps = compute_checkpoint_steps(actual_max_steps, CheckpointScheduleConfig())
    logger.info(f'Checkpoint steps: {checkpoint_steps}')

    run_root = (
        Path(config.output_dir) / args.run_subdir
        if args.run_subdir
        else Path(config.output_dir)
    )
    schedule_checkpoint_dir = run_root / 'checkpoints'
    resume_checkpoint_dir = run_root / 'resume'
    schedule_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_from = None

    if args.resume:
        resume_from = find_latest_checkpoint(str(resume_checkpoint_dir))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(resume_checkpoint_dir),
        remove_unused_columns=False,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        max_steps=actual_max_steps,
        warmup_steps=actual_warmup_steps,
        save_strategy='no' if args.no_save else 'steps',
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=1,
        bf16=True,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        report_to=['wandb'] if use_wandb else [],
        run_name=wandb_run_name if use_wandb else None,
    )
    metrics_path = Path(config.output_dir) / args.metrics_filename
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [MetricsLoggingCallback(metrics_path)]

    if not args.no_save:
        callbacks.append(
            CustomCheckpointCallback(
                checkpoint_steps, save_dir=schedule_checkpoint_dir, tokenizer=tokenizer,
            )
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    logger.info('Starting training')
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info('Training complete!')


if __name__ == '__main__':
    main()
