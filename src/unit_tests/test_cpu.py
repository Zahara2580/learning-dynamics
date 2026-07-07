import argparse
import logging
from argparse import Namespace

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0


class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_steps: list[int]):
        self.checkpoint_steps = set(checkpoint_steps)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        if state.global_step in self.checkpoint_steps:
            logger.info(f"Step {state.global_step} is a scheduled checkpoint - saving.")
            control.should_save = True
        return control


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.model_config)
    logger.info(f"Model: {config.model_name_or_path}")

    batch_size = args.batch_size if args.batch_size is not None else config.batch_size

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.float32,
    )

    logger.info(f"Loading preprocessed chunks from {args.input}...")
    dataset = load_from_disk(args.input)
    logger.info(f"Loaded {len(dataset):,} chunks.")

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

    checkpoint_steps = compute_checkpoint_steps(config.total_steps, CheckpointScheduleConfig())
    logger.info(f"Checkpoint schedule: {len(checkpoint_steps)} checkpoints at steps {checkpoint_steps}")

    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(config.output_dir)

    actual_max_steps = args.max_steps if args.max_steps is not None else config.total_steps

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir + "-cputest",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
        max_steps=actual_max_steps,
        warmup_steps=config.warmup_steps,
        save_strategy="no",
        logging_steps=1,
        use_cpu=True,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[CustomCheckpointCallback(checkpoint_steps)],
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()