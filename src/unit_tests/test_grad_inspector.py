"""
Disposable diagnostic: trains T5 twice, once with gradient_accumulation_steps=1
and once with =128, comparing the final accumulated embedding gradient
norm between the two. If accumulation is correctly scaled, the two
norms should be roughly comparable (same order of magnitude). If the
128-run's norm is ~128x the 1-run's, that's a real scaling bug.

Usage:
    uv run python3 -m src.unit_tests.test_grad_inspector --input /scratch/rmdrak003/data/preprocessed/t5
"""

import argparse
import logging

from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from src.pretraining.collator import DataCollatorForT5MLM, compute_input_and_target_lengths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "google-t5/t5-large"
MAX_SEQ_LENGTH = 512
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0
BATCH_SIZE = 2


class GradientInspectorCallback(TrainerCallback):
    def __init__(self, model, label):
        self.model = model
        self.label = label
        self.final_norm = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grad = self.model.shared.weight.grad
        if grad is not None:
            self.final_norm = grad.norm().item()
            logger.info(f"[{self.label}] FINAL accumulated grad norm before optimizer step = {self.final_norm:.6f}")
        return control


def run(input_path: str, grad_accum_steps: int) -> float:
    label = f"accum={grad_accum_steps}"
    logger.info(f"--- Loading fresh model for {label} ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_from_disk(input_path)

    _, target_length = compute_input_and_target_lengths(
        input_length=MAX_SEQ_LENGTH,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
    )
    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
        input_length=MAX_SEQ_LENGTH,
        target_length=target_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"/tmp/grad-inspector-{grad_accum_steps}",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=1,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
    )

    callback = GradientInspectorCallback(model, label)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[callback],
    )

    logger.info(f"Running 1 global step with gradient_accumulation_steps={grad_accum_steps}...")
    trainer.train()

    return callback.final_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    norm_1 = run(args.input, grad_accum_steps=1)
    norm_128 = run(args.input, grad_accum_steps=128)

    logger.info("--- COMPARISON ---")
    logger.info(f"accum=1:   final grad norm = {norm_1:.6f}")
    logger.info(f"accum=128: final grad norm = {norm_128:.6f}")
    logger.info(f"Ratio (128 / 1) = {norm_128 / norm_1:.2f}")
    logger.info("Ratio near 1 = gradients correctly averaged. Ratio near 128 = scaling bug.")


if __name__ == "__main__":
    main()