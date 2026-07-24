"""
Runtime probe: how this environment's Trainer logs loss under gradient
accumulation.

Prints, from inside the real Trainer.training_step path:
  1. model_accepts_loss_kwargs and the boolean deciding whether the
     divide-by-accumulation-steps reporting division runs
  2. whether training_step returns the pre-division compute_loss output
  3. _tr_loss accumulating raw per-micro-batch values
  4. the epoch-boundary dip: 100 samples, batch 1, accum 51 gives
     alternating windows of 51 and 49 micro-batches, so summed logging
     makes the reported loss drop by exactly 49/51 every second step

Subclasses and calls super() rather than patching internals, so the
behaviour printed is the library's own. Tiny randomly-initialised T5,
CPU-only, ~1-2 minutes.

    uv run python3 scripts/diagnostics/probe_trainer_loss_logging.py
"""

import os
import sys

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
    TrainerCallback,
)

print(f"transformers {transformers.__version__} | torch {torch.__version__} | python {sys.version.split()[0]}")
print(f"trainer.py: {transformers.trainer.__file__}")
print()

VOCAB, INPUT_LEN, LABEL_LEN = 128, 16, 8
DATASET_SIZE, ACCUM = 100, 51  # 100 micro-batches/epoch at bs=1 -> windows of 51 then 49

torch.manual_seed(0)
SAMPLE = {
    "input_ids": torch.randint(3, VOCAB, (INPUT_LEN,)).tolist(),
    "attention_mask": [1] * INPUT_LEN,
    "labels": torch.randint(3, VOCAB, (LABEL_LEN,)).tolist(),
}


class RepeatedSample(Dataset):
    """100 identical samples: per-micro-batch loss is constant within a
    window, so any variation in the logged loss is attributable purely to
    how many micro-batches the window contained."""

    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, idx: int) -> dict:
        return dict(SAMPLE)


records: dict = {"micro": [], "logged": [], "window_sizes": []}


class ProbeTrainer(Seq2SeqTrainer):
    _mb_in_window = 0
    _pre_division = None

    def compute_loss(self, model, inputs, *args, **kwargs):
        loss = super().compute_loss(model, inputs, *args, **kwargs)
        out = loss[0] if isinstance(loss, tuple) else loss
        self._pre_division = float(out.detach())
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        tr_loss_before = float(getattr(self, "_tr_loss", torch.tensor(float("nan"))))
        returned = super().training_step(model, inputs, num_items_in_batch)
        post = float(returned)
        pre = self._pre_division
        condition = (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and (
            self.compute_loss_func is None
        )
        i = self._mb_in_window
        if i == 0:
            n_items = int(num_items_in_batch) if num_items_in_batch is not None else None
            print(f"    model_accepts_loss_kwargs = {self.model_accepts_loss_kwargs}")
            print(f"    num_items_in_batch        = {n_items}")
            print(f"    compute_loss_func is None = {self.compute_loss_func is None}")
            print(
                f"    '(not accepts or items is None) and func is None' evaluates to {condition}"
                f"  ->  reporting division {'APPLIED' if condition else 'SKIPPED'}"
            )
        if i < 3 or i >= ACCUM - 2:
            print(
                f"  micro-batch {i:>3}: pre-division loss={pre:.4f} | training_step returned={post:.4f}"
                f" | pre/accum would be {pre / ACCUM:.4f} | _tr_loss before this add={tr_loss_before:.4f}"
            )
        records["micro"].append((pre, post))
        self._mb_in_window += 1
        return returned


class WindowReporter(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        records["window_sizes"].append(t._mb_in_window)
        print(f"== optimizer step {state.global_step} complete: window contained {t._mb_in_window} micro-batches ==")
        t._mb_in_window = 0
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            records["logged"].append(float(logs["loss"]))
            w = records["window_sizes"][-1]
            print(
                f"   LOGGED loss = {float(logs['loss']):.4f}"
                f"  (window {w} micro-batches; logged/window = {float(logs['loss']) / w:.4f} per micro-batch)"
            )
            print()
        return control


output_dir = os.environ.get("PROBE_OUTPUT_DIR", "/scratch/rmdrak003/tmp_probe_trainer_loss")

config = T5Config(
    vocab_size=VOCAB,
    d_model=64,
    d_ff=128,
    d_kv=16,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    decoder_start_token_id=0,
    pad_token_id=0,
    eos_token_id=1,
    dropout_rate=0.0,  # identical samples + no dropout -> per-micro loss exactly constant
)
model = T5ForConditionalGeneration(config)

args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=ACCUM,
    max_steps=4,
    learning_rate=1e-9,  # keep the model essentially frozen so per-micro loss stays constant
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    seed=0,
    use_cpu=True,
    disable_tqdm=True,
)

reporter = WindowReporter()
trainer = ProbeTrainer(model=model, args=args, train_dataset=RepeatedSample(), callbacks=[reporter])
reporter.trainer = trainer
trainer.train()

pre0, post0 = records["micro"][0]
logged = records["logged"]
sizes = records["window_sizes"]

print("================ VERDICTS ================")
bypassed = abs(post0 - pre0) < 1e-8
print(
    f"1. training_step returned {post0:.6f} vs pre-division {pre0:.6f}"
    f" (pre/accum would be {pre0 / ACCUM:.6f}) -> division {'BYPASSED' if bypassed else 'APPLIED'}"
)
print(
    f"2. logged step-1 loss {logged[0]:.3f} = {logged[0] / pre0:.2f} x the per-micro-batch loss {pre0:.3f}"
    f"  (expected ~{sizes[0]} if Trainer sums the window; ~1 if it averaged)"
)
dip_ratio = logged[1] / logged[0]
window_ratio = sizes[1] / sizes[0]
print(
    f"3. window sizes per optimizer step: {sizes}"
    f" -> logged step2/step1 = {dip_ratio:.4f} vs window ratio {sizes[1]}/{sizes[0]} = {window_ratio:.4f}"
)
print(
    f"   epoch-boundary dip equals the partial-window ratio: "
    f"{'CONFIRMED' if abs(dip_ratio - window_ratio) < 0.02 else 'NOT CONFIRMED - investigate further'}"
)
