"""
Configuration for downstream finetuning of a single CPT checkpoint.

A run is fully specified by (config, model_name, checkpoint_path, seed).
Everything in this dataclass is held CONSTANT across every checkpoint of
every model - the finetuning protocol is the measurement instrument, so
varying it would confound the learning-dynamics signal we are measuring.
Hyperparameters come from Meyer et al. (2024) / NGLUEni and are not tuned.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """
    Configuration for finetuning one checkpoint on one downstream task.

    Attributes:
        task: Task identifier, "d2t" (data-to-text, T2X dataset) or "mt".
        data_dir: Directory holding the task's data files.
        learning_rate: Finetuning learning rate (locked per task).
        batch_size: Per-device micro-batch. With
            gradient_accumulation_steps this multiplies to the paper's
            batch size; split purely to fit memory.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        num_epochs: Fixed number of training epochs (locked per task).
        lr_scheduler_type: "constant" for T2X (no warmup, no decay) or
            "linear" for MT (linear decay, no warmup), per the paper.
        warmup_steps: Absolute warmup steps. Leave 0 and use
            warmup_ratio instead, which adapts to each task's step count.
        warmup_ratio: Fraction of total steps spent warming up to the
            peak LR. 0.0 reproduces the paper (no warmup).
        eval_batch_size: Batch size for validation loss and generation.
            Affects throughput only, not results.
        max_source_length: Source truncation length. Byte-level models
            (ByT5) need far more positions than subword models for the
            same text, so this is set generously enough to never truncate
            either family on these datasets.
        max_target_length: Target truncation length during training.
        num_beams: Beam count for generation. FROZEN across checkpoints.
        max_new_tokens: Generation cap. FROZEN across checkpoints. ByT5
            emits BYTES, and isiXhosa is agglutinative with long words,
            so an undersized cap silently truncates output and destroys
            chrF without raising anything.
        source_prefix: Task prefix prepended to every source string.
            Identical for all checkpoints. "" unless the supervisor
            specifies one.
        direction_prefix: MT-only direction tag, e.g.
            "Translate English to Xhosa: ". Ignored for T2X.
        n_train_pairs: MT only: number of WMT22 pairs used for training,
            selected by the dedupe-then-top-N-by-laser_score rule in
            load_mt_train. 0 = unused (t2x).
        wandb_project: W&B project for the --wandb curves. Infra, not
            protocol (excluded from the hash).
        work_dir: Scratch directory for transient finetuned weights.
            Everything under here is deleted after scoring.
        results_dir: Permanent directory for results.jsonl + predictions.
    """
    task: str
    data_dir: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 8
    max_source_length: int = 512
    max_target_length: int = 256
    num_beams: int = 5
    max_new_tokens: int = 256
    source_prefix: str = ""
    direction_prefix: str = ""
    n_train_pairs: int = 0
    wandb_project: str = "cpt-finetune-dynamics"
    work_dir: str = "/scratch/rmdrak003/finetune_work"
    results_dir: str = "results/finetune"

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.task not in {"d2t", "mt"}:
            raise ValueError(f"task must be 'd2t' or 'mt', got {self.task!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.lr_scheduler_type not in {"constant", "linear"}:
            raise ValueError(
                f"lr_scheduler_type must be 'constant' or 'linear', got {self.lr_scheduler_type!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.warmup_steps and self.warmup_ratio:
            raise ValueError("set warmup_steps or warmup_ratio, not both")
        if self.num_beams <= 0:
            raise ValueError(f"num_beams must be positive, got {self.num_beams}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.task == "mt" and self.n_train_pairs <= 0:
            raise ValueError(
                f"mt task requires n_train_pairs > 0, got {self.n_train_pairs}"
            )

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimiser step - the number the paper specifies."""
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FinetuneConfig":
        """
        Load a FinetuneConfig from a YAML file.

        Unknown keys are ignored, mirroring ModelConfig.from_yaml, so a
        single YAML can carry annotations this class does not consume.

        :param path: Path to the YAML config file.
        :return: Populated FinetuneConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        dropped = set(data) - known_fields
        if dropped:
            logger.warning(f"ignoring unknown config keys: {sorted(dropped)}")
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def hash(self) -> str:
        """
        Stable short hash of the protocol-defining fields.

        Recorded on every results row so that rows produced under a
        changed protocol are detectable after the fact. Path fields are
        excluded: where results are written does not affect what they are.

        :return: First 12 hex chars of the sha256 of the config.
        """
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in {"work_dir", "results_dir", "data_dir", "eval_batch_size",
                         "wandb_project",
                         # split of the effective batch is a memory choice, not
                         # protocol: 4x4 and 16x1 optimise identically, so the
                         # product is hashed instead (below).
                         "batch_size", "gradient_accumulation_steps"}
        }
        payload["effective_batch_size"] = self.effective_batch_size
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
