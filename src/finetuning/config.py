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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class FinetuneConfig:
    """
    Configuration for finetuning one checkpoint on one downstream task.

    Attributes:
        task: Task identifier, "t2x" (data-to-text) or "mt" (translation).
        data_dir: Directory holding the task's data files.
        learning_rate: Finetuning learning rate (locked per task).
        batch_size: Per-device training batch size (locked per task).
        num_epochs: Fixed number of training epochs (locked per task).
        lr_scheduler_type: "constant" for T2X (no warmup, no decay) or
            "linear" for MT (linear decay, no warmup), per the paper.
        warmup_steps: Always 0 for both tasks - the paper specifies no
            warmup for downstream finetuning.
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
    eval_batch_size: int = 8
    max_source_length: int = 512
    max_target_length: int = 256
    num_beams: int = 5
    max_new_tokens: int = 256
    source_prefix: str = ""
    direction_prefix: str = ""
    work_dir: str = "/scratch/rmdrak003/finetune_work"
    results_dir: str = "results/finetune"

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.task not in {"t2x", "mt"}:
            raise ValueError(f"task must be 't2x' or 'mt', got {self.task!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.lr_scheduler_type not in {"constant", "linear"}:
            raise ValueError(
                f"lr_scheduler_type must be 'constant' or 'linear', got {self.lr_scheduler_type!r}"
            )
        if self.warmup_steps != 0:
            raise ValueError(
                f"the finetuning protocol specifies no warmup, got warmup_steps={self.warmup_steps}"
            )
        if self.num_beams <= 0:
            raise ValueError(f"num_beams must be positive, got {self.num_beams}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

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
            if k not in {"work_dir", "results_dir", "data_dir", "eval_batch_size"}
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
