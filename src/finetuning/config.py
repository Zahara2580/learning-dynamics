"""Load and validate fine-tuning settings."""

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
    """Store and validate settings for one fine-tuning task."""
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
    n_train_examples: int = 0
    direction: str = "en-xh"
    wandb_project: str = "cpt-finetune-dynamics"
    work_dir: str = "/scratch/rmdrak003/finetune_work"
    results_dir: str = "results/finetune"

    def __post_init__(self) -> None:
        """Validate the task and training settings."""
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
        if self.direction not in {"en-xh", "xh-en"}:
            raise ValueError(f"direction must be 'en-xh' or 'xh-en', got {self.direction!r}")
        if self.task == "mt" and self.n_train_pairs <= 0:
            raise ValueError(
                f"mt task requires n_train_pairs > 0, got {self.n_train_pairs}"
            )
        if self.n_train_examples < 0:
            raise ValueError(
                f"n_train_examples must be >= 0 (0 = full set), got {self.n_train_examples}")
        if self.n_train_examples and self.task != "d2t":
            raise ValueError("n_train_examples is d2t-only; MT sizes via n_train_pairs")

    @property
    def effective_batch_size(self) -> int:
        """Return the number of examples per optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'FinetuneConfig':
        """Load settings from YAML and warn about unknown keys."""
        with open(path) as f:
            data = yaml.safe_load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        dropped = set(data) - known_fields
        if dropped:
            logger.warning(f"ignoring unknown config keys: {sorted(dropped)}")
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def hash(self) -> str:
        """Return a stable hash of the experimental settings"""
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in {"work_dir", "results_dir", "data_dir", "eval_batch_size",
                         "wandb_project",
                         "batch_size", "gradient_accumulation_steps"}
        }
        if payload.get("direction") == "en-xh":
            payload.pop("direction")
        if not payload.get("n_train_examples"):
            payload.pop("n_train_examples", None)
        payload["effective_batch_size"] = self.effective_batch_size
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
