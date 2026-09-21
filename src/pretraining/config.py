"""Load and validate model settings for continued pretraining"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Store and validate model settings for continued pretraining"""
    model_name_or_path: str
    max_seq_length: int
    learning_rate: float
    per_device_batch_size: int
    total_steps: int
    warmup_steps: int
    output_dir: str
    gradient_accumulation_steps: int = 1
    wandb_project: str = ""
    wandb_run_name: str = ""
    max_target_length: int = 512

    def __post_init__(self) -> None:
        """Validate configuration values after construction"""
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.per_device_batch_size <= 0:
            raise ValueError(f"per_device_batch_size must be positive, got {self.per_device_batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.max_target_length <= 0:
            raise ValueError(f"max_target_length must be positive, got {self.max_target_length}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load model settings from YAML """
        with open(path) as f:
            data = yaml.safe_load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        dropped = set(data) - known_fields
        if dropped:
            logger.warning(f"ignoring unknown config keys in {path}: {sorted(dropped)}")
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
