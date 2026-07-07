"""Configuration for pretraining a single model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class ModelConfig:
    """
    Configuration for continued pretraining a single model.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
        max_seq_length: Fixed chunk length used for preprocessing and
            training (this is the block size preprocess.py chunks to).
        learning_rate: Pretraining learning rate.
        batch_size: Per-device batch size.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        output_dir: Directory to save checkpoints to.
    """
    model_name_or_path: str
    max_seq_length: int
    learning_rate: float
    batch_size: int
    total_steps: int
    warmup_steps: int
    output_dir: str

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """
        Load a ModelConfig from a YAML file.

        Only recognised fields are read from the file - extra keys
        (e.g. wandb_project, checkpoint_schedule) are ignored here,
        since this class only covers what preprocessing/training need.

        :param path: Path to the YAML config file.
        :return: Populated ModelConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)