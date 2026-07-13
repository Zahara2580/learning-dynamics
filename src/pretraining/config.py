"""Configuration for pretraining a single model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
        per_device_batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of steps to accumulate
            gradients over before updating weights.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        output_dir: Directory to save checkpoints to.
        wandb_project: Weights & Biases project name to log to.
        wandb_run_name: Weights & Biases run name.
        mean_noise_span_length: Average corrupted-span length for span
            corruption. 3.0 is the T5 default; the ByT5 paper uses 20
            (bytes). Changing this changes the pre-corruption chunk
            length, so the corpus must be re-preprocessed to match.
        sentinel_base: Sentinel ids count down from sentinel_base - 1.
            None means len(tokenizer), which lands on the trained
            <extra_id_*> tokens for t5 and (empirically, see the
            sentinel diagnostics) matches nguni-byt5's MAFT. byt5 needs
            259 instead: the ByT5 paper reuses the final byte ids
            (258 down) as sentinels, and byt5's rows above 258 were
            never trained.
    """
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
    mean_noise_span_length: float = 3.0
    sentinel_base: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
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
        if self.mean_noise_span_length <= 0:
            raise ValueError(
                f"mean_noise_span_length must be positive, got {self.mean_noise_span_length}"
            )
        if self.sentinel_base is not None and self.sentinel_base <= 0:
            raise ValueError(f"sentinel_base must be positive or None, got {self.sentinel_base}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """
        Load a ModelConfig from a YAML file.

        Only recognised fields are read from the file - extra keys
        (e.g. checkpoint_schedule) are ignored here, since this class
        only covers what preprocessing/training need.

        :param path: Path to the YAML config file.
        :return: Populated ModelConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)