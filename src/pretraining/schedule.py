"""Define dense early checkpoints (the first 10 percent) and 
    sparse later checkpoints (remaining 90 percent)"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointScheduleConfig:
    """Configure checkpoint intervals for the early and late training phases"""
    initial_phase_ratio: float = 0.10
    initial_interval_ratio: float = 0.01
    final_interval_ratio: float = 0.10

    def __post_init__(self) -> None:
        """Validate configuration values """

        if not 0 < self.initial_phase_ratio < 1:
            raise ValueError(f"initial_phase_ratio must be in (0, 1), got {self.initial_phase_ratio}")
        if not 0 < self.initial_interval_ratio <= self.initial_phase_ratio:
            raise ValueError(
                f"initial_interval_ratio must be in (0, initial_phase_ratio], got {self.initial_interval_ratio}"
            )
        if not 0 < self.final_interval_ratio <= 1:
            raise ValueError(f"final_interval_ratio must be in (0, 1], got {self.final_interval_ratio}")


def compute_checkpoint_steps(total_steps: int,config: Optional[CheckpointScheduleConfig]=None,) -> list[int]:
    """Return the scheduled checkpoint steps, including the final step"""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    if config is None:
        config = CheckpointScheduleConfig()

    checkpoint_steps = []

    initial_interval = max(1, int(total_steps * config.initial_interval_ratio))
    initial_phase_end = int(total_steps * config.initial_phase_ratio)
    checkpoint_steps.extend(
        range(initial_interval, initial_phase_end + 1, initial_interval)
    )

    final_interval = max(1, int(total_steps * config.final_interval_ratio))
    final_phase_start = initial_phase_end + final_interval
    checkpoint_steps.extend(
        range(final_phase_start, total_steps + 1, final_interval)
    )

    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)

    return sorted(set(checkpoint_steps))


def log_checkpoint_schedule(total_steps: int, steps: list[int]) -> None:
    """Log each checkpoint step and its share of total training."""
    logger.info(f"Checkpoint schedule: {len(steps)} checkpoints over {total_steps:} steps")
    for i, step in enumerate(steps):
        pct = (step / total_steps) * 100
        logger.info(f"Checkpoint {i + 1}: step {step} ({pct:.0f}%)")


def main() -> None:
    """Print the checkpoint schedule for a 10000 step run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    steps = compute_checkpoint_steps(total_steps=10_000)
    log_checkpoint_schedule(10_000, steps)


if __name__ == "__main__":
    main()
