import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointScheduleConfig:
    """
    Configuration for the two-phase checkpoint schedule.

    Attributes:
        initial_phase_ratio: Fraction of total steps in the initial dense checkpoint phase.
        initial_interval_ratio: Fraction of total steps between checkpoints during the initial phase.
        final_interval_ratio: Fraction of total steps between checkpoints during the final phase.
    """
    initial_phase_ratio: float = 0.10
    initial_interval_ratio: float = 0.01
    final_interval_ratio: float = 0.10

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if not 0 < self.initial_phase_ratio < 1:
            raise ValueError(f"initial_phase_ratio must be in (0, 1), got {self.initial_phase_ratio}")
        if not 0 < self.initial_interval_ratio <= self.initial_phase_ratio:
            raise ValueError(
                f"initial_interval_ratio must be in (0, initial_phase_ratio], got {self.initial_interval_ratio}"
            )
        if not 0 < self.final_interval_ratio <= 1:
            raise ValueError(f"final_interval_ratio must be in (0, 1], got {self.final_interval_ratio}")


def compute_checkpoint_steps(total_steps: int, config: Optional[CheckpointScheduleConfig] = None) -> list[int]:
    """
    Compute the training steps at which checkpoints should be saved.

    :param total_steps: Total number of training steps.
    :param config: Checkpoint schedule configuration. Uses default two-phase schedule if not provided.
    :return: Sorted list of steps at which to save checkpoints.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    if config is None:
        config = CheckpointScheduleConfig()

    checkpoint_steps = []

    # Phase 1: dense checkpointing during the initial phase
    initial_interval = max(1, int(total_steps * config.initial_interval_ratio))
    initial_phase_end = int(total_steps * config.initial_phase_ratio)
    checkpoint_steps.extend(
        range(initial_interval, initial_phase_end + 1, initial_interval)
    )

    # Phase 2: sparser checkpointing for the remaining steps. Starting
    # from initial_phase_end + final_interval avoids double-counting
    # the step already checkpointed at the end of phase 1.
    final_interval = max(1, int(total_steps * config.final_interval_ratio))
    final_phase_start = initial_phase_end + final_interval
    checkpoint_steps.extend(
        range(final_phase_start, total_steps + 1, final_interval)
    )

    # Guarantee the final step is always checkpointed, even if the
    # interval arithmetic above doesn't land exactly on it.
    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)

    return sorted(set(checkpoint_steps))


def log_checkpoint_schedule(total_steps: int, steps: list[int]) -> None:
    """
    Log a human-readable summary of the checkpoint schedule.

    :param total_steps: Total number of training steps.
    :param steps: List of checkpoint steps.
    """
    logger.info(f"Checkpoint schedule: {len(steps)} checkpoints over {total_steps:,} steps")
    for i, step in enumerate(steps):
        pct = (step / total_steps) * 100
        logger.info(f"  Checkpoint {i + 1:>2}: step {step:>7,} ({pct:.0f}%)")


def main() -> None:
    """Standalone entry point for sanity-checking the checkpoint schedule."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Sanity check for encoder-decoder models (10k steps, per the proposal)
    steps = compute_checkpoint_steps(total_steps=10_000)
    log_checkpoint_schedule(10_000, steps)


if __name__ == "__main__":
    main()