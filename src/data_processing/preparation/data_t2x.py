"""Load T2X triples and their reference verbalisations."""

from pathlib import Path
from typing import Union

SEPARATOR = "*#"


def load_t2x_split(data_dir: Union[str, Path], split: str) -> tuple[list[str], list[list[str]]]:
    """Load aligned T2X inputs and reference lists without changing their encoding."""
    data_dir = Path(data_dir)
    data_lines = (data_dir / f"{split}.data").read_text(encoding="utf-8").splitlines()
    text_lines = (data_dir / f"{split}.text").read_text(encoding="utf-8").splitlines()

    assert len(data_lines) == len(text_lines), (
        f"{split}: {len(data_lines)} inputs vs {len(text_lines)} targets - alignment broken"
    )

    inputs = [line.strip() for line in data_lines]
    references = [
        [ref.strip() for ref in line.split(SEPARATOR)] for line in text_lines
    ]

    if split == "train":
        assert all(len(r) == 1 for r in references), "train should be single-reference"

    return inputs, references


def build_training_pairs(inputs: list[str], references: list[list[str]], source_prefix: str='') -> tuple[list[str], list[str]]:
    """Pair each prefixed input with its first reference for teacher-forced training."""
    sources = [source_prefix + inp for inp in inputs]
    targets = [refs[0] for refs in references]
    return sources, targets
