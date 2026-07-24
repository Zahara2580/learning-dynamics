"""
T2X (isiXhosa data-to-text) data loading.

Six flat line-aligned files from github.com/francois-meyer/t2x:
{train,valid,test}.{data,text}, where line i of .data is the input for
line i of .text. Counts: train 3859, valid 460, test 378.

Two properties are easy to get silently wrong. Multiple references are
packed into ONE line separated by the literal string '*#', so a naive
reader scores a 3-reference line as one long reference. And the files
contain mojibake, which is passed through unchanged - the published
baselines were scored on these exact bytes.
"""

from pathlib import Path
from typing import Union

# Multi-reference separator used inside a single line of .text.
MULTI_REF_SEPARATOR = "*#"


def load_t2x_split(
    data_dir: Union[str, Path], split: str
) -> tuple[list[str], list[list[str]]]:
    """
    Load one T2X split as line-aligned inputs and reference lists.

    Files are read as UTF-8 and passed through unmodified (see module
    docstring on mojibake). splitlines() is used rather than a line
    count from the shell because the files lack a trailing newline.

    :param data_dir: Directory containing {split}.data and {split}.text.
    :param split: One of "train", "valid", "test".
    :return: (inputs, references) where references[i] is the list of
        1-3 acceptable outputs for inputs[i].
    """
    data_dir = Path(data_dir)
    data_lines = (data_dir / f"{split}.data").read_text(encoding="utf-8").splitlines()
    text_lines = (data_dir / f"{split}.text").read_text(encoding="utf-8").splitlines()

    assert len(data_lines) == len(text_lines), (
        f"{split}: {len(data_lines)} inputs vs {len(text_lines)} targets - alignment broken"
    )

    inputs = [line.strip() for line in data_lines]
    references = [
        [ref.strip() for ref in line.split(MULTI_REF_SEPARATOR)] for line in text_lines
    ]

    if split == "train":
        # Training must be single-reference: the training pair builder
        # below takes references[i][0], so a packed multi-reference line
        # in train would silently discard supervision.
        assert all(len(r) == 1 for r in references), "train should be single-reference"

    return inputs, references


def build_training_pairs(
    inputs: list[str], references: list[list[str]], source_prefix: str = ""
) -> tuple[list[str], list[str]]:
    """
    Build (source, target) string pairs for training or validation loss.

    Only the FIRST reference is used as the target. Test-time scoring
    uses all references; teacher forcing needs exactly one.

    :param inputs: Linearised triple strings from load_t2x_split.
    :param references: Reference lists from load_t2x_split.
    :param source_prefix: Prefix prepended to every source. Identical
        for every checkpoint.
    :return: (sources, targets), both length len(inputs).
    """
    sources = [source_prefix + inp for inp in inputs]
    targets = [refs[0] for refs in references]
    return sources, targets
