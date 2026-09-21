"""Load English–isiXhosa translation data for training and evaluation."""

import logging
from pathlib import Path
from typing import Iterable, Union

from datasets import load_from_disk

from src.data_processing.downloads.wmt22 import load_wmt22_train

logger = logging.getLogger(__name__)

FLORES_VALIDATION_SPLIT = "dev"
FLORES_TEST_SPLIT = "devtest"


def load_flores_split(data_dir: Union[str, Path], split: str, direction: str='en-xh') -> tuple[list[str], list[list[str]]]:
    """Load a FLORES split in the requested translation direction."""
    dataset = load_from_disk(str(data_dir))
    if split not in dataset:
        raise KeyError(
            f"split {split!r} not in FLORES data at {data_dir} "
            f"(have: {list(dataset.keys())}). Provide a prepared FLORES DatasetDict "
            f"with dev and devtest splits."
        )

    rows = dataset[split]
    if direction == "en-xh":
        sources = [s.strip() for s in rows["source"]]
        references = [[t.strip()] for t in rows["target"]]
    else:
        sources = [t.strip() for t in rows["target"]]
        references = [[s.strip()] for s in rows["source"]]

    assert len(sources) == len(references), "FLORES source/target misalignment"
    return sources, references


def build_mt_sources(sources: list[str], source_prefix: str, direction_prefix: str) -> list[str]:
    """Prepend task and direction prefixes to each source sentence."""
    return [f"{source_prefix}{direction_prefix}{s}" for s in sources]


def _pair_of(item: dict) -> tuple[str, str]:
    """Extract the English and isiXhosa texts from a translation record."""
    tr = item["translation"] if "translation" in item else item
    return tr["eng"], tr["xho"]


def select_top_n_unique(ranked_items: Iterable[dict], n_train_pairs: int, get_pair=_pair_of) -> tuple[list[dict], int]:
    """Select the first N nonempty, unique sentence pairs and count skipped duplicates."""
    kept, seen, n_dup = [], set(), 0
    for item in ranked_items:
        s, t = get_pair(item)
        s, t = s.strip(), t.strip()
        if not s or not t:
            continue
        key = (s, t)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append(item)
        if len(kept) == n_train_pairs:
            break

    if len(kept) < n_train_pairs:
        raise ValueError(
            f"only {len(kept)} unique pairs available, wanted {n_train_pairs}")
    return kept, n_dup


def rank_by_laser(raw) -> 'Iterable[dict]':
    """Return translation records in descending LASER score order."""
    scores = raw["laser_score"]
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return (raw[i] for i in order)


def load_mt_train(cfg) -> tuple[list[str], list[str]]:
    """Load the highest ranked unique training pairs in the requested direction."""
    raw = load_wmt22_train()
    kept, n_dup = select_top_n_unique(rank_by_laser(raw), cfg.n_train_pairs)
    src_key, tgt_key = ("eng", "xho") if cfg.direction == "en-xh" else ("xho", "eng")
    sources = [row["translation"][src_key].strip() for row in kept]
    targets = [row["translation"][tgt_key].strip() for row in kept]

    logger.info(
        f"MT train: kept {len(sources)} pairs "
        f"(skipped {n_dup} exact duplicates while filling top-{cfg.n_train_pairs})")
    return sources, targets
