"""
Machine translation (en -> xh) data loading.

FLORES-200 dev (997) = validation, devtest (1012) = test. FLORES has no
training split by design.

Training corpus (decided 2026-07-24): WMT22 African allenai/wmt22_african
eng-xho. Exact-duplicate (source, target) pairs are removed, then the
top n_train_pairs by laser_score are kept. Deterministic.
"""

import logging
from pathlib import Path
from typing import Iterable, Union

from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# FLORES-200 split names -> our roles. dev is for model selection,
# devtest is the held-out test set everyone reports on.
FLORES_VALIDATION_SPLIT = "dev"
FLORES_TEST_SPLIT = "devtest"

WMT22_DATASET = "allenai/wmt22_african"
WMT22_CONFIG = "eng-xho"


def load_flores_split(
    data_dir: Union[str, Path], split: str
) -> tuple[list[str], list[list[str]]]:
    """
    Load one FLORES-200 split as parallel English/isiXhosa sentences.

    Reads the DatasetDict written by
    src/data_processing/download_finetune_data.py, which stores columns
    (id, source=eng_Latn, target=xho_Latn) already joined on sentence id.

    FLORES is single-reference, but references are returned in the same
    list-of-lists shape as T2X so that both tasks feed the identical
    scoring path in metrics.score_corpus.

    :param data_dir: Directory holding the saved FLORES DatasetDict.
    :param split: "dev" (validation) or "devtest" (test).
    :return: (sources, references) with exactly one reference each.
    """
    dataset = load_from_disk(str(data_dir))
    if split not in dataset:
        raise KeyError(
            f"split {split!r} not in FLORES data at {data_dir} "
            f"(have: {list(dataset.keys())}). Run "
            f"src.data_processing.download_finetune_data --tasks mt first."
        )

    rows = dataset[split]
    sources = [s.strip() for s in rows["source"]]
    references = [[t.strip()] for t in rows["target"]]

    assert len(sources) == len(references), "FLORES source/target misalignment"
    return sources, references


def build_mt_sources(sources: list[str], source_prefix: str, direction_prefix: str) -> list[str]:
    """
    Prepend the task and direction prefixes to every source sentence.

    Adelani et al. (2022a), whose MT hyperparameters this pipeline
    follows, concatenate a direction description to each source at both
    training and test time. Both prefixes are identical for every
    checkpoint.

    :param sources: Raw source sentences.
    :param source_prefix: Optional global task prefix.
    :param direction_prefix: e.g. "Translate English to Xhosa: ".
    :return: Prefixed source strings.
    """
    return [f"{source_prefix}{direction_prefix}{s}" for s in sources]


def select_top_n_unique(
    ranked_pairs: Iterable[dict], n_train_pairs: int
) -> tuple[list[str], list[str], int]:
    """Take the first n unique (eng, xho) pairs from a laser-desc iterable.

    Skips empty pairs and exact (source, target) duplicates, keeping the
    first (= highest-scoring) occurrence. Separated from the download so
    the selection rule is testable without fetching WMT22.
    """
    sources, targets, seen = [], [], set()
    n_dup = 0
    for pair in ranked_pairs:
        s, t = pair["eng"].strip(), pair["xho"].strip()
        if not s or not t:
            continue
        key = (s, t)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        sources.append(s)
        targets.append(t)
        if len(sources) == n_train_pairs:
            break

    if len(sources) < n_train_pairs:
        raise ValueError(
            f"only {len(sources)} unique pairs available, wanted {n_train_pairs}")
    return sources, targets, n_dup


def load_mt_train(cfg) -> tuple[list[str], list[str]]:
    """WMT22 African eng-xho: dedupe exact pairs, then top-N by laser_score.

    Deterministic: same corpus + same N -> same training set, always.
    """
    raw = load_dataset(WMT22_DATASET, WMT22_CONFIG, split="train")

    # Rank by laser_score via argsort on the score column rather than
    # Dataset.sort (whose fingerprinting pickles the whole table and
    # breaks on some Python builds). The generator then pulls only as
    # many rows as the top-N needs, not all 8.7M.
    scores = raw["laser_score"]
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    pairs = (raw[i]["translation"] for i in order)
    sources, targets, n_dup = select_top_n_unique(pairs, cfg.n_train_pairs)

    logger.info(
        f"MT train: kept {len(sources)} pairs "
        f"(skipped {n_dup} exact duplicates while filling top-{cfg.n_train_pairs})")
    return sources, targets
