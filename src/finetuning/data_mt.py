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
    data_dir: Union[str, Path], split: str, direction: str = "en-xh"
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
    # The saved DatasetDict has source=eng_Latn, target=xho_Latn; xh-en
    # simply swaps which side is input and which is reference.
    if direction == "en-xh":
        sources = [s.strip() for s in rows["source"]]
        references = [[t.strip()] for t in rows["target"]]
    else:
        sources = [t.strip() for t in rows["target"]]
        references = [[s.strip()] for s in rows["source"]]

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


def _pair_of(item: dict) -> tuple[str, str]:
    """Default (eng, xho) extractor: a WMT22 row's translation sub-dict."""
    tr = item["translation"] if "translation" in item else item
    return tr["eng"], tr["xho"]


def select_top_n_unique(
    ranked_items: Iterable[dict], n_train_pairs: int, get_pair=_pair_of
) -> tuple[list[dict], int]:
    """First n items whose (eng, xho) pair is non-empty and not a duplicate.

    Returns the KEPT ITEMS themselves (in laser-desc order), carrying any
    extra fields such as laser_score, plus the count of exact duplicates
    skipped. Callers extract what they need - the single source of the
    selection rule, so load_mt_train and the inspector can never drift.
    """
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


def rank_by_laser(raw) -> "Iterable[dict]":
    """Yield WMT22 rows in descending laser_score order.

    argsort on the score column rather than Dataset.sort (whose
    fingerprinting pickles the whole table and breaks on some Python
    builds); the generator pulls only as many rows as the caller consumes,
    not all 8.7M.
    """
    scores = raw["laser_score"]
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return (raw[i] for i in order)


def load_mt_train(cfg) -> tuple[list[str], list[str]]:
    """WMT22 African eng-xho: dedupe exact pairs, then top-N by laser_score.

    Deterministic: same corpus + same N -> same training set, always.
    """
    raw = load_dataset(WMT22_DATASET, WMT22_CONFIG, split="train")
    kept, n_dup = select_top_n_unique(rank_by_laser(raw), cfg.n_train_pairs)
    # Same top-N pairs whichever way round: dedup + ranking ignore
    # direction, so en-xh and xh-en train on identical sentence pairs.
    src_key, tgt_key = ("eng", "xho") if cfg.direction == "en-xh" else ("xho", "eng")
    sources = [row["translation"][src_key].strip() for row in kept]
    targets = [row["translation"][tgt_key].strip() for row in kept]

    logger.info(
        f"MT train: kept {len(sources)} pairs "
        f"(skipped {n_dup} exact duplicates while filling top-{cfg.n_train_pairs})")
    return sources, targets
