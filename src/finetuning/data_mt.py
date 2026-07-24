"""
Machine translation (en -> xh) data loading.

FLORES-200 dev (997 sentences) is the validation set, devtest (1012) the
test set. FLORES has no training split: it is an n-way parallel
benchmark whose cross-language comparability depends on nobody training
on it.

The training corpus is not yet chosen - see load_mt_train. MAFAND has no
en-xho train split (isiXhosa ships as validation/test only);
allenai/wmt22_african eng-xho does, but is 8.7M LASER-mined pairs with
loose alignments and needs filtering on its laser_score column.
"""

from pathlib import Path
from typing import Union

from datasets import load_from_disk

# FLORES-200 split names -> our roles. dev is for model selection,
# devtest is the held-out test set everyone reports on.
FLORES_VALIDATION_SPLIT = "dev"
FLORES_TEST_SPLIT = "devtest"


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


def load_mt_train(cfg) -> tuple[list[str], list[str]]:
    """
    Load the MT training corpus.

    NOT IMPLEMENTED PENDING SUPERVISOR DECISION. The two candidates are
    sketched below; both are a few lines, but the choice is
    methodological rather than technical and changes what the MT results
    mean, so it is not made here.

    :param cfg: FinetuneConfig for the MT task.
    :return: (sources, targets) parallel training strings.
    :raises NotImplementedError: always, until the corpus is chosen.
    """
    raise NotImplementedError(
        "awaiting corpus decision - see the two candidate implementations "
        "in the body of src/finetuning/data_mt.py::load_mt_train"
    )

    # ---------------------------------------------------------------
    # CANDIDATE (a): WMT22 African, eng-xho.
    #
    # 8.7M LASER-mined pairs. Needs BOTH a quality filter and a
    # subsample: at batch size 16 for 3 epochs the full corpus is ~1.6M
    # optimiser steps per checkpoint, times 20 checkpoints times 3
    # models. Not feasible, and most of the tail is noise anyway.
    #
    # laser_score > 1.06 is the NLLB bitext-mining threshold. n=50_000
    # gives ~9.4k steps per finetune, in the range Adelani et al. found
    # sufficient ("a few thousand translations go a long way").
    #
    # from datasets import load_dataset
    # raw = load_dataset("allenai/wmt22_african", "eng-xho", split="train")
    # kept = raw.filter(lambda ex: ex["laser_score"] > cfg.laser_threshold)
    # kept = kept.shuffle(seed=42).select(range(cfg.n_train_pairs))
    # sources = [ex["eng"].strip() for ex in kept["translation"]]
    # targets = [ex["xho"].strip() for ex in kept["translation"]]
    # return sources, targets
    #
    # ---------------------------------------------------------------
    # CANDIDATE (b): Autshumato en-xh, from SADiLaR.
    #
    # Human-translated South African government text - the corpus Meyer
    # et al. actually used for English -> Nguni. Cleaner than mined
    # bitext and needs no quality filtering, but is domain-narrow and
    # must be downloaded manually (its HF copy uses a loading script the
    # datasets server cannot introspect).
    #
    # Line-aligned plain text, so it reuses the T2X read/strip/assert
    # pattern exactly:
    #
    # src_path = Path(cfg.data_dir) / "autshumato.en"
    # tgt_path = Path(cfg.data_dir) / "autshumato.xh"
    # sources = src_path.read_text(encoding="utf-8").splitlines()
    # targets = tgt_path.read_text(encoding="utf-8").splitlines()
    # assert len(sources) == len(targets), (
    #     f"Autshumato: {len(sources)} en vs {len(targets)} xh - alignment broken")
    # return [s.strip() for s in sources], [t.strip() for t in targets]
    # ---------------------------------------------------------------
