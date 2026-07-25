"""Tests for the finetuning config, MT training selection, and predictions IO."""

import pytest

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import select_top_n_unique
from src.finetuning.predictions_io import read_predictions, write_predictions


def _mt_cfg(n_train_pairs: int = 10000) -> FinetuneConfig:
    return FinetuneConfig(
        task="mt", data_dir="d", learning_rate=1e-5, batch_size=16,
        num_epochs=3, lr_scheduler_type="linear", n_train_pairs=n_train_pairs)


def _pairs(*items):
    """Build translation dicts in laser-descending order (as WMT22 is sorted)."""
    return [{"eng": e, "xho": x} for e, x in items]


# ---- FIX 1: n_train_pairs is a real, validated, protocol-hashed field ----

def test_mt_requires_positive_n_train_pairs():
    with pytest.raises(ValueError):
        _mt_cfg(n_train_pairs=0)


def test_d2t_does_not_require_n_train_pairs():
    cfg = FinetuneConfig(task="d2t", data_dir="d", learning_rate=1e-4,
                         batch_size=4, num_epochs=5)
    assert cfg.n_train_pairs == 0


def test_n_train_pairs_changes_hash():
    assert _mt_cfg(10000).hash() != _mt_cfg(20000).hash()


def test_mt_yaml_carries_n_train_pairs():
    cfg = FinetuneConfig.from_yaml("configs/finetune/mt.yaml")
    assert cfg.n_train_pairs == 10000


def test_d2t_yaml_still_loads():
    cfg = FinetuneConfig.from_yaml("configs/finetune/d2t.yaml")
    assert cfg.task == "d2t"
    assert cfg.n_train_pairs == 0


# ---- FIX 2: dedupe-then-top-N selection ----

def _pairs_of(kept):
    return [(p["eng"], p["xho"]) for p in kept]


def test_select_skips_exact_duplicates():
    # scores [5,4,4,3]; items 2,3 are exact dups of item 1 -> top-2 = [1,4].
    ranked = _pairs(("A", "a"), ("A", "a"), ("A", "a"), ("B", "b"))
    kept, n_dup = select_top_n_unique(ranked, 2)
    assert _pairs_of(kept) == [("A", "a"), ("B", "b")]
    assert n_dup == 2


def test_select_is_deterministic():
    ranked = _pairs(("A", "a"), ("A", "a"), ("B", "b"), ("C", "c"))
    assert select_top_n_unique(ranked, 3) == select_top_n_unique(ranked, 3)


def test_select_carries_extra_fields():
    # kept items keep their extra keys (e.g. a score), for the inspector.
    ranked = [{"eng": "A", "xho": "a", "laser_score": 1.9},
              {"eng": "B", "xho": "b", "laser_score": 1.5}]
    kept, _ = select_top_n_unique(ranked, 2)
    assert [r["laser_score"] for r in kept] == [1.9, 1.5]


def test_select_skips_empty_sides():
    ranked = _pairs(("", "a"), ("A", "a"), ("B", ""), ("C", "c"))
    kept, _ = select_top_n_unique(ranked, 2)
    assert [p["eng"] for p in kept] == ["A", "C"]


def test_select_raises_when_too_few_unique():
    ranked = _pairs(("A", "a"), ("A", "a"))
    with pytest.raises(ValueError):
        select_top_n_unique(ranked, 2)


# ---- FIX 3: predictions JSONL survives embedded newlines ----

def test_predictions_roundtrip_with_embedded_newline(tmp_path):
    preds = ["hello\nworld", "normal prediction", "", "iAarhus\nsecond line"]
    path = tmp_path / "p.jsonl"
    write_predictions(path, preds)
    recovered = read_predictions(path)
    assert recovered == preds
    assert len(recovered) == len(preds)
