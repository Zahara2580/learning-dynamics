"""
Tests for ragged multi-reference scoring.

The transposition in metrics.score_corpus is the single easiest place in
this pipeline to get silently wrong numbers, so it is tested directly:
a wrong transposition still returns a plausible float.

Run with pytest, or standalone:
    uv run python3 -m tests.test_metrics
"""

from src.finetuning.metrics import (
    prediction_diagnostics,
    score_corpus,
    transpose_references,
)

# Sentence 0 has 1 reference, sentence 1 has 3 - the ragged case.
PREDS = ["the cat sat on the mat", "hello world"]
REFERENCES = [
    ["the cat sat on the mat"],
    ["hello world", "hi world", "hello earth"],
]


def test_transpose_pads_with_none() -> None:
    """Ragged reference sets transpose to None-padded columns."""
    refs_t = transpose_references(REFERENCES)
    assert len(refs_t) == 3, "should have max_refs = 3 rows"
    assert all(len(row) == 2 for row in refs_t), "each row spans all sentences"
    assert refs_t[0] == ["the cat sat on the mat", "hello world"]
    # Sentence 0 has no 2nd or 3rd reference.
    assert refs_t[1][0] is None
    assert refs_t[2][0] is None
    assert refs_t[1][1] == "hi world"


def test_perfect_prediction_scores_100() -> None:
    """Predictions matching a reference exactly score 100 chrF."""
    result = score_corpus(PREDS, REFERENCES)
    assert result["metrics"]["chrf"] == 100.0
    assert result["metrics"]["chrf_pp"] == 100.0
    assert result["metrics"]["bleu"] > 99.9
    assert result["metrics"]["ter"] == 0.0


def test_signatures_are_populated() -> None:
    """Signatures must be real strings, not silently empty."""
    result = score_corpus(PREDS, REFERENCES)
    for name, sig in result["sacrebleu_signatures"].items():
        assert sig, f"{name} signature is empty"
        assert "version:" in sig, f"{name} signature looks malformed: {sig}"
        # sacreBLEU reports nrefs:var when it sees a variable number of
        # references, which confirms the ragged padding was understood.
        assert "nrefs:var" in sig, f"{name} did not see variable refs: {sig}"


def test_imperfect_prediction_scores_below_100() -> None:
    """A wrong prediction must not score 100 (guards a no-op scorer)."""
    result = score_corpus(["completely different text", "hello world"], REFERENCES)
    assert result["metrics"]["chrf"] < 100.0


def test_second_reference_is_used() -> None:
    """Matching a NON-first reference must still score 100.

    This is what the transposition exists for: if refs_t were built
    wrongly, alternative references would be invisible and this would
    score well below 100.
    """
    result = score_corpus(["the cat sat on the mat", "hello earth"], REFERENCES)
    assert result["metrics"]["chrf"] == 100.0, (
        "3rd reference of sentence 1 was not matched - transposition is broken"
    )


def test_diagnostics_flag_empty_and_repetitive() -> None:
    """Degenerate outputs are counted rather than silently scored."""
    diag = prediction_diagnostics(
        ["", "aa aa aa aa aa", "a normal looking sentence here"],
        [["ref one"], ["ref two"], ["ref three"]],
    )
    assert diag["n_preds"] == 3
    assert diag["n_empty"] == 1
    assert diag["n_repetitive"] == 1


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll metrics tests passed.")
