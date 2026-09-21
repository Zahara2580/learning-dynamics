"""Score generated text against single or multiple references."""

from sacrebleu.metrics import BLEU, CHRF, TER


def transpose_references(references: list[list[str]]) -> list[list[str | None]]:
    """Group references by reference index, padding missing alternatives with None."""
    max_refs = max(len(r) for r in references)
    return [
        [r[k] if k < len(r) else None for r in references] for k in range(max_refs)
    ]


def prediction_diagnostics(preds: list[str], references: list[list[str]]) -> dict:
    """Summarize empty outputs, repetition and prediction lengths."""
    n = len(preds)
    n_empty = sum(1 for p in preds if not p.strip())

    n_repetitive = 0
    for p in preds:
        tokens = p.split()
        if len(tokens) >= 4:
            most_common = max(set(tokens), key=tokens.count)
            if tokens.count(most_common) / len(tokens) > 0.5:
                n_repetitive += 1

    mean_pred_chars = sum(len(p) for p in preds) / n if n else 0.0
    mean_ref_chars = (
        sum(len(r[0]) for r in references) / len(references) if references else 0.0
    )

    return {
        "n_preds": n,
        "n_empty": n_empty,
        "n_repetitive": n_repetitive,
        "mean_pred_chars": round(mean_pred_chars, 2),
        "mean_ref_chars": round(mean_ref_chars, 2),
        "pred_ref_len_ratio": round(mean_pred_chars / mean_ref_chars, 3) if mean_ref_chars else 0.0,
    }


def score_corpus(preds: list[str], references: list[list[str]]) -> dict:
    """Compute corpus metrics and signatures using all available references."""
    assert len(preds) == len(references), (
        f"{len(preds)} predictions vs {len(references)} reference sets"
    )

    refs_t = transpose_references(references)

    bleu_metric = BLEU()
    chrf_metric = CHRF()
    chrfpp_metric = CHRF(word_order=2)
    ter_metric = TER()

    bleu = bleu_metric.corpus_score(preds, refs_t)
    chrf = chrf_metric.corpus_score(preds, refs_t)
    chrfpp = chrfpp_metric.corpus_score(preds, refs_t)
    ter = ter_metric.corpus_score(preds, refs_t)

    return {
        "metrics": {
            "bleu": bleu.score,
            "chrf": chrf.score,
            "chrf_pp": chrfpp.score,
            "ter": ter.score,
        },
        "sacrebleu_signatures": {
            "bleu": str(bleu_metric.get_signature()),
            "chrf": str(chrf_metric.get_signature()),
            "chrf_pp": str(chrfpp_metric.get_signature()),
            "ter": str(ter_metric.get_signature()),
        },
        "diagnostics": prediction_diagnostics(preds, references),
    }
