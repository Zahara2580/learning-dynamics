"""
Corpus scoring with sacreBLEU, including ragged multi-reference handling.

THE TRANSPOSITION IS THE DANGEROUS PART. sacreBLEU's corpus functions
want references grouped BY REFERENCE INDEX, not by sentence:

    refs_t[k][j] = the k-th reference of the j-th sentence

Sentences with fewer references than the maximum are padded with None.
Passing the natural per-sentence nesting instead produces numbers that
look plausible and are wrong, with no error raised. The signature string
recorded alongside each score reports 'nrefs:var', which is sacreBLEU
confirming it saw a variable number of references - a useful check that
the transposition did what we think.

Predictions and references are fed to sacreBLEU as RAW STRINGS. No
sentence splitting, lowercasing, detokenising, or normalisation:
sacreBLEU does its own tokenisation internally, and pre-processing here
would both break comparability with published baselines and make the
recorded signature a lie.
"""

from sacrebleu.metrics import BLEU, CHRF, TER


def transpose_references(references: list[list[str]]) -> list[list[str | None]]:
    """
    Transpose per-sentence reference lists into sacreBLEU's layout.

    :param references: references[j] = list of refs for sentence j.
    :return: refs_t[k][j] = k-th ref of sentence j, None-padded where
        sentence j has fewer than k+1 references.
    """
    max_refs = max(len(r) for r in references)
    return [
        [r[k] if k < len(r) else None for r in references] for k in range(max_refs)
    ]


def prediction_diagnostics(preds: list[str], references: list[list[str]]) -> dict:
    """
    Cheap degenerate-output detectors.

    A checkpoint early in CPT can emit empty strings or repeat a single
    token forever. Such output can still post a non-trivial chrF, so
    without these counters a broken point on the learning-dynamics curve
    looks merely low rather than degenerate.

    :param preds: Generated strings.
    :param references: Reference lists, used only for a length baseline.
    :return: Diagnostic counters and length statistics.
    """
    n = len(preds)
    n_empty = sum(1 for p in preds if not p.strip())

    # Crude repetition detector: fraction of predictions where a single
    # whitespace token accounts for over half the tokens produced.
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
        # Ratio far from 1.0 means systematic over/under-generation,
        # which is the signature of a truncating max_new_tokens.
        "pred_ref_len_ratio": round(mean_pred_chars / mean_ref_chars, 3) if mean_ref_chars else 0.0,
    }


def score_corpus(preds: list[str], references: list[list[str]]) -> dict:
    """
    Score predictions against ragged multi-reference sets.

    Uses sacreBLEU's class-based API rather than the corpus_bleu()
    convenience functions. The scores are identical; the difference is
    that only the metric objects expose get_signature(), so the
    functional API would force us to record empty signature strings.

    :param preds: Generated strings, one per sentence.
    :param references: references[j] = list of refs for sentence j.
    :return: Scores, signature strings, and degenerate-output counters.
    """
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
