"""Metrics. Owner: Member A (step A5). The single place every comparison takes its numbers from.

Score convention (docs/INTERFACES.md section 5): higher score = more likely SYNTHETIC (label 1).
For a model with `logits_bin` [B, 2] in repo order [bona fide, synthetic]:
    score = logits_bin[:, 1] - logits_bin[:, 0]
so the natural decision point (both classes equally likely) is score = 0.
For the released baseline (log-softmax over [synthetic, bona fide]) the same score is
logp[:, 0] - logp[:, 1]; see docs/DATASET_NOTES.md section 6.

The baseline repo has no EER code (DATASET_NOTES section 6), so there is no baseline function to
match. `compute_eer` uses the ROC curve; tests/test_metrics.py checks it against an independent
implementation of the ASVspoof challenge formula (detection-error curve), and PR-06's numbers
were produced with the same ROC method.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

REAL, FAKE = 0, 1
NATURAL_THRESHOLD = 0.0  # score 0 = both classes equally likely


def _check(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} differ in length")
    if not set(np.unique(labels)) <= {REAL, FAKE}:
        raise ValueError("labels must be 0 (bona fide) or 1 (synthetic)")
    if not (labels == REAL).any() or not (labels == FAKE).any():
        raise ValueError("EER and AUC need at least one bona fide and one synthetic clip")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    return scores, labels


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (eer, threshold).

    EER = the error rate at the threshold where false alarms (bona fide called synthetic) equal
    misses (synthetic called bona fide). A clip is called synthetic when score >= threshold.
    """
    scores, labels = _check(scores, labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=FAKE, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    threshold = float(thresholds[i])
    if not np.isfinite(threshold):  # sklearn's first point is +inf ("call nothing synthetic")
        threshold = float(scores.max()) + 1.0
    return float((fpr[i] + fnr[i]) / 2), threshold


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: the chance a random synthetic clip scores above a random bona fide one."""
    scores, labels = _check(scores, labels)
    return float(roc_auc_score(labels, scores))


def compute_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Share of clips classified correctly when 'synthetic' means score >= threshold."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} differ in length")
    if len(scores) == 0:
        raise ValueError("no scores")
    return float(((scores >= threshold).astype(int) == labels).mean())


def summarize(
    scores: np.ndarray,
    labels: np.ndarray,
    vocoder_ids: Optional[np.ndarray] = None,
    vocoders: Optional[Sequence[str]] = None,
) -> Dict:
    """{'eer','eer_threshold','auc','accuracy','accuracy_at_eer'} (+ 'per_vocoder' when vocoder_ids given).

    `accuracy` is at the natural point (score 0), `accuracy_at_eer` at the EER threshold.
    Per vocoder: EER and AUC of that vocoder's fakes against ALL bona fide clips, plus the share
    of its fakes caught at the natural point.
    """
    scores, labels = _check(scores, labels)
    eer, thr = compute_eer(scores, labels)
    out: Dict = {
        "n_clips": int(len(scores)),
        "eer": eer,
        "eer_threshold": thr,
        "auc": compute_auc(scores, labels),
        "accuracy": compute_accuracy(scores, labels, NATURAL_THRESHOLD),
        "accuracy_at_eer": compute_accuracy(scores, labels, thr),
    }
    if vocoder_ids is not None:
        vocoder_ids = np.asarray(vocoder_ids).ravel().astype(int)
        real = vocoder_ids == 0
        ids: List[int] = sorted(int(v) for v in np.unique(vocoder_ids) if v != 0)
        per = {}
        for vid in ids:
            name = vocoders[vid - 1] if vocoders is not None and vid - 1 < len(vocoders) else f"vocoder_{vid}"
            sel = real | (vocoder_ids == vid)
            v_eer, _ = compute_eer(scores[sel], labels[sel])
            per[name] = {
                "eer": v_eer,
                "auc": compute_auc(scores[sel], labels[sel]),
                "caught": float((scores[vocoder_ids == vid] >= NATURAL_THRESHOLD).mean()),
                "n_fake": int((vocoder_ids == vid).sum()),
            }
        out["per_vocoder"] = per
        out["bona_fide_correct"] = float((scores[real] < NATURAL_THRESHOLD).mean())
    return out
