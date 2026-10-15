"""Metrics. Owner: Member A (step A5).
Score convention: higher score = more likely SYNTHETIC (label 1). Document how baseline logits map to this."""
import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (eer, threshold). Must agree with the baseline repo's EER on identical scores (test it)."""
    raise NotImplementedError("TODO(A5)")


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("TODO(A5)")


def compute_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    raise NotImplementedError("TODO(A5)")


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict:
    """{'eer','eer_threshold','auc','accuracy'}; accuracy at the EER threshold AND at the natural 0.5 point."""
    raise NotImplementedError("TODO(A5)")
