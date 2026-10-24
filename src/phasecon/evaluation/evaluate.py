"""Model-agnostic evaluation loop. Owner: Member A (step A5).
Works for ANY module whose forward(wav) returns a dict with 'logits_bin' [B,2]
(repo order: index 0 = bona fide, index 1 = synthetic; docs/INTERFACES.md section 5)."""
import csv
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .metrics import summarize


@torch.inference_mode()
def run_inference(model, loader, device: str) -> Dict:
    """Return {'utt_id': [...], 'score': np.ndarray, 'label': np.ndarray, 'vocoder_id': np.ndarray}.

    Uses ONLY out['logits_bin']: score = logits_bin[:, 1] - logits_bin[:, 0] (higher = more synthetic).
    """
    was_training = model.training
    model.eval()
    utt_ids, scores, labels, vocoder_ids = [], [], [], []
    try:
        for batch in loader:
            out = model(batch["waveform"].to(device))
            logits = out["logits_bin"].float()
            if logits.dim() != 2 or logits.shape[1] != 2:
                raise ValueError(f"logits_bin must be [B, 2], got {tuple(logits.shape)}")
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            vocoder_ids.append(batch["vocoder_id"].cpu().numpy())
            utt_ids.extend(batch["utt_id"])
    finally:
        model.train(was_training)
    return {
        "utt_id": utt_ids,
        "score": np.concatenate(scores),
        "label": np.concatenate(labels).astype(int),
        "vocoder_id": np.concatenate(vocoder_ids).astype(int),
    }


def save_scores(result: Dict, path: str) -> None:
    """One row per clip: utt_id, label, vocoder_id, score."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "vocoder_id", "score"])
        for u, lb, v, s in zip(result["utt_id"], result["label"], result["vocoder_id"], result["score"]):
            w.writerow([u, int(lb), int(v), f"{s:.6f}"])


def evaluate(
    model, loader, device: str, save_path: Optional[str] = None, vocoders: Optional[Sequence[str]] = None
) -> Dict:
    """run_inference + metrics.summarize (+ per-vocoder breakdown). Optionally dump per-utterance scores to CSV.

    `vocoders` = config `data.vocoders`, used to name the per-vocoder rows.
    """
    result = run_inference(model, loader, device)
    metrics = summarize(result["score"], result["label"], result["vocoder_id"], vocoders)
    if save_path:
        save_scores(result, save_path)
    return metrics
