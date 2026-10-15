"""Model-agnostic evaluation loop. Owner: Member A (step A5).
Works for ANY module whose forward(wav) returns a dict with 'logits_bin' [B,2]."""
from typing import Dict


def run_inference(model, loader, device: str) -> Dict:
    """Return {'utt_id': [...], 'score': np.ndarray, 'label': np.ndarray, 'vocoder_id': np.ndarray}."""
    raise NotImplementedError("TODO(A5)")


def evaluate(model, loader, device: str, save_path: str | None = None) -> Dict:
    """run_inference + metrics.summarize (+ per-vocoder breakdown). Optionally dump per-utterance scores to CSV."""
    raise NotImplementedError("TODO(A5)")
