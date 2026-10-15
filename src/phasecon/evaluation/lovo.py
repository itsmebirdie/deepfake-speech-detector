"""Leave-one-vocoder-out driver. Owner: Member A (step A7 for baseline, A9 for PhaseCon-Net)."""
from typing import Dict, List


def run_lovo(cfg: Dict, model_name: str, vocoders: List[str]) -> Dict:
    """For each held-out vocoder: build split -> train fresh model (baseline arch or PhaseCon-Net) -> evaluate.
    Returns {vocoder: metrics}. Baseline must be RETRAINED per fold (released weights saw all vocoders)."""
    raise NotImplementedError("TODO(A7)")
