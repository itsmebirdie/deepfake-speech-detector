"""Fixed, class-balanced subsets per vocoder. Owner: Member A (step A6)."""
from typing import List

from .splits import Entry


def build_balanced_subset(
    entries: List[Entry], per_vocoder: int, seed: int, balance_real: bool = True
) -> List[Entry]:
    """Deterministically pick `per_vocoder` clips per vocoder and an equal TOTAL number of bona fide
    clips. Pairs (real clip + its resynthesis) should be kept together where possible so the
    'same speaker/content' property of LibriSeVoc is preserved."""
    raise NotImplementedError("TODO(A6)")


def save_manifest(entries: List[Entry], path: str) -> None:
    raise NotImplementedError("TODO(A6)")
