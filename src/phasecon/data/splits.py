"""Split-file parsing and leave-one-vocoder-out (LOVO) split construction. Owner: Member A."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Entry:
    path: str          # relative to data.root
    utt_id: str        # LibriTTS-style id (reader_chapter_...)
    label: int         # 0 bona fide, 1 synthetic
    vocoder_id: int    # 0 bona fide, 1..6 vocoders


def read_split_file(split_file: str, root: str, vocoders: List[str]) -> List[Entry]:
    """Parse an official train/dev/test .txt file into Entry objects (step A3)."""
    raise NotImplementedError("TODO(A3)")


def build_lovo_splits(
    entries_by_split: Dict[str, List[Entry]], held_out: str, vocoders: List[str]
) -> Dict[str, List[Entry]]:
    """LOVO protocol (step A7), to be written down in docs/DATASET_NOTES.md:
    - train/dev: bona fide + all vocoders EXCEPT `held_out`
    - test: bona fide test clips + `held_out` fakes only (report pooled and per-vocoder EER)
    Must assert zero overlap of utt_id between train/dev and test where the official split guarantees it.
    """
    raise NotImplementedError("TODO(A7)")


def write_split(entries: List[Entry], out_path: str) -> None:
    raise NotImplementedError("TODO(A7)")
