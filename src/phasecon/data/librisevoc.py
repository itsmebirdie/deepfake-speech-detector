"""LibriSeVoc dataset. Owner: Member A (step A3). Contract: docs/INTERFACES.md section 1."""
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .splits import Entry


class LibriSeVocDataset(Dataset):
    """Returns dict:
        waveform:   FloatTensor [num_samples]  (16 kHz, mono)
        label:      LongTensor  []             0 bona fide / 1 synthetic
        vocoder_id: LongTensor  []             0 bona fide / 1..6
        utt_id:     str
    """

    def __init__(
        self,
        entries: List[Entry],
        num_samples: int,
        sample_rate: int = 16000,
        train: bool = False,
    ):
        raise NotImplementedError("TODO(A3)")

    def __len__(self) -> int:
        raise NotImplementedError("TODO(A3)")

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError("TODO(A3)")


def build_dataloader(cfg: Dict, split: str, train: bool, entries: Optional[List[Entry]] = None) -> DataLoader:
    """Resolve entries from cfg (official split, subset manifest, or LOVO split) and wrap in a DataLoader."""
    raise NotImplementedError("TODO(A3)")
