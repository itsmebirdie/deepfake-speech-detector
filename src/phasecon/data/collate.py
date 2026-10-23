from typing import Dict, List

import torch


def collate_fn(batch: List[Dict]) -> Dict:
    """Stack into {waveform [B,T], label [B], vocoder_id [B], utt_id list[str]}. Owner: Member A (A3)."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    return {
        "waveform": torch.stack([item["waveform"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "vocoder_id": torch.stack([item["vocoder_id"] for item in batch]),
        "utt_id": [item["utt_id"] for item in batch],
    }
