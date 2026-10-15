from typing import Dict, List


def collate_fn(batch: List[Dict]) -> Dict:
    """Stack into {waveform [B,T], label [B], vocoder_id [B], utt_id list[str]}. Owner: Member A (A3)."""
    raise NotImplementedError("TODO(A3)")
