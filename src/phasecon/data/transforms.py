"""Waveform transforms. Owner: Member A (step A3)."""
import torch


def pad_or_crop(wav: torch.Tensor, num_samples: int, random_crop: bool = False) -> torch.Tensor:
    """Return a 1-D tensor of exactly `num_samples` samples, treated the way the baseline treats clips.

    Baseline reference: `pad()` in the authors' main.py / eval.py (see docs/DATASET_NOTES.md section 6).
    - Long clip: keep the FIRST `num_samples` samples. The baseline never random-crops, so
      `random_crop` is off by default; set it to True only for an explicit augmentation experiment.
    - Short clip: repeat the whole clip end to end until it is long enough, then cut
      (same as the baseline's `np.tile(x, int(max_len / len(x)) + 1)[:max_len]`).
    """
    if wav.dim() != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {tuple(wav.shape)}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    n = wav.shape[0]
    if n == 0:
        raise ValueError("cannot pad an empty waveform")
    if n >= num_samples:
        start = 0
        if random_crop and n > num_samples:
            start = int(torch.randint(0, n - num_samples + 1, (1,)))
        return wav[start:start + num_samples]
    repeats = num_samples // n + 1
    return wav.repeat(repeats)[:num_samples]


def peak_normalize(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale so the loudest sample has magnitude 1.

    The baseline does NOT normalise (it feeds librosa's raw float audio), so the dataset does not
    call this. It is here for experiments that need it.
    """
    return wav / (wav.abs().max() + eps)
