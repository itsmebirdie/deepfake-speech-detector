"""LibriSeVoc dataset. Owner: Member A (step A3). Contract: docs/INTERFACES.md section 1."""
import math
import os
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from .collate import collate_fn
from .splits import Entry, read_split_file
from .subset import load_manifest
from .transforms import pad_or_crop


class LibriSeVocDataset(Dataset):
    """Returns dict:
        waveform:   FloatTensor [num_samples]  (mono, at `sample_rate`; files are 24 kHz and are
                    resampled only if `sample_rate` differs, see DATASET_NOTES.md section 7)
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
        root: str = "data/raw/LibriSeVoc",
        random_crop: bool = False,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.entries = list(entries)
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.train = train
        self.root = root
        # The baseline always keeps the first num_samples, so random cropping stays off unless asked for.
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.entries)

    def load_waveform(self, path: str) -> torch.Tensor:
        """Read a WAV file as mono float32 at `self.sample_rate`."""
        with sf.SoundFile(path) as f:
            file_rate = f.samplerate
            frames = -1  # whole file
            if not self.random_crop:
                # Only the start of the clip is used, so read just enough (+0.1 s margin for resampling).
                frames = math.ceil(self.num_samples * file_rate / self.sample_rate) + file_rate // 10
            audio = f.read(frames, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if file_rate != self.sample_rate:
            g = math.gcd(file_rate, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, file_rate // g).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(audio))

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        wav = self.load_waveform(os.path.join(self.root, entry.path))
        wav = pad_or_crop(wav, self.num_samples, random_crop=self.random_crop)
        return {
            "waveform": wav,
            "label": torch.tensor(entry.label, dtype=torch.long),
            "vocoder_id": torch.tensor(entry.vocoder_id, dtype=torch.long),
            "utt_id": entry.utt_id,
        }


def build_dataloader(cfg: Dict, split: str, train: bool, entries: Optional[List[Entry]] = None) -> DataLoader:
    """Resolve entries from cfg (official split, subset manifest, or LOVO split) and wrap in a DataLoader.

    With `data.subset.enabled: true`, a split whose `per_vocoder_<split>` is set reads
    `<manifest_dir>/<split>.txt`; a split left at null uses the full official split.

    Needs `cfg["data"]` (configs/data/librisevoc.yaml) and `cfg["audio"]` (configs/default.yaml).
    Batch size comes from `cfg["train"]["batch_size"]` (default 32). Shuffling happens only when
    `train=True`, with a generator seeded from `cfg["seed"]` so runs are repeatable.
    """
    data_cfg = cfg["data"]
    audio_cfg = cfg["audio"]
    if entries is None:
        if data_cfg.get("lovo", {}).get("held_out"):
            raise NotImplementedError("LOVO splits are added in PR-13")
        subset = data_cfg.get("subset", {})
        if subset.get("enabled") and subset.get(f"per_vocoder_{split}") is not None:
            # balanced subset manifest built by scripts/make_subset.py (DATASET_NOTES.md section 9)
            entries = load_manifest(os.path.join(subset.get("manifest_dir", "data/subsets"), f"{split}.txt"))
        else:
            entries = read_split_file(data_cfg["split_files"][split], data_cfg["root"], data_cfg["vocoders"])
    dataset = LibriSeVocDataset(
        entries,
        num_samples=audio_cfg["num_samples"],
        sample_rate=audio_cfg["sample_rate"],
        train=train,
        root=data_cfg["root"],
    )
    num_workers = data_cfg.get("num_workers", 0)
    generator = torch.Generator().manual_seed(cfg.get("seed", 0)) if train else None
    return DataLoader(
        dataset,
        batch_size=cfg.get("train", {}).get("batch_size", 32),
        shuffle=train,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
