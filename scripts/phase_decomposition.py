"""Split the PR-10 clip score into its two ingredients (PR-10 follow-up check). Owner: Member B.

The feasibility score mean(w * (1 - cos dphi)) mixes two things:
  * w = A / (A + tau): how many time-frequency bins carry energy (a MAGNITUDE pattern), and
  * 1 - cos dphi: how far the real phase is from the Griffin-Lim phase (the TIMING signal).
This script reports, per vocoder, the AUC (real vs that vocoder) of
  rms          overall loudness of the clip
  mean_weight  mean of w                          -> magnitude pattern only
  timing_only  sum(w * (1 - cos dphi)) / sum(w)   -> phase difference, normalised by the weights
0.5 = no difference; far from 0.5 in either direction = separable.

Example (from the repo root, ~2 min):
    python scripts/phase_decomposition.py --n-utts 100
"""
import argparse
import random

import numpy as np
import torch

from phasecon.data.librisevoc import LibriSeVocDataset
from phasecon.data.splits import read_split_file
from phasecon.evaluation.metrics import compute_auc
from phasecon.signal.griffin_lim import fast_griffin_lim
from phasecon.signal.residual import initial_phase
from phasecon.signal.stft import MultiResSTFT
from phasecon.utils.config import load_many


def main():
    ap = argparse.ArgumentParser(description="Split the PR-10 phase-residual score into magnitude and timing parts.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml",
                                                    "configs/model/phasecon.yaml"])
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n-utts", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--init", default="zeros")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data, audio, pb = cfg["data"], cfg["audio"], cfg["model"]["phase_branch"]
    stft = MultiResSTFT(pb["stft_resolutions"])
    tau = float(pb["discrepancy"]["tau"])
    vocoders = data["vocoders"]
    entries = read_split_file(data["split_files"][args.split], data["root"], vocoders)
    utts = sorted({e.utt_id for e in entries})
    random.Random(cfg["seed"]).shuffle(utts)
    keep = set(utts[: args.n_utts])
    entries = [e for e in entries if e.utt_id in keep]
    ds = LibriSeVocDataset(entries, num_samples=audio["num_samples"], sample_rate=audio["sample_rate"],
                           root=data["root"])
    r, c = args.resolution, stft.configs[args.resolution]
    rms, mean_w, timing = [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            wav = ds[i]["waveform"].unsqueeze(0)
            X = stft.stft(wav, r)
            A = X.abs()
            Y = fast_griffin_lim(A, stft, r, wav.shape[-1], n_iter=args.iters,
                                 init_phase=initial_phase(A.shape, c.n_fft, args.init))
            d = 1 - torch.cos(torch.angle(X * torch.conj(Y)))
            w = A / (A + tau)
            rms.append(wav.pow(2).mean().sqrt().item())
            mean_w.append(w.mean().item())
            timing.append(((w * d).sum() / w.sum()).item())
    labels = np.array([e.label for e in entries])
    vids = np.array([e.vocoder_id for e in entries])
    real = vids == 0
    print(f"{len(entries)} clips, n_fft={c.n_fft}, init={args.init}, K={args.iters}, sample_rate={audio['sample_rate']}")
    print(f"{'vocoder':18s} {'rms':>8s} {'mean_weight':>12s} {'timing_only':>12s}   (AUC; 0.5 = no difference)")
    for vid, name in enumerate(vocoders, start=1):
        sel = real | (vids == vid)
        aucs = [compute_auc(np.array(v)[sel], labels[sel]) for v in (rms, mean_w, timing)]
        print(f"{name:18s} {aucs[0]:8.3f} {aucs[1]:12.3f} {aucs[2]:12.3f}")


if __name__ == "__main__":
    main()
