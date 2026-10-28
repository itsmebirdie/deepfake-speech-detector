"""Plot phase-residual maps for real vs fake clips of the same utterance (PR-10 figures). Owner: B3.

One figure per utterance: the real clip and its 6 vocoder versions side by side, each showing
w * (1 - cos dphi) over time (x) and frequency (y) at one STFT resolution, with fixed Griffin-Lim.

Example (from the repo root):
    python scripts/visualize_phase.py --n-utts 3
"""
import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from phasecon.data.librisevoc import LibriSeVocDataset  # noqa: E402
from phasecon.data.splits import read_split_file  # noqa: E402
from phasecon.signal.residual import INITS, clip_score, phase_residual_map  # noqa: E402
from phasecon.signal.stft import MultiResSTFT  # noqa: E402
from phasecon.utils.config import load_many  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Plot phase-residual maps for real vs fake clips.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml",
                                                    "configs/model/phasecon.yaml"])
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n-utts", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=0, help="index into model.phase_branch.stft_resolutions")
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--init", default="zeros", choices=INITS)
    ap.add_argument("--out", default="experiments/results/phase_feasibility/figures")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data, audio, pb = cfg["data"], cfg["audio"], cfg["model"]["phase_branch"]
    stft = MultiResSTFT(pb["stft_resolutions"])
    tau = float(pb["discrepancy"]["tau"])
    entries = read_split_file(data["split_files"][args.split], data["root"], data["vocoders"])
    utts = sorted({e.utt_id for e in entries})
    random.Random(cfg["seed"]).shuffle(utts)
    os.makedirs(args.out, exist_ok=True)
    c = stft.configs[args.resolution]

    for utt in utts[: args.n_utts]:
        group = sorted((e for e in entries if e.utt_id == utt), key=lambda e: e.vocoder_id)
        ds = LibriSeVocDataset(group, num_samples=audio["num_samples"], sample_rate=audio["sample_rate"],
                               root=data["root"])
        fig, axes = plt.subplots(1, len(group), figsize=(3 * len(group), 3.4), sharey=True)
        with torch.inference_mode():
            for ax, e, i in zip(axes, group, range(len(group))):
                m = phase_residual_map(ds[i]["waveform"].unsqueeze(0), stft, args.resolution, args.iters, args.init, tau)[0]
                ax.imshow(m.numpy(), origin="lower", aspect="auto", vmin=0, vmax=2, cmap="magma",
                          extent=[0, audio["num_samples"] / audio["sample_rate"], 0, audio["sample_rate"] / 2000])
                name = "real" if e.vocoder_id == 0 else data["vocoders"][e.vocoder_id - 1]
                ax.set_title(f"{name}\nmean {clip_score(m.unsqueeze(0)).item():.3f}", fontsize=9)
                ax.set_xlabel("time (s)")
        axes[0].set_ylabel("frequency (kHz)")
        fig.suptitle(f"{utt}: w*(1-cos dphi), n_fft={c.n_fft}, init={args.init}, K={args.iters}", fontsize=10)
        fig.tight_layout()
        path = os.path.join(args.out, f"{utt}_nfft{c.n_fft}_{args.init}_K{args.iters}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
