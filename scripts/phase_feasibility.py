"""Phase feasibility study, GO / NO-GO (PR-10). Owner: Member B (step B3).

Question: with FIXED fast Griffin-Lim (no learning at all), does the phase residual separate real
from vocoder-made speech? If not, the learnable parts (PR-14 onwards) should not be built as planned.

For N utterances of the dev split (each with its real clip + 6 fakes, same speaker and words), for each
STFT resolution, each Griffin-Lim initialisation and each iteration count K:
    clip score = mean over bins and frames of  w * (1 - cos(angle(X * conj(Y_K))))   (signal/residual.py)
then the AUC of real vs each vocoder (and all fakes pooled).

How to read AUC: 0.5 = no separation. Above 0.5 = fakes have the LARGER residual; below 0.5 = smaller.
Either direction is usable by a model, so separability = max(AUC, 1 - AUC).

Decision rule, fixed BEFORE looking at results (written into the output too):
    GO       separability >= 0.80 for at least 4 of 6 vocoders in some (resolution, init, K) setting
    WEAK     separability >= 0.65 for at least 2 vocoders in some setting
    NO-GO    otherwise

Example (from the repo root; ~10-20 min on a laptop CPU for 100 utterances):
    python scripts/phase_feasibility.py --n-utts 100
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from phasecon.data.librisevoc import LibriSeVocDataset
from phasecon.data.splits import read_split_file
from phasecon.evaluation.metrics import compute_auc
from phasecon.signal.residual import INITS, clip_score, phase_residual_map
from phasecon.signal.stft import MultiResSTFT
from phasecon.utils.config import load_many

GO, WEAK = (0.80, 4), (0.65, 2)


def verdict(table):
    """table[(r, init, K)][vocoder] = separability -> 'GO' | 'WEAK' | 'NO-GO' and the best setting."""
    best_setting, best_count = None, -1
    level = "NO-GO"
    for setting, per_voc in table.items():
        n_go = sum(v >= GO[0] for v in per_voc.values())
        n_weak = sum(v >= WEAK[0] for v in per_voc.values())
        if n_go >= GO[1]:
            level = "GO"
        elif n_weak >= WEAK[1] and level != "GO":
            level = "WEAK"
        if n_weak > best_count:
            best_setting, best_count = setting, n_weak
    return level, best_setting


def main():
    ap = argparse.ArgumentParser(description="Phase feasibility study (GO / NO-GO) with fixed Griffin-Lim.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml",
                                                    "configs/model/phasecon.yaml"])
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n-utts", type=int, default=100, help="utterances (each gives 1 real + 6 fake clips)")
    ap.add_argument("--iters", type=int, nargs="+", default=[5, 32], help="Griffin-Lim iteration counts K")
    ap.add_argument("--out", default="experiments/results/phase_feasibility")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data, audio = cfg["data"], cfg["audio"]
    pb = cfg["model"]["phase_branch"]
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
    settings = [(r, init, k) for r in range(len(stft.configs)) for init in INITS for k in args.iters]
    if args.n_utts < 100:
        print(f"WARNING: --n-utts {args.n_utts} < 100. With few clips, AUCs are noisy and the verdict can be "
              "wrong by chance (many settings x vocoders are compared). Use >= 100 for the real verdict.", flush=True)
    if audio["sample_rate"] != 24000:
        print(f"WARNING: sample_rate={audio['sample_rate']}; LibriSeVoc is 24 kHz. Apply the 24 kHz switch first.",
              flush=True)
    print(f"{len(entries)} clips ({len(keep)} utterances x 7), sample_rate={audio['sample_rate']}, "
          f"{len(settings)} settings (resolution x init x K)", flush=True)

    scores = {s: [] for s in settings}
    start = time.time()
    with torch.inference_mode():
        for i in range(len(ds)):
            wav = ds[i]["waveform"].unsqueeze(0)
            for (r, init, k) in settings:
                scores[(r, init, k)].append(clip_score(phase_residual_map(wav, stft, r, k, init, tau)).item())
            if (i + 1) % 70 == 0:
                print(f"  {i + 1}/{len(ds)} clips, {(time.time() - start) / 60:.1f} min", flush=True)

    labels = np.array([e.label for e in entries])
    vids = np.array([e.vocoder_id for e in entries])
    real = vids == 0
    table, rows = {}, []
    for s in settings:
        sc = np.array(scores[s])
        per_voc = {}
        for vid, name in enumerate(vocoders, start=1):
            sel = real | (vids == vid)
            auc = compute_auc(sc[sel], labels[sel])
            per_voc[name] = max(auc, 1 - auc)
            rows.append({"resolution": stft.configs[s[0]].n_fft, "init": s[1], "K": s[2], "vocoder": name,
                         "auc": auc, "separability": max(auc, 1 - auc),
                         "mean_real": float(sc[real].mean()), "mean_fake": float(sc[vids == vid].mean())})
        pooled = compute_auc(sc, labels)
        rows.append({"resolution": stft.configs[s[0]].n_fft, "init": s[1], "K": s[2], "vocoder": "ALL",
                     "auc": pooled, "separability": max(pooled, 1 - pooled),
                     "mean_real": float(sc[real].mean()), "mean_fake": float(sc[~real].mean())})
        table[s] = per_voc
    level, best = verdict(table)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"verdict": level, "best_setting": {"resolution": stft.configs[best[0]].n_fft, "init": best[1],
                                                      "K": best[2]},
                   "rule": {"GO": f"separability >= {GO[0]} for >= {GO[1]} vocoders",
                            "WEAK": f"separability >= {WEAK[0]} for >= {WEAK[1]} vocoders"},
                   "n_utts": len(keep), "split": args.split, "sample_rate": audio["sample_rate"],
                   "rows": rows}, f, indent=2)

    print(f"\nDONE in {(time.time() - start) / 60:.1f} min -> {args.out}/results.json")
    print("separability = max(AUC, 1-AUC); 0.5 = none, 1.0 = perfect. Columns: vocoders, then ALL (pooled AUC).")
    header = "res   init               K  " + "  ".join(f"{v[:8]:>8s}" for v in vocoders) + "      ALL"
    print(header)
    for s in settings:
        pooled = next(r["auc"] for r in rows if r["vocoder"] == "ALL" and r["resolution"] == stft.configs[s[0]].n_fft
                      and r["init"] == s[1] and r["K"] == s[2])
        print(f"{stft.configs[s[0]].n_fft:<5d} {s[1]:<18s} {s[2]:<3d} "
              + "  ".join(f"{table[s][v]:8.3f}" for v in vocoders) + f"   {pooled:6.3f}")
    print(f"\nVERDICT: {level}  (best setting: n_fft={stft.configs[best[0]].n_fft}, init={best[1]}, K={best[2]})")
    print(f"Rule: GO = >= {GO[1]} vocoders at >= {GO[0]}; WEAK = >= {WEAK[1]} vocoders at >= {WEAK[0]}; else NO-GO")


if __name__ == "__main__":
    main()
