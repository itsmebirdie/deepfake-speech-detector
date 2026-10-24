"""Evaluate the released pretrained baseline on an official split (PR-06). Owner: Member A (step A4).

What it does:
  1. Builds the authors' RawNet model from their own code (`model.py` + `model_config_RawNet.yaml`
     in the baseline repo) and loads the released weights. The baseline repo is NOT modified.
  2. Feeds it clips exactly as the authors do: 24 kHz, mono, first 4 s (96,000 samples), short clips
     repeated (PR-04's `pad_or_crop`).
  3. Saves one row per clip to scores.csv and the summary to metrics.json.

Score convention (docs/INTERFACES.md section 5): higher score = more likely SYNTHETIC.
The baseline's binary head is log-softmax over [synthetic, bona fide] (its training target is
`y == 0`, so index 1 = bona fide), therefore
    score = logp[:, 0] - logp[:, 1]
The baseline repo has no EER code; EER here comes from the ROC curve (scikit-learn), and PR-07
replaces it with the shared `evaluation/metrics.py`.

Two input modes:
  first4s  (default) the first 4 s of every clip, as in the authors' training/dev loop (main.py)
  eval_py  the authors' eval.py: the clip cut into whole 4 s pieces, softmax probabilities averaged

Example (from the repo root):
  python scripts/eval_baseline.py --baseline-dir ../baseline-repo --split test
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from phasecon.data.librisevoc import LibriSeVocDataset
from phasecon.data.splits import read_split_file
from phasecon.data.transforms import pad_or_crop
from phasecon.utils.config import load_many

BASELINE_RATE = 24000
BASELINE_LEN = 4 * BASELINE_RATE  # 96,000 samples
REAL, FAKE = 0, 1  # repo convention: 0 bona fide, 1 synthetic


def load_baseline(baseline_dir: str, weights: str, device: str):
    """Build the authors' RawNet from their own code and load the released weights."""
    sys.path.insert(0, os.path.abspath(baseline_dir))
    from model import RawNet  # noqa: E402  (the authors' model.py)

    with open(os.path.join(baseline_dir, "model_config_RawNet.yaml")) as f:
        model_cfg = yaml.safe_load(f)["model"]
    model = RawNet(model_cfg, device).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """EER from the ROC curve: the point where false acceptance = false rejection."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=FAKE)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def summarize(scores, labels, pred_voc, vocoder_ids, vocoders):
    eer, thr = compute_eer(scores, labels)
    pred = (scores > 0).astype(int)  # argmax of the binary head
    out = {
        "n_clips": int(len(scores)),
        "eer": eer,
        "eer_percent": round(100 * eer, 4),
        "eer_threshold": thr,
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy_argmax": float((pred == labels).mean()),
        "accuracy_at_eer_threshold": float(((scores >= thr).astype(int) == labels).mean()),
        "vocoder_id_accuracy": float((pred_voc == vocoder_ids).mean()),
        "per_vocoder": {},
    }
    real = vocoder_ids == 0
    for vid, name in enumerate(vocoders, start=1):
        sel = real | (vocoder_ids == vid)
        v_eer, _ = compute_eer(scores[sel], labels[sel])
        out["per_vocoder"][name] = {
            "eer_percent": round(100 * v_eer, 4),
            "detected_as_fake": float((pred[vocoder_ids == vid] == FAKE).mean()),
            "vocoder_id_accuracy": float((pred_voc[vocoder_ids == vid] == vid).mean()),
        }
    out["real_detected_as_real"] = float((pred[real] == REAL).mean())
    return out


@torch.inference_mode()
def run_first4s(model, entries, root, device, batch_size, num_workers):
    """The authors' training/dev input: first 4 s of every clip, batched."""
    ds = LibriSeVocDataset(entries, num_samples=BASELINE_LEN, sample_rate=BASELINE_RATE, root=root)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    logp_bin, logp_voc = [], []
    start, done = time.time(), 0
    for batch in loader:
        out_bin, out_voc = model(batch["waveform"].to(device))
        logp_bin.append(out_bin.float().cpu())
        logp_voc.append(out_voc.float().cpu())
        done += len(batch["utt_id"])
        if len(logp_bin) % 20 == 0:
            print(f"  {done}/{len(ds)} clips, {done / (time.time() - start):.1f} clips/s", flush=True)
    return torch.cat(logp_bin), torch.cat(logp_voc)


@torch.inference_mode()
def run_eval_py(model, entries, root, device):
    """The authors' eval.py: whole 4 s pieces, softmax probabilities averaged per clip."""
    ds = LibriSeVocDataset(entries, num_samples=BASELINE_LEN, sample_rate=BASELINE_RATE, root=root,
                           random_crop=True)  # random_crop=True makes the dataset read the whole file
    logp_bin, logp_voc = [], []
    start = time.time()
    for i, entry in enumerate(entries):
        wav = ds.load_waveform(os.path.join(root, entry.path))
        if len(wav) <= BASELINE_LEN:
            pieces = [pad_or_crop(wav, BASELINE_LEN)]
        else:  # eval.py keeps only whole 4 s pieces
            pieces = [wav[k * BASELINE_LEN:(k + 1) * BASELINE_LEN] for k in range(len(wav) // BASELINE_LEN)]
        out_bin, out_voc = model(torch.stack(pieces).to(device))
        p_bin = F.softmax(out_bin, dim=-1).mean(0)
        p_voc = F.softmax(out_voc, dim=-1).mean(0)
        logp_bin.append(torch.log(p_bin).float().cpu())
        logp_voc.append(torch.log(p_voc).float().cpu())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(entries)} clips, {(i + 1) / (time.time() - start):.1f} clips/s", flush=True)
    return torch.stack(logp_bin), torch.stack(logp_voc)


def git_commit(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Evaluate the released pretrained baseline on an official split.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml",
                                                    "configs/model/baseline_only.yaml"],
                    help="YAML configs, merged left to right")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. data.root=...")
    ap.add_argument("--baseline-dir", default=None,
                    help="folder of the authors' repo (default: model.baseline.repo_dir in the config)")
    ap.add_argument("--weights", default=None, help="released weights (default: model.baseline.weights)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--mode", default="first4s", choices=["first4s", "eval_py"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances (quick check)")
    ap.add_argument("--out", default=None, help="output folder (default: experiments/results/baseline_released_<split>_<mode>)")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data_cfg, base_cfg = cfg["data"], cfg["model"]["baseline"]
    baseline_dir = args.baseline_dir or base_cfg["repo_dir"]
    weights = args.weights or base_cfg["weights"]
    out_dir = args.out or os.path.join("experiments", "results", f"baseline_released_{args.split}_{args.mode}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    entries = read_split_file(data_cfg["split_files"][args.split], data_cfg["root"], data_cfg["vocoders"])
    if args.limit:
        entries = entries[: args.limit * (1 + len(data_cfg["vocoders"]))]
    print(f"split={args.split} mode={args.mode} clips={len(entries)} device={device} "
          f"baseline={baseline_dir} weights={weights}", flush=True)

    model = load_baseline(baseline_dir, weights, device)
    start = time.time()
    if args.mode == "first4s":
        logp_bin, logp_voc = run_first4s(model, entries, data_cfg["root"], device, args.batch_size, args.num_workers)
    else:
        logp_bin, logp_voc = run_eval_py(model, entries, data_cfg["root"], device)
    minutes = (time.time() - start) / 60

    scores = (logp_bin[:, 0] - logp_bin[:, 1]).numpy()  # baseline index 0 = synthetic, 1 = bona fide
    labels = np.array([e.label for e in entries])
    vocoder_ids = np.array([e.vocoder_id for e in entries])
    pred_voc = logp_voc.argmax(1).numpy()

    metrics = summarize(scores, labels, pred_voc, vocoder_ids, data_cfg["vocoders"])
    metrics.update({
        "split": args.split,
        "mode": args.mode,
        "paper_eer_percent": 0.13,  # Sun et al., CVPRW 2023, Table 3, "Ours" on LibriSeVoc
        "baseline_commit": git_commit(baseline_dir),
        "weights": weights,
        "device": device,
        "minutes": round(minutes, 1),
    })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "path", "label", "vocoder_id", "score", "logp_synthetic", "logp_bona_fide",
                    "pred_vocoder_id"])
        for e, s, lb, pv in zip(entries, scores, logp_bin.numpy(), pred_voc):
            w.writerow([e.utt_id, e.path, e.label, e.vocoder_id, f"{s:.6f}", f"{lb[0]:.6f}", f"{lb[1]:.6f}", int(pv)])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"DONE in {minutes:.1f} min -> {out_dir}")
    print(f"EER {metrics['eer_percent']:.3f}% (paper: 0.13%)  AUC {metrics['auc']:.5f}  "
          f"accuracy {100 * metrics['accuracy_argmax']:.2f}%  vocoder-ID accuracy "
          f"{100 * metrics['vocoder_id_accuracy']:.2f}%")
    for name, v in metrics["per_vocoder"].items():
        print(f"  {name:18s} EER {v['eer_percent']:.3f}%  caught {100 * v['detected_as_fake']:.2f}%  "
              f"vocoder named right {100 * v['vocoder_id_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
