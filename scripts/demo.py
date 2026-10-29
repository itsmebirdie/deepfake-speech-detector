"""Gradio demo for PhaseCon-Net (inference only, no training needed).

Put this file in scripts/demo.py and run from the repo root:

  pip install gradio
  python scripts/demo.py
  python scripts/demo.py --checkpoint experiments/checkpoints/phasecon_seed1234_train/best.pt
  python scripts/demo.py --baseline-checkpoint <path to your fine-tuned baseline best.pt>   # optional side-by-side

Preprocessing mirrors LibriSeVocDataset: mono -> resample to 24 kHz -> 96,000 samples (4 s).
Clips longer than 4 s are split into 4 s windows and the scores are averaged.
Score = logits_bin[:, 1] - logits_bin[:, 0]  (higher = more synthetic), same as run_inference.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

import gradio as gr

from phasecon.data.transforms import pad_or_crop
from phasecon.models import build_model
from phasecon.training.checkpoint import load_checkpoint
from phasecon.utils.config import load_many

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIGS = ["configs/default.yaml", "configs/data/librisevoc.yaml"]
OVERRIDES = ["audio.sample_rate=24000", "audio.num_samples=96000", "model.baseline.weights=null"]
MAX_WINDOWS = 8  # cap compute on long uploads (~32 s)


def rel(p):
    return p if Path(p).exists() else str(REPO / p)


def load_model(model_cfg_path, checkpoint, device):
    cfg = load_many([rel(p) for p in BASE_CONFIGS + [model_cfg_path, "configs/eval/in_distribution.yaml"]], OVERRIDES)
    model = build_model(cfg).to(device)
    try:
        load_checkpoint(rel(checkpoint), model, map_location=device)
    except RuntimeError:
        # Checkpoint has extra weights for the last ULCP step's correction layer (code changed after training).
        # Tolerate ONLY that known difference; anything else is a real mismatch and still fails.
        ckpt = torch.load(rel(checkpoint), map_location=device, weights_only=False)
        res = model.load_state_dict(ckpt["model"], strict=False)
        extra = [k for k in res.unexpected_keys if ".steps.4.correction." not in k]
        if res.missing_keys or extra:
            raise RuntimeError(f"checkpoint does not match model: missing={res.missing_keys} unexpected={extra}")
        print(f"note: ignored {len(res.unexpected_keys)} unused last-step ULCP correction weights", flush=True)
    model.eval()
    return model, cfg


def load_audio(path, sample_rate, num_samples):
    """Return a [W, num_samples] float tensor of windows."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wav = audio.mean(axis=1)
    if rate != sample_rate:
        g = math.gcd(rate, sample_rate)
        wav = resample_poly(wav, sample_rate // g, rate // g).astype(np.float32)
    wav = torch.from_numpy(np.ascontiguousarray(wav))
    if len(wav) <= num_samples:
        windows = [pad_or_crop(wav, num_samples, random_crop=False)]
    else:
        windows = [wav[s:s + num_samples] for s in range(0, len(wav) - num_samples + 1, num_samples)]
        if len(wav) % num_samples >= num_samples // 4:  # keep a long enough tail
            windows.append(wav[-num_samples:])
    return torch.stack(windows[:MAX_WINDOWS]), len(wav) / sample_rate


@torch.inference_mode()
def score_windows(model, windows, device, vocoder_names):
    out = model(windows.to(device))
    logits = out["logits_bin"].float()
    scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    voc = None
    for k, v in out.items():  # optional vocoder-ID head; key name is not part of the contract, so look it up
        if "voc" in k and torch.is_tensor(v) and v.dim() == 2:
            names = None
            if v.shape[1] == len(vocoder_names) + 1:
                names = ["bona fide"] + list(vocoder_names)
            elif v.shape[1] == len(vocoder_names):
                names = list(vocoder_names)
            if names:
                p = torch.softmax(v.float(), dim=1).mean(0).cpu().numpy()
                voc = sorted(zip(names, p), key=lambda t: -t[1])[:3]
            break
    return scores, voc


def results_table(results_dir):
    rows = []
    for mf in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        row = {"Run": mf.parent.name, "EER %": round(100 * m["eer"], 2), "AUC": round(m["auc"], 5),
               "Acc %": round(100 * m["accuracy"], 2)}
        for name, v in m.get("per_vocoder", {}).items():
            row[f"{name} EER %"] = round(100 * v["eer"], 2)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["no results found in " + str(results_dir)]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/checkpoints/phasecon_seed1234_train/best.pt")
    ap.add_argument("--baseline-checkpoint", default=None, help="optional fine-tuned baseline for side-by-side")
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--share", action="store_true", help="public gradio link (handy for showing the demo remotely)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phasecon, cfg = load_model("configs/model/phasecon.yaml", args.checkpoint, device)
    baseline = None
    if args.baseline_checkpoint:
        baseline, _ = load_model("configs/model/baseline_only.yaml", args.baseline_checkpoint, device)
    sr, n = cfg["audio"]["sample_rate"], cfg["audio"]["num_samples"]
    vocoders = cfg["data"]["vocoders"]
    print(f"models loaded on {device}", flush=True)

    def analyse(path, threshold):
        if path is None:
            return None, "Upload or record an audio clip first.", None
        try:
            windows, seconds = load_audio(path, sr, n)
        except Exception as e:  # unreadable / unsupported file
            return None, f"Could not read this audio file: {e}", None
        s, voc = score_windows(phasecon, windows, device, vocoders)
        score = float(s.mean())
        p_fake = 1 / (1 + math.exp(-score))
        verdict = "SYNTHETIC (deepfake)" if score > threshold else "BONA FIDE (real)"
        lines = [f"### Verdict: {verdict}",
                 f"PhaseCon-Net score **{score:+.2f}** (threshold {threshold:+.1f}), "
                 f"P(synthetic) = {p_fake:.3f}",
                 f"Clip length {seconds:.1f} s, {len(windows)} window(s) of {n / sr:.0f} s averaged. "
                 f"Per-window scores: {', '.join(f'{x:+.2f}' for x in s)}"]
        if voc:
            lines.append("Vocoder head top guesses: " + ", ".join(f"{k} ({v:.2f})" for k, v in voc))
        compare = None
        if baseline is not None:
            bs, _ = score_windows(baseline, windows, device, vocoders)
            b = float(bs.mean())
            compare = pd.DataFrame({"Model": ["PhaseCon-Net", "Fine-tuned baseline"],
                                    "Score": [round(score, 3), round(b, 3)],
                                    "Verdict": [verdict, "SYNTHETIC" if b > threshold else "BONA FIDE"]})
        return {"synthetic": p_fake, "bona fide": 1 - p_fake}, "\n\n".join(lines), compare

    with gr.Blocks(title="PhaseCon-Net deepfake speech detector") as app:
        gr.Markdown("# PhaseCon-Net: deepfake speech detector\n"
                    "Detects speech made by neural vocoders (trained and tested on LibriSeVoc).")
        with gr.Tab("Try it"):
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio clip")
            thr = gr.Slider(-5, 5, value=0, step=0.1, label="Decision threshold (0 = default)")
            btn = gr.Button("Analyse", variant="primary")
            label = gr.Label(label="Prediction")
            details = gr.Markdown()
            cmp_table = gr.Dataframe(label="Side-by-side with baseline", visible=baseline is not None)
            btn.click(analyse, [audio, thr], [label, details, cmp_table])
            gr.Markdown("*Caution: trained only on LibriSeVoc (English audiobook speech, 6 vocoders). "
                        "Phone-quality, noisy or other-language audio is out of distribution.*")
        with gr.Tab("Evaluation results"):
            gr.Markdown("Test-split results saved under `experiments/results/`.")
            gr.Dataframe(value=lambda: results_table(rel(args.results_dir)), label="Runs")
    app.launch(share=args.share)


if __name__ == "__main__":
    main()