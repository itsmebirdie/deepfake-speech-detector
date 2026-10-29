"""Evidence for gates G3 and G4 (PR-23) on ONE REAL BATCH.  Owner: Member C.

Run from anywhere (needs the dataset, the baseline repo and the released weights, i.e. your laptop):
  python scripts/gate_check.py                              # defaults: batch 8, 100 overfit steps, lr 3e-4
  python scripts/gate_check.py --batch-size 4 --steps 150
  python scripts/gate_check.py --override model.phase_branch.ulcp.num_steps=3     # lower K if memory is too high
  python scripts/gate_check.py --synthetic --override model.baseline.weights=null  # plumbing smoke test, no data

What it checks (the PASS/FAIL lines map onto the gate checklists in docs/ISSUES.md):
  G3  interface   PhaseBranch output keys/shapes match docs/INTERFACES.md section 3 on the batch, and the
                  fusion block accepts H_d next to the baseline's H_b (finite, same shape, gradients reach both).
  G4  backward    full model forward + backward on the batch: finite loss, finite gradients, and which
                  trainable parameters received none (expected: the last unrolled ULCP step, see models/ulcp.py).
  G4  amp         (CUDA only) a short run with train.amp=true: no non-finite loss, phase stream fp32 inside autocast.
  G4  overfit     the trainer's overfit_batch on the full model: the classification loss (binary + beta * vocoder)
                  goes to ~0 without a NaN. The consistency term is reported separately: it has a floor on one batch.
  G4  dummy       the model really uses the real PhaseBranch.
  PR  profile     step time and peak memory at the configured K and R (PR-23 asks for these numbers in the PR).
Writes experiments/gates/pr23_gate_check.{json,md}; paste the .md into the PR and the gate issues.
Exit code 1 if any check FAILs (SKIPPED does not fail).
"""
import argparse
import copy
import json
import math
import os
import platform
try:
    import resource  # Unix only
except ImportError:  # Windows
    resource = None
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIGS = ["configs/default.yaml", "configs/data/librisevoc.yaml", "configs/train/default.yaml"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _finite(x):
    return bool(torch.isfinite(x).all())


def get_batch(cfg, synthetic, batch_size):
    """One batch, fetched once and reused by every check (so they all see the same audio)."""
    from phasecon.data.librisevoc import build_dataloader

    if synthetic:
        g = torch.Generator().manual_seed(0)
        n = int(cfg["audio"]["num_samples"])
        label = torch.arange(batch_size) % 2
        vid = label * (1 + (torch.arange(batch_size) // 2) % 6)
        return {"waveform": torch.randn(batch_size, n, generator=g) * 0.1, "label": label.long(),
                "vocoder_id": vid.long(), "utt_id": [f"synthetic_{i}" for i in range(batch_size)]}
    return next(iter(build_dataloader(cfg, "train", train=True)))


def fresh(cfg, batch, run_suffix, overrides=None):
    """A newly built model and its Trainer on [batch] (a one-batch loader), in overfit mode."""
    from phasecon.models import build_model
    from phasecon.training.trainer import Trainer

    c = copy.deepcopy(cfg)
    c["run_name"] = f"gate_pr23_{run_suffix}"
    for k, v in (overrides or {}).items():
        c["train"][k] = v
    model = build_model(c)
    return c, model, Trainer(c, model, [batch], None)


def check_interface(cfg, batch, device):
    from phasecon.models import build_model

    model = build_model(cfg).to(device).eval()
    wav = batch["waveform"].to(device)
    enc = cfg["model"]["phase_branch"]["encoder"]
    t_d, c_d = int(enc["t_out"]), int(enc.get("out_dim", 128))
    res = {"real_branch": not model.phase_branch_is_dummy}
    with torch.no_grad():
        phase = model.phase_branch(wav)
        h_b = model.baseline.forward_features(wav)
    res["keys_ok"] = {"H_d", "cons_per_sample"} <= set(phase)
    res["H_d_shape"], res["H_b_shape"] = list(phase["H_d"].shape), list(h_b.shape)
    res["cons_shape"] = list(phase["cons_per_sample"].shape)
    res["shapes_ok"] = (tuple(phase["H_d"].shape) == (wav.shape[0], t_d, c_d)
                        and tuple(phase["cons_per_sample"].shape) == (wav.shape[0],))
    res["finite"] = _finite(phase["H_d"]) and _finite(phase["cons_per_sample"])
    res["cons_range"] = [float(phase["cons_per_sample"].min()), float(phase["cons_per_sample"].max())]
    hb = h_b.detach().clone().requires_grad_(True)
    hd = phase["H_d"].detach().clone().requires_grad_(True)
    fused = model.fusion(hb, hd)
    fused.square().sum().backward()
    res["fusion_shape_equals_H_b"] = tuple(fused.shape) == tuple(h_b.shape)
    res["fusion_finite"] = _finite(fused)
    res["grads_reach_both_streams"] = (hb.grad is not None and hd.grad is not None
                                       and _finite(hb.grad) and _finite(hd.grad) and bool(hd.grad.abs().sum() > 0))
    res["T_b_differs_from_T_d"] = res["H_b_shape"][1] != res["H_d_shape"][1]
    res["status"] = "PASS" if (res["real_branch"] and res["keys_ok"] and res["shapes_ok"] and res["finite"]
                               and res["fusion_shape_equals_H_b"] and res["fusion_finite"]
                               and res["grads_reach_both_streams"]) else "FAIL"
    return res


def check_backward(cfg, batch, device):
    from phasecon.losses.combined import total_loss
    from phasecon.models import build_model

    model = build_model(cfg).to(device).train()
    beta, mu = float(cfg["train"]["loss"]["beta"]), float(cfg["train"]["loss"]["mu"])
    wav = batch["waveform"].to(device)
    target = {k: batch[k].to(device) for k in ("label", "vocoder_id")}
    loss, parts = total_loss(model(wav), target, beta, mu)
    loss.backward()
    def dead_params(m):  # trainable parameters that got no gradient (the baseline's own heads are never used)
        skip = {id(p) for p in m.baseline.head_parameters()}
        return {n for n, p in m.named_parameters() if p.requires_grad and id(p) not in skip and p.grad is None}

    dead = dead_params(model)
    # Reference: the same baseline with the phase branch off. Whatever is dead there is the authors' RawNet2 (the
    # `bn1` of each stage's first residual block is bypassed), not something the integration cut off.
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["model"]["name"] = "baseline_only"
    ref_cfg["model"]["phase_branch"] = {"enabled": False}
    ref = build_model(ref_cfg).to(device).train()
    total_loss(ref(wav), target, beta, 0.0)[0].backward()
    dead_in_baseline = dead_params(ref)
    k = len(model.phase_branch.ulcp[0].steps) if model.phase_branch.ulcp is not None else 0
    last_step = tuple(f"ulcp.{r}.steps.{k - 1}." for r in range(len(model.phase_branch.ulcp or [])))
    unexpected = sorted(n for n in dead - dead_in_baseline if not any(s in n for s in last_step))
    finite_grads = all(_finite(p.grad) for p in model.parameters() if p.grad is not None)
    return {"loss": float(loss), "parts": {a: float(b) for a, b in parts.items()},
            "loss_finite": math.isfinite(float(loss)), "grads_finite": finite_grads,
            "params_without_grad": len(dead), "dead_in_baseline_too": len(dead & dead_in_baseline),
            "dead_last_ulcp_step": len(dead) - len(dead & dead_in_baseline) - len(unexpected),
            "unexpected_without_grad": unexpected[:10],
            "note": "expected without grad: RawNet2's bypassed bn1 layers (also dead in the plain baseline) and the K-th "
                    "ULCP step's alpha and g_theta (Y_K = T_K is taken before they act). Anything else FAILs.",
            "status": "PASS" if (math.isfinite(float(loss)) and finite_grads and not unexpected) else "FAIL"}


def check_amp(cfg, batch, device, steps):
    if device.type != "cuda":
        return {"status": "SKIPPED", "reason": "needs CUDA (the trainer enables AMP on CUDA only); run this on the GPU"}
    c, model, tr = fresh(cfg, batch, "amp", {"amp": True, "amp_dtype": "float16"})
    seen = {}

    def probe(_, inputs, output):
        seen["in"] = str(inputs[0].dtype)
        seen["autocast"] = torch.is_autocast_enabled("cuda")
        seen["out"] = sorted({str(v.dtype) for v in output.values()})

    model.phase_branch.register_forward_hook(probe)
    res = tr.overfit_batch(num_steps=steps)
    ok = (tr.nonfinite_skipped == 0 and all(math.isfinite(x) for x in res["losses"])
          and seen.get("in") == "torch.float32" and seen.get("autocast") is False)
    return {"steps": steps, "dtype": "float16 + GradScaler", "nonfinite_skipped": tr.nonfinite_skipped,
            "initial_loss": res["initial_loss"], "final_loss": res["final_loss"],
            "phase_branch_saw": seen, "status": "PASS" if ok else "FAIL"}


def check_overfit(cfg, batch, device, steps):
    from phasecon.losses.combined import total_loss

    c, model, tr = fresh(cfg, batch, "overfit")
    beta, mu = tr.beta, tr.mu
    res = tr.overfit_batch(num_steps=steps)
    wav = batch["waveform"].to(tr.device)
    target = {k: batch[k].to(tr.device) for k in ("label", "vocoder_id")}
    out = {}
    for mode in ("train", "eval"):  # train = the mode the trainer optimised in; eval = BatchNorm running statistics
        getattr(model, mode)()
        with torch.no_grad():
            _, p = total_loss(model(wav), target, beta, mu)
        out[mode] = {a: float(b) for a, b in p.items()}
    cls = out["train"]["bin"] + beta * out["train"]["voc"]
    ok = tr.nonfinite_skipped == 0 and all(math.isfinite(x) for x in res["losses"]) and cls < 0.05
    return {"steps": steps, "lr": cfg["train"].get("overfit_lr"), "amp": tr.use_amp, "beta": beta, "mu": mu,
            "initial_loss": res["initial_loss"], "final_loss": res["final_loss"], "min_loss": res["min_loss"],
            "accuracy": res["accuracy"], "nonfinite_skipped": tr.nonfinite_skipped,
            "classification_loss_train_mode": cls, "parts_train_mode": out["train"], "parts_eval_mode": out["eval"],
            "note": "pass = classification loss < 0.05 in train mode; L_cons keeps a floor on one batch (total ~ mu*cons). "
                    "Eval-mode losses can be large on a tiny batch (BatchNorm running stats): informational only.",
            "status": "PASS" if ok else "FAIL"}


def profile(cfg, batch, device, warmup=2, timed=5):
    c, model, tr = fresh(cfg, batch, "profile")
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + timed):
        _sync(tr.device)
        t = time.perf_counter()
        tr._train_step(batch, step_scheduler=False)
        _sync(tr.device)
        if i >= warmup:
            times.append(time.perf_counter() - t)
    pb = cfg["model"]["phase_branch"]
    out = {"batch_size": int(batch["waveform"].shape[0]), "num_samples": int(batch["waveform"].shape[1]),
           "K": int(pb["ulcp"]["num_steps"]), "R": len(pb["stft_resolutions"]), "amp": tr.use_amp,
           "step_time_s_median": statistics.median(times), "step_time_s_min": min(times), "timed_steps": timed,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else f"cpu ({platform.processor() or platform.machine()})",
           "torch": torch.__version__}
    if device.type == "cuda":
        out["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    out["process_peak_rss_gb"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
                                  if resource is not None else None)  # Linux: KB; not available on Windows
    out["status"] = "INFO"
    return out


def markdown(r):
    def row(name, d, detail):
        return f"| {name} | {d['status']} | {detail} |"

    i, b, a, o, p = r["g3_interface"], r["g4_backward"], r["g4_amp"], r["g4_overfit"], r["profile"]
    rows = [
        row("G3 PhaseBranch matches INTERFACES.md s.3 on a real batch", i,
            f"H_d {i['H_d_shape']}, cons {i['cons_shape']}, H_b {i['H_b_shape']}; real branch: {i['real_branch']}"),
        row("G3 fusion accepts real H_d / H_b", i,
            f"shape ok: {i['fusion_shape_equals_H_b']}, finite: {i['fusion_finite']}, grads reach both: {i['grads_reach_both_streams']}"),
        row("G4 forward + backward on a real batch", b,
            f"loss {b['loss']:.4f}, grads finite: {b['grads_finite']}, params without grad: {b['params_without_grad']} "
            f"({b['dead_in_baseline_too']} dead in the plain baseline, {b['dead_last_ulcp_step']} last ULCP step, "
            f"{len(b['unexpected_without_grad'])} unexpected)"),
        row("G4 AMP run without NaN", a,
            a.get("reason") or f"{a['steps']} steps fp16, non-finite skipped: {a['nonfinite_skipped']}, phase stream saw {a['phase_branch_saw']}"),
        row("G4 overfit_batch drives loss to ~0", o,
            f"{o['steps']} steps @ lr {o['lr']}: loss {o['initial_loss']:.3f} -> {o['final_loss']:.4f}; classification part "
            f"{o['classification_loss_train_mode']:.4f}; L_cons {o['parts_train_mode'].get('cons', float('nan')):.3f}; "
            f"acc {o['accuracy']:.2f}; NaN skips {o['nonfinite_skipped']}"),
        row("G4 DummyPhaseBranch not in use", {"status": "PASS" if i["real_branch"] else "FAIL"}, f"real branch: {i['real_branch']}"),
    ]
    mem = f"GPU peak {p['gpu_peak_memory_gb']:.2f} GB, " if "gpu_peak_memory_gb" in p else ""
    rss = (f"process peak RSS {p['process_peak_rss_gb']:.2f} GB, "
           if p.get("process_peak_rss_gb") is not None else "")
    return "\n".join([
        f"### Gate check (PR-23), batch source: {r['batch_source']}, {p['device']}, torch {p['torch']}", "",
        "| Check | Result | Detail |", "|---|---|---|", *rows, "",
        f"**Cost at K={p['K']}, R={p['R']}** (batch {p['batch_size']} x {p['num_samples']} samples, AMP {p['amp']}): "
        f"{mem}{rss}"
        f"step time {p['step_time_s_median']:.2f} s (median of {p['timed_steps']}, min {p['step_time_s_min']:.2f} s).", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description="G3/G4 evidence for PR-23 on one batch.")
    ap.add_argument("--config", nargs="*", default=["configs/model/phasecon.yaml"])
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. model.phase_branch.ulcp.num_steps=3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100, help="overfit steps")
    ap.add_argument("--amp-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4, help="constant LR for the overfit run (3e-3 is unstable)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--synthetic", action="store_true", help="random audio instead of the dataset (plumbing smoke test only)")
    ap.add_argument("--out", default="experiments/gates")
    args = ap.parse_args(argv)

    os.chdir(REPO)  # the configs use repo-relative paths
    from phasecon.utils.config import load_many
    from phasecon.utils.seed import set_seed

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("--device cuda was asked for but CUDA is not available")
    device = torch.device(device_name)
    overrides = [f"train.batch_size={args.batch_size}", "train.overfit_batch=true", f"train.overfit_lr={args.lr}",
                 f"train.overfit_steps={args.steps}", f"device={device_name}", f"output_dir={args.out}/runs",
                 *args.override]
    cfg = load_many(BASE_CONFIGS + list(args.config), overrides)
    if not cfg["model"]["phase_branch"].get("enabled"):
        sys.exit("this check needs a config with model.phase_branch.enabled=true (the default configs/model/phasecon.yaml)")
    if cfg["model"]["phase_branch"].get("impl", "real") != "real":
        sys.exit("model.phase_branch.impl must be real for the gates")
    set_seed(int(cfg.get("seed", 0)), False)

    batch = get_batch(cfg, args.synthetic, args.batch_size)
    results = {"batch_source": "synthetic noise (SMOKE TEST, not gate evidence)" if args.synthetic
               else f"real LibriSeVoc train batch ({batch['waveform'].shape[0]} clips, labels {batch['label'].tolist()})",
               "config": cfg}
    for name, fn in (("g3_interface", lambda: check_interface(cfg, batch, device)),
                     ("g4_backward", lambda: check_backward(cfg, batch, device)),
                     ("g4_amp", lambda: check_amp(cfg, batch, device, args.amp_steps)),
                     ("g4_overfit", lambda: check_overfit(cfg, batch, device, args.steps)),
                     ("profile", lambda: profile(cfg, batch, device))):
        print(f"[gate_check] {name} ...", flush=True)
        results[name] = fn()
        print(f"[gate_check] {name}: {results[name]['status']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pr23_gate_check.json").write_text(json.dumps(results, indent=2, default=str))
    md = markdown(results)
    (out / "pr23_gate_check.md").write_text(md)
    print("\n" + md)
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("status") == "FAIL"]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    return results


if __name__ == "__main__":
    main()
