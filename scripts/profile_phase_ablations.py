"""PR-27: time and memory of one training step for each phase-side ablation setting. Owner: Member B.

  python scripts/profile_phase_ablations.py                 # batch 4, 4 s clips, 3 repeats, random weights
  python scripts/profile_phase_ablations.py --batch-size 8 --repeats 5

Settings: the default model (K=5, R=3), (i) raw phase, (ii) fixed Griffin-Lim, and the (vi) sweeps K in {1,3,5,8}
and R in {1,2,3}. For each: phase-branch parameter count, median seconds for one forward + backward of the whole
model, and memory. Memory is the CUDA peak on a GPU; on a CPU it is the size of the tensors autograd keeps for the
backward pass (the part that grows with K and R). Random audio and random weights: speed and memory do not depend
on the content. Writes experiments/profiles/phase_ablations.md.
"""
import argparse
import os
import statistics
import time
from pathlib import Path

import torch

from phasecon.losses.combined import total_loss
from phasecon.models import build_model
from phasecon.utils.config import load_many

REPO = Path(__file__).resolve().parents[1]
BASE = ["configs/default.yaml", "configs/data/librisevoc.yaml", "configs/train/default.yaml"]
RES = [[512, 128, 512], [1024, 256, 1024], [2048, 512, 2048]]
SETTINGS = [
    ("default (K=5, R=3)", [], []),
    ("(i) raw phase", ["configs/ablation/a1_raw_phase.yaml"], []),
    ("(ii) fixed Griffin-Lim", ["configs/ablation/a2_fixed_griffin_lim.yaml"], []),
    *[(f"(vi) K={k}", [], [f"model.phase_branch.ulcp.num_steps={k}"]) for k in (1, 3, 8)],
    *[(f"(vi) R={r}", [], [f"model.phase_branch.stft_resolutions={RES[3 - r:]}"]) for r in (1, 2)],
]


def step_cost(model, wav, batch, device):
    """Seconds and memory (bytes) for one forward + backward."""
    saved = 0

    def pack(t):
        nonlocal saved
        saved += t.numel() * t.element_size()
        return t

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss, _ = total_loss(model(wav), batch, beta=0.5, mu=0.1)
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return seconds, torch.cuda.max_memory_allocated() if device == "cuda" else saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_kind = "CUDA peak" if device == "cuda" else "autograd saved tensors"
    rows = []
    for name, files, overrides in SETTINGS:
        cfg = load_many([str(REPO / p) for p in BASE + ["configs/model/phasecon.yaml", *files]],
                        ["model.baseline.weights=null", *overrides, *args.override])
        torch.manual_seed(0)
        model = build_model(cfg).to(device).train()
        n = cfg["audio"]["num_samples"]
        wav = torch.randn(args.batch_size, n, device=device) * 0.1
        label = torch.arange(args.batch_size, device=device) % 2
        batch = {"label": label, "vocoder_id": label}
        step_cost(model, wav, batch, device)  # warm-up
        costs = [step_cost(model, wav, batch, device) for _ in range(args.repeats)]
        params = sum(p.numel() for p in model.phase_branch.parameters())
        seconds = statistics.median(c[0] for c in costs)
        mem = max(c[1] for c in costs) / 2**20
        rows.append((name, params, seconds, mem))
        print(f"{name:24s} phase params {params:>8,}  {seconds:6.2f} s/step  {mem:8.0f} MiB", flush=True)
        del model

    table = [f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else os.cpu_count()} "
             f"{'' if device == 'cuda' else 'CPU threads'}), batch {args.batch_size} x 4 s clips, "
             f"median of {args.repeats}; memory = {memory_kind}.", "",
             "| Setting | Phase-branch params | Seconds per step | Memory (MiB) |", "|---|---|---|---|"]
    table += [f"| {n} | {p:,} | {s:.2f} | {m:.0f} |" for n, p, s, m in rows]
    out = REPO / "experiments" / "profiles" / "phase_ablations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(table) + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
