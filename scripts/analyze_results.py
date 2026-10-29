"""Detailed graphs + comparison metrics for evaluation runs. Inference-only, no training.

Put this in scripts/analyze_results.py and run from the repo root:

  pip install matplotlib
  python scripts/analyze_results.py
  python scripts/analyze_results.py --patterns "baseline_*" "phasecon_*" --reference baseline_pipeline_finetune
  python scripts/analyze_results.py --runs baseline_pipeline_finetune phasecon_seed1234

Reads, for every run folder: metrics.json, scores.csv (utt_id,label,vocoder_id,score), config.yaml.
Convention (same as run_inference): score = logit_synthetic - logit_bona_fide, higher = more synthetic,
label 1 = synthetic. "Threshold 0" is the model's natural operating point; "EER threshold" is tuned on the test set.

Writes to experiments/report/ : comparison.md, comparison.csv, per_vocoder.csv, paired_tests.csv, *.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

REPO = Path(__file__).resolve().parents[1]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]


# ----------------------------------------------------------------------------- metrics
def roc_curve(score, y):
    """Return fpr, tpr, thr (predict synthetic when score >= thr). y: 1 = synthetic."""
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    tps, fps = np.cumsum(yy), np.cumsum(1 - yy)
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, tps[idx] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[idx] / max(fps[-1], 1)]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def eer_from_roc(fpr, tpr, thr):
    fnr = 1 - tpr
    d = fpr - fnr  # increases from -1 to +1 as the threshold drops
    i = int(np.searchsorted(d, 0.0))
    if i == 0:
        return float(fpr[0]), float(thr[1]) if len(thr) > 1 else 0.0
    if i >= len(d):
        return float(fpr[-1]), float(thr[-1])
    t = -d[i - 1] / (d[i] - d[i - 1] + 1e-300)
    eer = fpr[i - 1] + t * (fpr[i] - fpr[i - 1])
    return float(eer), float(thr[i])


def eer_auc(score, y):
    fpr, tpr, thr = roc_curve(score, y)
    eer, t = eer_from_roc(fpr, tpr, thr)
    return eer, float(np.trapezoid(tpr, fpr)), t


def eer_only(score, y):
    return eer_from_roc(*roc_curve(score, y))[0]


def confusion(score, y, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def prf(tp, tn, fp, fn):
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1, fp / max(fp + tn, 1), fn / max(fn + tp, 1)


def bootstrap_indices(y, n_boot, seed):
    """Stratified resample indices (same draws can be reused for paired comparisons)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n_boot):
        yield np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]


# ----------------------------------------------------------------------------- loading
def load_runs(args):
    base = REPO / args.results_dir if not Path(args.results_dir).exists() else Path(args.results_dir)
    folders = []
    if args.runs:
        folders = [base / r for r in args.runs]
    else:
        for p in args.patterns:
            folders += [Path(f) for f in sorted(glob.glob(str(base / p)))]
    runs, seen = [], set()
    for f in folders:
        if f.name in seen or not (f / "metrics.json").exists():
            continue
        seen.add(f.name)
        m = json.loads((f / "metrics.json").read_text())
        if not args.include_all and (m.get("protocol", "in_distribution") != "in_distribution" or m.get("held_out")):
            print(f"skip {f.name}: protocol={m.get('protocol')} held_out={m.get('held_out')} (use --include-all)")
            continue
        sc = f / "scores.csv"
        if not sc.exists():
            print(f"skip {f.name}: no scores.csv (needed for curves); re-run evaluate.py with eval.save_scores=true")
            continue
        df = pd.read_csv(sc)
        names = list(m.get("per_vocoder", {}))
        runs.append({"name": f.name, "metrics": m, "df": df, "vocoders": names,
                     "score": df["score"].to_numpy(float), "y": df["label"].to_numpy(int),
                     "vid": df["vocoder_id"].to_numpy(int)})
    if not runs:
        raise SystemExit(f"no usable runs found in {base} (patterns={args.patterns}, runs={args.runs})")
    return runs


def voc_name(run, k):
    v = run["vocoders"]
    return v[k - 1] if 0 < k <= len(v) else f"voc{k}"


# ----------------------------------------------------------------------------- tables
def summary_row(run, n_boot, seed):
    s, y = run["score"], run["y"]
    fpr, tpr, thr = roc_curve(s, y)
    eer, t_eer = eer_from_roc(fpr, tpr, thr)
    auc = float(np.trapezoid(tpr, fpr))
    tp, tn, fp, fn = confusion(s, y, 0.0)
    prec, rec, f1, far, frr = prf(tp, tn, fp, fn)
    tp2, tn2, fp2, fn2 = confusion(s, y, t_eer)
    boots = np.array([eer_only(s[i], y[i]) for i in bootstrap_indices(y, n_boot, seed)]) if n_boot else np.array([eer])
    return {
        "Run": run["name"], "Clips": len(y), "Real": int((y == 0).sum()), "Fake": int((y == 1).sum()),
        "EER %": 100 * eer, "EER 95% CI low": 100 * np.percentile(boots, 2.5),
        "EER 95% CI high": 100 * np.percentile(boots, 97.5),
        "AUC": auc, "EER thr": t_eer,
        "TPR@1%FPR": 100 * np.interp(0.01, fpr, tpr), "TPR@0.1%FPR": 100 * np.interp(0.001, fpr, tpr),
        "Acc@0 %": 100 * (tp + tn) / len(y), "Precision@0 %": 100 * prec, "Recall@0 %": 100 * rec,
        "F1@0 %": 100 * f1, "FAR@0 % (real->fake)": 100 * far, "FRR@0 % (fake->real)": 100 * frr,
        "Acc@EERthr %": 100 * (tp2 + tn2) / len(y),
        "Errors@0": fp + fn, "FP@0": fp, "FN@0": fn,
    }


def per_vocoder_table(runs):
    rows = []
    for run in runs:
        s, y, vid = run["score"], run["y"], run["vid"]
        for k in sorted(set(vid) - {0}):
            m = (vid == 0) | (vid == k)
            eer, auc, _ = eer_auc(s[m], y[m])
            fake = s[vid == k]
            rows.append({"Run": run["name"], "Vocoder": voc_name(run, k), "n": int((vid == k).sum()),
                         "EER %": 100 * eer, "AUC": auc, "Caught@0 %": 100 * float((fake >= 0).mean()),
                         "Mean score": float(fake.mean())})
    return pd.DataFrame(rows)


def paired_tests(runs, ref, n_boot, seed):
    rows = []
    for run in runs:
        if run is ref:
            continue
        a = ref["df"][["utt_id", "label", "score"]].merge(run["df"][["utt_id", "score"]], on="utt_id",
                                                           suffixes=("_ref", "_run"))
        if a.empty:
            continue
        y = a["label"].to_numpy(int)
        sr, sn = a["score_ref"].to_numpy(float), a["score_run"].to_numpy(float)
        ok_r, ok_n = (sr >= 0).astype(int) == y, (sn >= 0).astype(int) == y
        b, c = int((ok_r & ~ok_n).sum()), int((~ok_r & ok_n).sum())  # ref-only right / run-only right
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        deltas = np.array([eer_only(sn[i], y[i]) - eer_only(sr[i], y[i])
                           for i in bootstrap_indices(y, n_boot, seed)]) if n_boot else np.array([np.nan])
        rows.append({"Reference": ref["name"], "Run": run["name"], "Paired clips": len(a),
                     "Ref right, run wrong": b, "Run right, ref wrong": c, "McNemar p (acc@0)": p,
                     "dEER pp (run-ref)": 100 * (eer_only(sn, y) - eer_only(sr, y)),
                     "dEER 95% CI low": 100 * np.percentile(deltas, 2.5),
                     "dEER 95% CI high": 100 * np.percentile(deltas, 97.5)})
    return pd.DataFrame(rows)


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- plots
def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", out / name)


def plot_roc_det(runs, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for c, run in zip(COLORS, runs):
        fpr, tpr, _ = roc_curve(run["score"], run["y"])
        ax[0].plot(np.maximum(fpr, 1e-5), tpr, color=c, label=run["name"])
        fnr = 1 - tpr
        eps = 1e-5
        ax[1].plot(norm.ppf(np.clip(fpr, eps, 1 - eps)), norm.ppf(np.clip(fnr, eps, 1 - eps)), color=c,
                   label=run["name"])
    ax[0].set(xscale="log", xlabel="False positive rate (real flagged fake)", ylabel="True positive rate",
              title="ROC (log FPR axis)", ylim=(0.9, 1.001))
    ax[0].grid(alpha=.3, which="both")
    ticks = np.array([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.4])
    ax[1].set(xticks=norm.ppf(ticks), yticks=norm.ppf(ticks), xlim=(norm.ppf(1e-4), norm.ppf(.5)),
              ylim=(norm.ppf(1e-4), norm.ppf(.5)), xlabel="False alarm rate", ylabel="Miss rate",
              title="DET curve (lower-left is better)")
    ax[1].set_xticklabels([f"{100 * t:g}%" for t in ticks])
    ax[1].set_yticklabels([f"{100 * t:g}%" for t in ticks])
    ax[1].plot([norm.ppf(1e-4), norm.ppf(.5)], [norm.ppf(1e-4), norm.ppf(.5)], "k:", lw=.8, label="EER line")
    ax[1].grid(alpha=.3)
    ax[0].legend(), ax[1].legend()
    save(fig, out, "roc_det.png")


def plot_per_vocoder(pv, out):
    runs = list(dict.fromkeys(pv["Run"]))
    vocs = list(dict.fromkeys(pv["Vocoder"]))
    x, w = np.arange(len(vocs)), 0.8 / len(runs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, (c, r) in enumerate(zip(COLORS, runs)):
        d = pv[pv["Run"] == r].set_index("Vocoder").reindex(vocs)
        b = ax[0].bar(x + j * w - 0.4 + w / 2, d["EER %"], w, color=c, label=r)
        ax[0].bar_label(b, fmt="%.2f", fontsize=7, padding=1)
        ax[1].bar(x + j * w - 0.4 + w / 2, d["Caught@0 %"], w, color=c, label=r)
    for a in ax:
        a.set_xticks(x, vocs, rotation=20)
        a.grid(axis="y", alpha=.3)
    ax[0].set(ylabel="EER % (real vs this vocoder)", title="Per-vocoder EER (lower is better)")
    ax[1].set(ylabel="Caught %", title="Fakes caught at threshold 0 (higher is better)")
    lo = max(0, pv["Caught@0 %"].min() - 2)
    ax[1].set_ylim(lo, 100.5)
    ax[0].legend()
    save(fig, out, "per_vocoder.png")


def plot_score_dists(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        s, y = run["score"], run["y"]
        lo, hi = np.percentile(s, [0.2, 99.8])
        bins = np.linspace(lo, hi, 80)
        ax.hist(np.clip(s[y == 0], lo, hi), bins, alpha=.6, color="#2ca02c", label="bona fide", density=True)
        ax.hist(np.clip(s[y == 1], lo, hi), bins, alpha=.6, color="#d62728", label="synthetic", density=True)
        ax.axvline(0, color="k", ls="--", lw=1, label="threshold 0")
        ax.axvline(eer_auc(s, y)[2], color="b", ls=":", lw=1, label="EER thr")
        ax.set(title=run["name"], xlabel="score (logit fake - logit real)", yscale="log")
        ax.legend(fontsize=8)
    save(fig, out, "score_distributions.png")


def plot_vocoder_scores(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.5), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        ids = sorted(set(run["vid"]))
        data = [run["score"][run["vid"] == k] for k in ids]
        labels = ["bona fide" if k == 0 else voc_name(run, k) for k in ids]
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, k in zip(bp["boxes"], ids):
            patch.set_facecolor("#2ca02c" if k == 0 else "#d62728")
            patch.set_alpha(.5)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set(title=run["name"], ylabel="score")
        ax.tick_params(axis="x", rotation=30)
    save(fig, out, "score_by_vocoder.png")


def plot_confusions(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        tp, tn, fp, fn = confusion(run["score"], run["y"], 0.0)
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm[i].sum():.2f}%)", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred real", "pred fake"],
               yticklabels=["real", "fake"], title=f"{run['name']} @ thr 0")
    save(fig, out, "confusion_matrices.png")


def plot_far_frr(runs, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, run in zip(COLORS, runs):
        fpr, tpr, thr = roc_curve(run["score"], run["y"])
        m = np.isfinite(thr)
        ax.plot(thr[m], 100 * fpr[m], color=c, ls="-", label=f"{run['name']} FAR")
        ax.plot(thr[m], 100 * (1 - tpr[m]), color=c, ls="--", label=f"{run['name']} FRR")
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.set(xlabel="threshold", ylabel="error rate % (log)", yscale="log",
           title="FAR (real->fake, solid) and FRR (fake->real, dashed) vs threshold")
    ax.set_ylim(0.01, 100)
    ax.set_xlim(-8, 8)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7)
    save(fig, out, "far_frr_vs_threshold.png")


def plot_overall_bars(summary, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(summary))
    names = summary["Run"].tolist()
    err = np.vstack([summary["EER %"] - summary["EER 95% CI low"], summary["EER 95% CI high"] - summary["EER %"]])
    ax[0].bar(x, summary["EER %"], yerr=np.clip(err, 0, None), color=COLORS[:len(x)], capsize=4)
    ax[0].set(title="EER % with 95% bootstrap CI (lower is better)")
    ax[1].bar(x, summary["Errors@0"], color=COLORS[:len(x)])
    ax[1].set(title="Total misclassified clips @ threshold 0")
    w = .38
    ax[2].bar(x - w / 2, summary["FAR@0 % (real->fake)"], w, label="FAR (real->fake)", color="#2ca02c")
    ax[2].bar(x + w / 2, summary["FRR@0 % (fake->real)"], w, label="FRR (fake->real)", color="#d62728")
    ax[2].set(title="Error split @ threshold 0 (%)")
    ax[2].legend()
    for a in ax:
        a.set_xticks(x, names, rotation=20, ha="right")
        a.grid(axis="y", alpha=.3)
    save(fig, out, "overall_comparison.png")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Detailed graphs + comparison metrics for evaluation runs")
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--patterns", nargs="*", default=["baseline_*", "phasecon_*"])
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run folder names (overrides --patterns)")
    ap.add_argument("--reference", default=None, help="run for paired tests (default: first baseline*)")
    ap.add_argument("--out", default="experiments/report")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples (0 disables CIs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-all", action="store_true", help="also include LOVO / non-in-distribution runs")
    args = ap.parse_args()

    runs = load_runs(args)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    print("runs:", ", ".join(r["name"] for r in runs))

    summary = pd.DataFrame([summary_row(r, args.bootstrap, args.seed) for r in runs])
    pv = per_vocoder_table(runs)
    ref = next((r for r in runs if r["name"] == args.reference), None) or \
        next((r for r in runs if r["name"].startswith("baseline")), runs[0])
    paired = paired_tests(runs, ref, args.bootstrap, args.seed)

    # sanity check against the numbers evaluate.py stored
    for r, (_, row) in zip(runs, summary.iterrows()):
        stored = r["metrics"].get("eer")
        if stored is not None and abs(stored * 100 - row["EER %"]) > 0.05:
            print(f"WARNING {r['name']}: recomputed EER {row['EER %']:.3f}% differs from metrics.json "
                  f"{100 * stored:.3f}% (check score direction / label convention)")

    summary.to_csv(out / "comparison.csv", index=False)
    pv.to_csv(out / "per_vocoder.csv", index=False)
    paired.to_csv(out / "paired_tests.csv", index=False)

    plot_roc_det(runs, out)
    plot_per_vocoder(pv, out)
    plot_score_dists(runs, out)
    plot_vocoder_scores(runs, out)
    plot_confusions(runs, out)
    plot_far_frr(runs, out)
    plot_overall_bars(summary, out)

    wide = pv.pivot(index="Vocoder", columns="Run", values="EER %").reset_index()
    wide_caught = pv.pivot(index="Vocoder", columns="Run", values="Caught@0 %").reset_index()
    md = [f"# Evaluation comparison\n\nRuns: {', '.join(r['name'] for r in runs)}. Reference for paired tests: "
          f"`{ref['name']}`. Bootstrap resamples: {args.bootstrap}.\n",
          "## Overall metrics (test split)\n", md_table(summary.set_index("Run").T.reset_index().rename(
              columns={"index": "Metric"})), "\n## Per-vocoder EER % (real vs that vocoder)\n", md_table(wide),
          "\n## Per-vocoder fakes caught % at threshold 0\n", md_table(wide_caught)]
    if len(paired):
        md += ["\n## Paired comparison vs reference\n",
               "McNemar tests whether the two models' accuracy at threshold 0 differs on the same clips; "
               "the dEER interval comes from paired bootstrap (if it excludes 0 the EER difference is "
               "unlikely to be sampling noise).\n", md_table(paired, "{:.4g}")]
    md += ["\n## Figures\n"] + [f"![{p.stem}]({p.name})" for p in sorted(out.glob("*.png"))]
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", out / "comparison.md")
    print("\n" + summary[["Run", "EER %", "EER 95% CI low", "EER 95% CI high", "AUC", "Acc@0 %", "F1@0 %",
                           "TPR@1%FPR"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()