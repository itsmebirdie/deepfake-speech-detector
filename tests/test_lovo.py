"""Owner: Member A. Tests for evaluation/lovo.py and scripts/run_lovo.py (PR-26)."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from phasecon.evaluation.lovo import (  # noqa: E402
    fold_config, fold_run_name, format_lovo_table, run_lovo, summarize_lovo,
)
from phasecon.evaluation.results import write_results  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VOCODERS = yaml.safe_load(open(REPO / "configs" / "data" / "librisevoc.yaml"))["data"]["vocoders"]


def tiny_cfg(out_dir):
    return {"model": {"name": "baseline_only"}, "output_dir": str(out_dir), "seed": 1,
            "data": {"vocoders": list(VOCODERS), "lovo": {"held_out": None}}}


def fake_metrics(eer):
    return {"eer": eer, "eer_threshold": 0.0, "auc": 1 - eer, "accuracy": 1 - eer}


def test_fold_config_sets_fold_and_does_not_touch_input(tmp_path):
    cfg = tiny_cfg(tmp_path)
    fold = fold_config(cfg, "wavernn", "baseline_lovo")
    assert fold["data"]["lovo"]["held_out"] == "wavernn"
    assert fold["eval"]["protocol"] == "leave_one_vocoder_out"
    assert fold["run_name"] == "baseline_lovo_lovo-wavernn_train"
    assert cfg["data"]["lovo"]["held_out"] is None and "run_name" not in cfg
    with pytest.raises(ValueError):
        fold_config(cfg, "hifigan", "x")


def test_run_lovo_trains_each_fold_once_and_resumes(tmp_path):
    """One fresh training per fold, each on its own fold; finished folds are skipped on a re-run."""
    cfg = tiny_cfg(tmp_path)
    trained = []

    def train_fold(fold_cfg):
        trained.append(fold_cfg["data"]["lovo"]["held_out"])
        return f"ckpt/{fold_cfg['run_name']}/best.pt"

    def eval_fold(fold_cfg, checkpoint, run_dir):
        v = fold_cfg["data"]["lovo"]["held_out"]
        assert checkpoint == f"ckpt/{fold_run_name('p', v)}_train/best.pt"
        m = fake_metrics(0.01 * (VOCODERS.index(v) + 1))
        write_results(run_dir, fold_cfg, m, {"utt_id": ["a", "b"], "label": [0, 1], "vocoder_id": [0, 1],
                                             "score": [-1.0, 1.0]})
        return m

    out = run_lovo(cfg, "baseline_only", VOCODERS[:2], prefix="p", train_fold=train_fold, eval_fold=eval_fold)
    assert trained == VOCODERS[:2] and list(out) == VOCODERS[:2]

    out = run_lovo(cfg, "baseline_only", VOCODERS[:3], prefix="p", train_fold=train_fold, eval_fold=eval_fold)
    assert trained == VOCODERS[:3]  # only the new fold was trained
    assert out[VOCODERS[0]]["eer"] == pytest.approx(0.01)

    summary = summarize_lovo(out, os.path.join(tmp_path, "results"), "p")
    assert summary["num_folds"] == 3 and summary["mean"]["eer"] == pytest.approx(0.02)
    assert summary["pooled_eer"] == 0.0
    assert "mean of 3 folds" in format_lovo_table(summary)

    with pytest.raises(ValueError):
        run_lovo(cfg, "phasecon_net", VOCODERS[:1], train_fold=train_fold, eval_fold=eval_fold)


def test_end_to_end_tiny_fold(tmp_path, baseline_dir):
    """The real train + evaluate path on a tiny fake dataset: a fold folder appears with per-vocoder metrics."""
    from phasecon.data.splits import build_lovo_splits, read_split_file, write_split
    from tests.test_dataset import make_fake_librisevoc

    names = [f"1_2_00000{i}_000001" for i in range(6)]
    make_fake_librisevoc(tmp_path / "data", names, [1.0] * 6)
    official = {}
    for split, ids in (("train", names[:3]), ("dev", names[3:4]), ("test", names[4:])):
        path = tmp_path / f"{split}.txt"
        path.write_text("\n".join(n + ".wav" for n in ids) + "\n")
        official[split] = read_split_file(str(path), str(tmp_path / "data"), VOCODERS, check_exists=True)
    for name, rows in build_lovo_splits(official, "wavernn", VOCODERS).items():
        write_split(rows, str(tmp_path / "lovo" / "wavernn" / f"{name}.txt"))
    from phasecon.utils.config import load_many

    base = [str(REPO / p) for p in ("configs/default.yaml", "configs/data/librisevoc.yaml",
                                    "configs/train/default.yaml", "configs/model/baseline_only.yaml",
                                    "configs/eval/lovo.yaml")]
    cfg = load_many(base, [
        f"output_dir={tmp_path / 'exp'}", f"data.root={tmp_path / 'data'}",
        f"data.lovo.split_dir={tmp_path / 'lovo'}", "data.num_workers=0", "audio.num_samples=24000",
        "train.epochs=1", "train.batch_size=4", "train.amp=false", "model.baseline.weights=null", "device=cpu",
        f"model.baseline.repo_dir={baseline_dir}",
    ])
    out = run_lovo(cfg, "baseline_only", ["wavernn"], prefix="tiny")
    run_dir = tmp_path / "exp" / "results" / "tiny_lovo-wavernn"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["held_out"] == "wavernn" and list(metrics["per_vocoder"]) == ["wavernn"]
    assert (run_dir / "scores.csv").exists() and out["wavernn"]["eer"] == metrics["eer"]
