"""Owner: Member A. Tests for evaluation/metrics.py and evaluation/evaluate.py (PR-07)."""
import numpy as np
import pytest

from phasecon.evaluation.metrics import compute_accuracy, compute_auc, compute_eer, summarize


def asvspoof_eer(scores, labels):
    """Independent reference: the ASVspoof challenge formula (detection-error curve), written with
    numpy only. ASVspoof scores are 'higher = bona fide', so the synthetic score is negated."""
    bona = -scores[labels == 0]
    spoof = -scores[labels == 1]
    all_scores = np.concatenate((bona, spoof))
    lab = np.concatenate((np.ones(bona.size), np.zeros(spoof.size)))
    order = np.argsort(all_scores, kind="mergesort")
    lab = lab[order]
    tar_trial_sums = np.cumsum(lab)
    non_target_trial_sums = spoof.size - (np.arange(1, lab.size + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / bona.size))
    far = np.concatenate((np.atleast_1d(1), non_target_trial_sums / spoof.size))
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i])))


def test_eer_perfect_separation():
    """EER == 0 for perfectly separable scores."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    eer, thr = compute_eer(scores, labels)
    assert eer == 0.0
    assert compute_accuracy(scores, labels, thr) == 1.0
    eer_flipped, _ = compute_eer(-scores, labels)  # every clip on the wrong side
    assert eer_flipped == 1.0


def test_eer_random_scores():
    """EER ~ 0.5 for random scores (tolerance)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 20_000)
    scores = rng.standard_normal(20_000)
    eer, _ = compute_eer(scores, labels)
    assert abs(eer - 0.5) < 0.02


def brute_force_eer(scores, labels):
    """Tie-aware reference: try every distinct score as the threshold ('synthetic' = score >= t)."""
    best = None
    for t in np.concatenate((np.unique(scores), [np.inf])):
        far = np.mean(scores[labels == 0] >= t)  # bona fide called synthetic
        frr = np.mean(scores[labels == 1] < t)  # synthetic called bona fide
        if best is None or abs(frr - far) < best[0]:
            best = (abs(frr - far), (frr + far) / 2)
    return float(best[1])


def test_eer_matches_baseline_implementation():
    """Same scores -> same EER as the reference implementations.

    The baseline repo has no EER function (docs/DATASET_NOTES.md section 6), so the references are
    the ASVspoof challenge formula (scores without ties) and a brute-force threshold sweep (also
    with tied scores, where the ASVspoof formula splits ties in arbitrary order)."""
    rng = np.random.default_rng(1)
    for shift in (0.0, 0.5, 1.5, 3.0):
        labels = np.concatenate([np.zeros(700, int), np.ones(4200, int)])  # 1 real : 6 fake, like LibriSeVoc
        scores = rng.standard_normal(labels.size) + shift * labels
        eer, _ = compute_eer(scores, labels)
        assert eer == pytest.approx(asvspoof_eer(scores, labels), abs=1e-12)
        assert eer == pytest.approx(brute_force_eer(scores, labels), abs=1e-12)
    tied_labels = rng.integers(0, 2, 2000)
    tied = np.round(rng.standard_normal(2000) + tied_labels, 1)
    assert compute_eer(tied, tied_labels)[0] == pytest.approx(brute_force_eer(tied, tied_labels), abs=1e-12)


def test_auc_known_value():
    """AUC on a hand-built example equals the analytic value."""
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # synthetic/bona fide pairs ranked correctly: (0.35>0.1), (0.8>0.1), (0.8>0.4); wrong: (0.35<0.4) -> 3/4
    assert compute_auc(scores, labels) == pytest.approx(0.75)


def test_accuracy_threshold():
    """Accuracy at a given threshold on a toy set."""
    labels = np.array([0, 0, 1, 1, 1])
    scores = np.array([-1.0, 0.5, -0.2, 0.3, 2.0])
    assert compute_accuracy(scores, labels, 0.0) == pytest.approx(3 / 5)  # 0.5 and -0.2 are wrong
    assert compute_accuracy(scores, labels, 0.6) == pytest.approx(3 / 5)  # 0.5 right now, -0.2 and 0.3 wrong
    assert compute_accuracy(scores, labels, -5.0) == pytest.approx(3 / 5)  # everything called synthetic


def test_summarize_and_per_vocoder():
    labels = np.array([0, 0, 1, 1, 1, 1])
    vocoder_ids = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array([-2.0, -1.0, 3.0, 2.0, -0.5, 1.0])  # vocoder 2 has one fake that looks real
    out = summarize(scores, labels, vocoder_ids, ["wavegrad", "diffwave"])
    assert set(out) >= {"eer", "eer_threshold", "auc", "accuracy", "accuracy_at_eer", "per_vocoder"}
    assert out["per_vocoder"]["wavegrad"]["eer"] == 0.0
    assert out["per_vocoder"]["wavegrad"]["caught"] == 1.0
    assert out["per_vocoder"]["diffwave"]["caught"] == 0.5
    assert out["bona_fide_correct"] == 1.0
    assert out["accuracy"] == pytest.approx(5 / 6)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, 0.2]), np.array([1, 1]))  # no bona fide clips
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, np.nan]), np.array([0, 1]))
    with pytest.raises(ValueError):
        compute_auc(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))


def test_evaluate_uses_only_logits_bin(tmp_path):
    torch = pytest.importorskip("torch")
    from phasecon.evaluation.evaluate import evaluate, run_inference

    class Dummy(torch.nn.Module):
        """Says 'synthetic' when the waveform's mean is positive. Returns extra keys that must be ignored."""

        def forward(self, wav):
            m = wav.mean(dim=1)
            return {"logits_bin": torch.stack([-m, m], dim=1), "logits_voc": torch.zeros(len(wav), 7)}

    def batch(means, labels, vids):
        return {
            "waveform": torch.tensor(means, dtype=torch.float32)[:, None].repeat(1, 10),
            "label": torch.tensor(labels),
            "vocoder_id": torch.tensor(vids),
            "utt_id": [f"u{i}" for i in range(len(means))],
        }

    loader = [batch([-1.0, 2.0, 3.0], [0, 1, 1], [0, 1, 2]), batch([-2.0, 0.5], [0, 1], [0, 1])]
    model = Dummy().train()
    res = run_inference(model, loader, "cpu")
    assert model.training  # restored afterwards
    assert np.allclose(res["score"], [-2.0, 4.0, 6.0, -4.0, 1.0])
    out = evaluate(model, loader, "cpu", save_path=str(tmp_path / "scores.csv"), vocoders=["wavegrad", "diffwave"])
    assert out["eer"] == 0.0 and out["accuracy"] == 1.0
    assert set(out["per_vocoder"]) == {"wavegrad", "diffwave"}
    rows = (tmp_path / "scores.csv").read_text().strip().splitlines()
    assert rows[0] == "utt_id,label,vocoder_id,score" and len(rows) == 6


def test_results_round_trip(tmp_path):
    """write_results -> read_results gives back the same config, metrics and scores; no silent overwrite."""
    pytest.importorskip("torch")
    from phasecon.evaluation.results import read_results, write_results

    cfg = {"seed": 1, "data": {"vocoders": ["wavegrad"]}, "model": {"name": "baseline_only"}}
    scores = {
        "utt_id": ["a", "a", "b", "b"],
        "label": np.array([0, 1, 0, 1]),
        "vocoder_id": np.array([0, 1, 0, 1]),
        "score": np.array([-1.5, 2.25, 0.125, 3.0]),
    }
    metrics = summarize(scores["score"], scores["label"], scores["vocoder_id"], ["wavegrad"])
    run_dir = str(tmp_path / "run1")
    write_results(run_dir, cfg, metrics, scores)
    back = read_results(run_dir)
    assert back["config"] == cfg
    for k in ("eer", "eer_threshold", "auc", "accuracy", "per_vocoder"):
        assert back["metrics"][k] == metrics[k]
    assert {"git_hash", "git_dirty", "created"} <= set(back["metrics"])
    assert back["scores"]["utt_id"] == scores["utt_id"]
    assert np.array_equal(back["scores"]["label"], scores["label"])
    assert np.array_equal(back["scores"]["vocoder_id"], scores["vocoder_id"])
    assert np.allclose(back["scores"]["score"], scores["score"])
    with pytest.raises(FileExistsError):
        write_results(run_dir, cfg, metrics, scores)
    write_results(run_dir, cfg, metrics, None, overwrite=True)
    with pytest.raises(ValueError):
        write_results(str(tmp_path / "bad"), cfg, {"eer": 0.1}, None)


def test_evaluate_script_end_to_end(tmp_path, baseline_dir, random_baseline_weights):
    """scripts/evaluate.py on the baseline produces a complete results folder (in-distribution and LOVO).

    Uses a tiny fake LibriSeVoc and the baseline architecture with fixed random weights; on a machine
    with the real data, run the script itself (see its docstring)."""
    import importlib.util
    from pathlib import Path

    sf = pytest.importorskip("soundfile")
    yaml = pytest.importorskip("yaml")
    from phasecon.data.splits import build_lovo_splits, read_split_file, write_split
    from phasecon.evaluation.results import read_results

    repo = Path(__file__).resolve().parents[1]
    vocoders = yaml.safe_load(open(repo / "configs" / "data" / "librisevoc.yaml"))["data"]["vocoders"]
    names = ["1_2_000001_000001", "1_2_000002_000001"]
    rng = np.random.default_rng(0)
    for utt in names:
        for folder, fname in [("gt", f"{utt}.wav")] + [(v, f"{utt}_gen.wav") for v in vocoders]:
            (tmp_path / "data" / folder).mkdir(parents=True, exist_ok=True)
            sf.write(tmp_path / "data" / folder / fname, (rng.standard_normal(30000) * 0.1).astype(np.float32), 24000)
    split = tmp_path / "test.txt"
    split.write_text("\n".join(n + ".wav" for n in names) + "\n")
    entries = read_split_file(str(split), str(tmp_path / "data"), vocoders, check_exists=True)
    fold = build_lovo_splits({"train": [], "dev": [], "test": entries}, "wavernn", vocoders)
    write_split(fold["test"], str(tmp_path / "lovo" / "wavernn" / "test.txt"))

    spec = importlib.util.spec_from_file_location("evaluate_script", repo / "scripts" / "evaluate.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    common = [
        f"model.baseline.repo_dir={baseline_dir}", f"model.baseline.weights={random_baseline_weights}",
        "audio.sample_rate=24000", "audio.num_samples=24000", f"data.root={tmp_path / 'data'}",
        f"data.split_files.test={split}", "data.num_workers=0", f"output_dir={tmp_path / 'exp'}",
        f"data.lovo.split_dir={tmp_path / 'lovo'}",
    ]
    run_dir = script.main(["--override", *common])
    res = read_results(run_dir)
    assert res["metrics"]["n_clips"] == 14 and len(res["scores"]["score"]) == 14
    assert set(res["metrics"]["per_vocoder"]) == set(vocoders)
    assert res["config"]["model"]["name"] == "baseline_only"
    with pytest.raises(SystemExit):  # never overwrite silently
        script.main(["--override", *common])
    lovo_dir = script.main(["--config", "configs/model/baseline_only.yaml", "configs/eval/lovo.yaml",
                            "--override", *common, "data.lovo.held_out=wavernn", "eval.split=test"])
    lovo = read_results(lovo_dir)
    assert lovo_dir.endswith("lovo-wavernn")
    assert lovo["metrics"]["held_out"] == "wavernn" and lovo["metrics"]["n_clips"] == 4
    assert set(lovo["metrics"]["per_vocoder"]) == {"wavernn"}
