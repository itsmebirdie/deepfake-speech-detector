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
