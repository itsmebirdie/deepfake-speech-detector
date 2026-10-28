"""Owner: Member C (PR-24). Tests for scripts/compare_results.py (no model or data needed)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_results", REPO / "scripts" / "compare_results.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

VOC = ["wavegrad", "diffwave", "parallel_wave_gan", "wavernn", "wavenet", "melgan"]


def write_evaluate_style(folder, eer, auc=0.9438, acc=0.7537, voc=None):
    """The layout scripts/evaluate.py writes (PR-17): fractions, per_vocoder.eer."""
    folder.mkdir(parents=True)
    voc = voc or cr.PR06["per_vocoder"]
    (folder / "metrics.json").write_text(json.dumps({
        "eer": eer, "auc": auc, "accuracy": acc, "per_vocoder": {k: {"eer": v, "caught": 0.9} for k, v in voc.items()},
    }))
    return str(folder)


def write_eval_baseline_style(folder, eer, auc=0.9438, acc=0.7537, voc=None):
    """The layout scripts/eval_baseline.py writes (PR-06): eer as fraction AND percent, accuracy_argmax."""
    folder.mkdir(parents=True)
    voc = voc or cr.PR06["per_vocoder"]
    (folder / "metrics.json").write_text(json.dumps({
        "eer": eer, "eer_percent": 100 * eer, "auc": auc, "accuracy_argmax": acc,
        "per_vocoder": {k: {"eer_percent": 100 * v, "detected_as_fake": 0.9} for k, v in voc.items()},
    }))
    return str(folder)


def test_both_metrics_layouts_normalise_to_the_same_thing(tmp_path):
    a = cr.load_metrics(write_evaluate_style(tmp_path / "a", 0.1186))
    b = cr.load_metrics(write_eval_baseline_style(tmp_path / "b", 0.1186))
    assert a["eer"] == b["eer"] and a["accuracy"] == b["accuracy"] and a["auc"] == b["auc"]
    assert a["per_vocoder"].keys() == b["per_vocoder"].keys()
    assert all(a["per_vocoder"][k] == pytest.approx(b["per_vocoder"][k]) for k in VOC)
    # an eer given only as a percentage still works, and a metrics.json path works as well as a folder
    only_pct = tmp_path / "c.json"
    only_pct.write_text(json.dumps({"eer_percent": 11.86, "auc": 0.9, "accuracy_argmax": 0.7}))
    assert cr.load_metrics(str(only_pct))["eer"] == pytest.approx(0.1186)


def test_reference_is_the_pr06_table():
    assert cr.PR06["eer"] == pytest.approx(0.1186) and cr.PR06["auc"] == pytest.approx(0.9438)
    assert list(cr.PR06["per_vocoder"]) == VOC  # same order as data.vocoders
    assert cr.PR06["per_vocoder"]["wavernn"] == pytest.approx(0.3654)
    assert cr.load_reference("pr06") is cr.PR06


def test_parity_passes_for_float_noise_and_fails_for_real_differences(tmp_path):
    ref = cr.PR06
    near = cr.load_metrics(write_evaluate_style(tmp_path / "near", 0.1186 + 0.0001, auc=0.9438 + 0.0002))
    assert all(r["ok"] for r in cr.parity_checks(near, ref))
    worse = cr.load_metrics(write_evaluate_style(tmp_path / "worse", 0.1186 + 0.003))
    failed = [r["name"] for r in cr.parity_checks(worse, ref) if not r["ok"]]
    assert failed == ["eer"]
    # one vocoder far off (a label-order bug would look like this) is caught on its own
    swapped = dict(cr.PR06["per_vocoder"], wavernn=0.0011, wavegrad=0.3654)
    bad = cr.load_metrics(write_evaluate_style(tmp_path / "swap", 0.1186, voc=swapped))
    assert [r["name"] for r in cr.parity_checks(bad, ref) if not r["ok"]] == ["EER wavegrad", "EER wavernn"]
    # custom tolerances are honoured
    assert all(r["ok"] for r in cr.parity_checks(worse, ref, {"eer": 0.01}))


def test_regression_check_allows_improvement_and_small_losses():
    ref = cr.PR06
    better = {"eer": 0.02, "auc": 0.99, "accuracy": 0.95, "per_vocoder": {}}
    assert cr.regression_check(better, ref)["ok"]  # much better: fine, it is not a parity check
    assert cr.regression_check(dict(better, eer=0.1186 + 0.004), ref)["ok"]
    assert not cr.regression_check(dict(better, eer=0.1186 + 0.006), ref)["ok"]
    assert cr.regression_check(dict(better, eer=0.2), ref, max_regression=0.1)["ok"]


def test_table_has_reference_row_and_deltas(tmp_path):
    run = cr.load_metrics(write_evaluate_style(tmp_path / "ft", 0.0300, voc=dict(cr.PR06["per_vocoder"], wavernn=0.05)))
    table = cr.format_table([("reference (pr06)", cr.PR06), ("ft", run)])
    lines = table.splitlines()
    assert lines[0].startswith("| Run | EER % | AUC | Acc % | wavegrad EER %")
    assert "| reference (pr06) | 11.86 | 0.9438 | 75.37 |" in table
    assert "| ft | 3.00 |" in table
    assert "-8.86" in table  # pooled EER change in percentage points
    assert "-31.54" in table  # WaveRNN: 36.54 -> 5.00


def test_cli_exit_codes_and_output(tmp_path, capsys):
    ok = write_eval_baseline_style(tmp_path / "ok", 0.1186)
    bad = write_evaluate_style(tmp_path / "bad", 0.20)
    out = tmp_path / "table.md"
    assert cr.main([ok, "--parity", "--out", str(out)]) == 0
    assert "PASS  eer" in capsys.readouterr().out and out.read_text().startswith("| Run |")
    assert cr.main([bad, "--parity"]) == 1
    assert "FAIL  eer" in capsys.readouterr().out
    assert cr.main([bad, "--no-regress"]) == 1
    assert cr.main([ok, bad, "--no-regress"]) == 1  # any failing run fails the call
    assert cr.main([ok, "--no-regress", "--parity"]) == 0
    assert cr.main([bad]) == 0  # no check requested: just a table
    # a results folder can be the reference, e.g. a locally produced PR-06 run
    assert cr.main([ok, "--reference", bad, "--no-regress"]) == 0  # 11.86% is better than 20%
    with pytest.raises(FileNotFoundError):
        cr.main([str(tmp_path / "missing")])
