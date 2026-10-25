"""Owner: Member C (PR-20). Tests for training/*.py.

The trainer is tested with a tiny stand-in model so no baseline repo or dataset is needed; one
integration test at the end runs it on the real PhaseConNet (dummy phase branch) if the baseline
repo is available.
"""
import csv
import math

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from phasecon.training.callbacks import EarlyStopping, MetricLogger  # noqa: E402
from phasecon.training.checkpoint import git_info, load_checkpoint, save_checkpoint  # noqa: E402
from phasecon.training.optim import build_optimizer, build_scheduler, lr_factor  # noqa: E402
from phasecon.training.trainer import Trainer  # noqa: E402

VOCODERS = ["a", "b"]


class TinyNet(torch.nn.Module):
    """Same interface as PhaseConNet: forward -> dict, param_groups()."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(8, 6)
        self.head = torch.nn.Linear(6, 2)
        self.voc = torch.nn.Linear(6, 3)

    def forward(self, wav):
        h = torch.tanh(self.enc(wav))
        return {"logits_bin": self.head(h), "logits_voc": self.voc(h)}

    def param_groups(self, lr_encoder, lr_new, weight_decay):
        return [
            {"name": "encoder", "params": list(self.enc.parameters()), "lr": lr_encoder, "weight_decay": weight_decay},
            {"name": "new", "params": list(self.head.parameters()) + list(self.voc.parameters()),
             "lr": lr_new, "weight_decay": weight_decay},
        ]


def make_batches(n_batches=3, size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(n_batches):
        wav = torch.randn(size, 8, generator=g)
        label = (wav[:, 0] > 0).long()
        label[0], label[1] = 0, 1  # both classes in every batch (EER needs them)
        vid = label * (1 + (torch.arange(size) % 2))
        batches.append({"waveform": wav, "label": label, "vocoder_id": vid, "utt_id": [f"u{i}_{j}" for j in range(size)]})
    return batches


def make_cfg(tmp_path, **train):
    t = {
        "epochs": 3, "batch_size": 8, "amp": False, "grad_clip": 5.0,
        "optimizer": {"name": "adam", "lr_encoder": 1e-2, "lr_new": 1e-2, "weight_decay": 0.0},
        "scheduler": {"name": "cosine", "warmup_epochs": 1},
        "early_stopping": {"metric": "dev_eer", "mode": "min", "patience": 5},
        "loss": {"beta": 0.5, "mu": 0.0},
        "checkpoint": {"save_best": True, "save_last": True},
    }
    t.update(train)
    return {"seed": 0, "device": "cpu", "output_dir": str(tmp_path), "run_name": "run",
            "model": {"name": "tiny"}, "data": {"vocoders": VOCODERS}, "train": t}


def test_one_epoch_runs_on_dummy_data(tmp_path):
    """Trainer completes an epoch on a tiny fake loader."""
    cfg = make_cfg(tmp_path, epochs=1)
    tr = Trainer(cfg, TinyNet(), make_batches(), make_batches(2, seed=1))
    stats = tr.train_epoch(0)
    assert set(stats) >= {"loss", "bin", "voc", "acc", "skipped_batches", "lr_encoder", "lr_new"}
    assert math.isfinite(stats["loss"]) and stats["skipped_batches"] == 0
    assert "cons" not in stats  # tiny model has no phase branch
    best = tr.fit()
    assert best.endswith("best.pt")
    rows = list(csv.DictReader(open(tmp_path / "logs" / "run" / "metrics.csv")))
    assert len(rows) == 1 and {"train_loss", "dev_eer", "dev_auc", "epoch_seconds"} <= set(rows[0])
    assert (tmp_path / "checkpoints" / "run" / "config.yaml").exists()
    assert (tmp_path / "checkpoints" / "run" / "last.pt").exists()


def test_training_reduces_loss_and_dev_error(tmp_path):
    cfg = make_cfg(tmp_path, epochs=30, early_stopping=None,
                   optimizer={"name": "adam", "lr_encoder": 3e-2, "lr_new": 3e-2, "weight_decay": 0.0})
    tr = Trainer(cfg, TinyNet(), make_batches(4, 16), make_batches(2, 32, seed=1))
    first = tr.train_epoch(0)["loss"]
    tr.fit()
    last = tr.train_epoch(99)["loss"]
    assert last < 0.4 * first  # measured: about 0.16
    assert tr.evaluate_dev()["dev_eer"] < 0.25  # measured: about 0.11 on unseen data


def test_checkpoint_roundtrip(tmp_path):
    """Save/load restores weights and optimizer state."""
    cfg = make_cfg(tmp_path)
    model = TinyNet()
    cfg["train"]["early_stopping"] = None  # no dev loader here
    tr = Trainer(cfg, model, make_batches(), None)
    tr.train_epoch(0)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, tr.optimizer, 4, 0.123, cfg, scheduler=tr.scheduler)
    other = TinyNet()
    opt2 = build_optimizer(other, cfg)
    ckpt = load_checkpoint(path, other, opt2)
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)
    s1, s2 = tr.optimizer.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and len(s1) > 0
    assert all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
    assert ckpt["epoch"] == 4 and ckpt["best_metric"] == pytest.approx(0.123)
    assert ckpt["cfg"] == cfg and "git_hash" in ckpt and isinstance(ckpt["git_dirty"], bool)
    assert set(git_info()) == {"git_hash", "git_dirty"}
    with pytest.raises(ValueError):  # a checkpoint without optimizer state cannot restore one
        save_checkpoint(tmp_path / "noopt.pt", model, None, 0, None, cfg)
        load_checkpoint(tmp_path / "noopt.pt", other, opt2)


def test_early_stopping_logic():
    """Stops after `patience` non-improving epochs."""
    es = EarlyStopping("dev_eer", "min", patience=3)
    assert [es.step(v) for v in (0.5, 0.4, 0.41, 0.42)] == [False, False, False, False]
    assert es.best == 0.4 and es.num_bad_epochs == 2
    assert es.step(0.43) is True  # third bad epoch in a row
    es = EarlyStopping("dev_eer", "min", patience=2)
    assert not es.step(0.5) and not es.step(0.6) and es.step(0.7)
    es = EarlyStopping("dev_auc", "max", patience=2)
    assert not es.step(0.5) and es.is_best and not es.step(0.6) and es.is_best and not es.step(0.55)
    assert es.step(0.6)  # equal to the best is not an improvement
    es = EarlyStopping("m", "min", patience=2)
    es.step(0.5)
    assert not es.step(float("nan")) and es.step(float("nan"))  # NaN never improves
    es2 = EarlyStopping("m", "min", patience=2)
    es2.load_state_dict(es.state_dict())
    assert es2.best == es.best and es2.num_bad_epochs == es.num_bad_epochs
    with pytest.raises(ValueError):
        EarlyStopping("m", "sideways", 1)
    with pytest.raises(ValueError):
        EarlyStopping("m", "min", 0)


def test_trainer_early_stops_and_keeps_best(tmp_path):
    cfg = make_cfg(tmp_path, epochs=50, early_stopping={"metric": "dev_eer", "mode": "min", "patience": 2})
    cfg["train"]["optimizer"].update(lr_encoder=0.0, lr_new=0.0)  # nothing improves after epoch 0
    tr = Trainer(cfg, TinyNet(), make_batches(), make_batches(2, seed=1))
    best = tr.fit()
    rows = list(csv.DictReader(open(tmp_path / "logs" / "run" / "metrics.csv")))
    assert len(rows) == 3  # epoch 0 sets the best, epochs 1 and 2 do not improve -> stop
    assert best.endswith("best.pt") and torch.load(best, weights_only=False)["epoch"] == 0


def test_scheduler_warmup_and_cosine():
    assert lr_factor(0, 4, 20) == pytest.approx(0.25)
    assert lr_factor(3, 4, 20) == pytest.approx(1.0)
    assert lr_factor(4, 4, 20) == pytest.approx(1.0)  # decay starts at 1
    assert lr_factor(12, 4, 20) == pytest.approx(0.5)  # halfway through the decay
    assert lr_factor(20, 4, 20) == pytest.approx(0.0, abs=1e-12)
    assert lr_factor(500, 4, 20) == pytest.approx(0.0, abs=1e-12)  # stays at the floor
    assert lr_factor(7, 0, 20, "constant") == 1.0
    model = TinyNet()
    cfg = {"train": {"epochs": 5, "optimizer": {"name": "adam", "lr_encoder": 1e-5, "lr_new": 1e-4},
                     "scheduler": {"name": "cosine", "warmup_epochs": 1}}}
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=4)
    lrs = []
    for _ in range(20):
        lrs.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx([1e-5 * 0.25, 1e-4 * 0.25])  # each group keeps its own base LR
    assert lrs[3] == pytest.approx([1e-5, 1e-4])
    assert all(lrs[i][1] >= lrs[i + 1][1] for i in range(3, 19))  # decays monotonically
    with pytest.raises(ValueError):
        build_optimizer(model, {"train": {"optimizer": {"name": "sgd", "lr_encoder": 1, "lr_new": 1}}})
    with pytest.raises(ValueError):
        build_scheduler(opt, {"train": {"epochs": 1, "scheduler": {"name": "step"}}}, 4)


def test_resume_continues_where_it_stopped(tmp_path):
    cfg = make_cfg(tmp_path, epochs=2, early_stopping=None)
    tr = Trainer(cfg, TinyNet(), make_batches(), make_batches(2, seed=1))
    tr.fit()
    ck = torch.load(tmp_path / "checkpoints" / "run" / "last.pt", weights_only=False)
    assert ck["epoch"] == 1
    cfg2 = make_cfg(tmp_path, epochs=4, early_stopping=None, resume=str(tmp_path / "checkpoints" / "run" / "last.pt"))
    tr2 = Trainer(cfg2, TinyNet(), make_batches(), make_batches(2, seed=1))
    assert tr2.start_epoch == 2
    for a, b in zip(tr.model.state_dict().values(), tr2.model.state_dict().values()):
        assert torch.equal(a, b)
    tr2.fit()
    assert torch.load(tmp_path / "checkpoints" / "run" / "last.pt", weights_only=False)["epoch"] == 3
    rows = list(csv.DictReader(open(tmp_path / "logs" / "run" / "metrics.csv")))
    assert [r["step"] for r in rows][-2:] == ["2", "3"]


def test_refuses_to_overwrite_a_finished_run(tmp_path):
    cfg = make_cfg(tmp_path, epochs=1)
    Trainer(cfg, TinyNet(), make_batches(), make_batches(2, seed=1)).fit()
    with pytest.raises(FileExistsError):
        Trainer(make_cfg(tmp_path, epochs=1), TinyNet(), make_batches(), make_batches(2, seed=1))


def test_dev_metric_needs_dev_loader(tmp_path):
    with pytest.raises(ValueError):
        Trainer(make_cfg(tmp_path), TinyNet(), make_batches(), None)
    with pytest.raises(ValueError):
        Trainer(make_cfg(tmp_path), TinyNet(), [], None)


def test_overfit_batch_drives_loss_to_zero(tmp_path):
    cfg = make_cfg(tmp_path, overfit_batch=True, overfit_steps=300, early_stopping=None,
                   optimizer={"name": "adam", "lr_encoder": 3e-2, "lr_new": 3e-2, "weight_decay": 0.0})
    tr = Trainer(cfg, TinyNet(), make_batches(), None)
    res = tr.overfit_batch()
    assert res["initial_loss"] > 0.5 and res["final_loss"] < 0.05 and res["accuracy"] == 1.0
    assert len(res["losses"]) == 300
    assert tr.fit().endswith("last.pt")  # fit() delegates in this mode and saves a checkpoint


def test_nonfinite_loss_is_skipped_then_aborts(tmp_path):
    class Bad(TinyNet):
        def forward(self, wav):
            out = super().forward(wav)
            out["logits_bin"] = out["logits_bin"] * float("nan")
            return out

    cfg = make_cfg(tmp_path, early_stopping=None, max_nonfinite_skips=2)
    tr = Trainer(cfg, Bad(), make_batches(5), None)
    before = {k: v.clone() for k, v in tr.model.state_dict().items()}
    with pytest.raises(RuntimeError, match="non-finite"):
        tr.train_epoch(0)
    assert all(torch.equal(before[k], v) for k, v in tr.model.state_dict().items())  # nothing was updated
    assert tr.nonfinite_skipped == 3


def test_metric_logger_writes_csv_with_changing_columns(tmp_path):
    log = MetricLogger(str(tmp_path / "lg"))
    log.log(0, {"a": 1.0})
    log.log(1, {"a": 2.0, "b": 3.0})
    log.close()
    rows = list(csv.DictReader(open(tmp_path / "lg" / "metrics.csv")))
    assert rows[0]["b"] == "" and rows[1]["b"] == "3.0" and [r["step"] for r in rows] == ["0", "1"]


def test_integration_phasecon_net_with_dummy_branch(tmp_path, baseline_dir, random_baseline_weights):
    """The trainer drives the real PhaseConNet (baseline + fusion + heads, dummy phase stream) and the
    full model overfits one batch."""
    from phasecon.models import build_model
    from phasecon.utils.config import load_many

    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"
    cfg = load_many(
        [str(root / "default.yaml"), str(root / "data" / "librisevoc.yaml"), str(root / "train" / "default.yaml"),
         str(root / "model" / "phasecon.yaml")],
        [f"model.baseline.repo_dir={baseline_dir}", f"model.baseline.weights={random_baseline_weights}",
         "model.phase_branch.impl=dummy", "model.phase_branch.encoder.t_out=8", "model.fusion.dim=16",
         "model.fusion.dropout=0.0", "audio.sample_rate=24000", "audio.num_samples=96000",
         "train.loss.mu=0.0", "train.amp=false", "train.overfit_batch=true", "train.overfit_steps=25",
         "train.overfit_lr=0.0003", f"output_dir={tmp_path}", "run_name=integ", "device=cpu"],
    )
    torch.manual_seed(0)
    wav = torch.randn(4, 12_000) * 0.1
    vid = torch.tensor([0, 1, 4, 0])
    batch = {"waveform": wav, "label": (vid > 0).long(), "vocoder_id": vid, "utt_id": list("abcd")}
    tr = Trainer(cfg, build_model(cfg), [batch], None)
    res = tr.overfit_batch()
    assert all(math.isfinite(x) for x in res["losses"])
    # measured: 1.67 -> ~0.000 in 25 steps. (lr 3e-3 is unstable for the whole model: the loss spikes.)
    # A FROZEN random encoder cannot be overfitted: its last-step features differ by ~1e-4 between
    # clips, so the heads only learn class priors (loss stays at about 1.21). Check that with real weights.
    assert res["initial_loss"] > 1.0 and res["final_loss"] < 0.05
