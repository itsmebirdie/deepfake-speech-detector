"""Training loop. Owner: Member C (PR-20).
Uses evaluation.evaluate (Member A) for the per-epoch dev EER used by early stopping."""
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import yaml

from phasecon.evaluation.evaluate import evaluate
from phasecon.losses.combined import total_loss
from phasecon.utils.logging import get_logger
from phasecon.utils.seed import set_seed

from .callbacks import EarlyStopping, MetricLogger
from .checkpoint import load_checkpoint, save_checkpoint
from .optim import build_optimizer, build_scheduler

logger = get_logger("phasecon.train")


def _make_grad_scaler(enabled: bool):
    """torch.amp.GradScaler exists from torch 2.3; requirements.txt allows 2.1."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Config read from cfg['train'] (see configs/train/default.yaml) plus cfg['seed'], cfg['device'],
    cfg['output_dir'], cfg.get('run_name'), cfg['data']['vocoders'] (names for per-vocoder dev rows).

    Model contract: model(wav) -> dict as in docs/INTERFACES.md section 5, and
    model.param_groups(lr_encoder, lr_new, weight_decay). Loaders are any iterables of batches
    {waveform, label, vocoder_id, utt_id} with a len().

    Files: <output_dir>/checkpoints/<run_name>/{best.pt,last.pt,config.yaml} and
    <output_dir>/logs/<run_name>/{metrics.csv,tb/}. Refuses to overwrite an existing run unless resuming.
    Extra train keys: amp_dtype (float16|bfloat16), max_nonfinite_skips (10), resume (checkpoint path),
    overfit_batch (bool), overfit_steps (100), overfit_lr (None).
    """

    def __init__(self, cfg: Dict, model, train_loader, dev_loader=None):
        self.cfg = cfg
        tc = cfg["train"]
        set_seed(int(cfg.get("seed", 0)), bool(cfg.get("deterministic", False)))

        want = str(cfg.get("device", "cpu"))
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("device=%s requested but CUDA is not available: using the CPU", want)
            want = "cpu"
        self.device = torch.device(want)
        self.model = model.to(self.device)
        self.train_loader, self.dev_loader = train_loader, dev_loader
        self.steps_per_epoch = len(train_loader)
        if self.steps_per_epoch == 0:
            raise ValueError("the training loader is empty")

        self.epochs = int(tc["epochs"])
        self.beta = float(tc["loss"]["beta"])
        self.mu = float(tc["loss"]["mu"])
        self.grad_clip = float(tc["grad_clip"]) if tc.get("grad_clip") else None
        self.max_nonfinite = int(tc.get("max_nonfinite_skips", 10))
        self.use_amp = bool(tc.get("amp", False)) and self.device.type == "cuda"
        self.amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(tc.get("amp_dtype", "float16"))]
        self.scaler = _make_grad_scaler(self.use_amp and self.amp_dtype == torch.float16)

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, self.steps_per_epoch)
        self._params: List[torch.nn.Parameter] = [p for g in self.optimizer.param_groups for p in g["params"]]

        # best-checkpoint selection and (optional) early stopping share one tracker
        es = tc.get("early_stopping")
        if es:
            key, mode, patience = es["metric"], es.get("mode", "min"), int(es["patience"])
        else:
            key, mode, patience = ("dev_eer" if dev_loader is not None else "train_loss"), "min", 10**9
        if key.startswith("dev_") and dev_loader is None and not tc.get("overfit_batch"):
            raise ValueError(f"metric {key!r} needs a dev loader")
        self.early_stopping_enabled = bool(es)
        self.stopper = EarlyStopping(key, mode, patience)

        run_name = cfg.get("run_name") or f"{cfg['model']['name']}_seed{cfg.get('seed', 0)}"
        out = cfg.get("output_dir", "experiments")
        self.run_name = run_name
        self.ckpt_dir = os.path.join(out, "checkpoints", run_name)
        self.log_dir = os.path.join(out, "logs", run_name)
        self.best_path = os.path.join(self.ckpt_dir, "best.pt")
        self.last_path = os.path.join(self.ckpt_dir, "last.pt")
        self.save_best = bool((tc.get("checkpoint") or {}).get("save_best", True))
        self.save_last = bool((tc.get("checkpoint") or {}).get("save_last", True))
        self.start_epoch = 0
        resume = tc.get("resume")
        if not resume and not tc.get("overfit_batch"):
            existing = [p for p in (self.best_path, self.last_path) if os.path.exists(p)]
            if existing:
                raise FileExistsError(
                    f"{existing[0]} already exists. Pick another run_name (train override run_name=...) "
                    "or resume with train.resume=<checkpoint>."
                )
        self.logger = MetricLogger(self.log_dir)
        self.nonfinite_skipped = 0
        self._consecutive_bad = 0
        if resume:
            self.resume(resume)

    # ---- one optimisation step -------------------------------------------------------------
    def _train_step(self, batch: Dict, step_scheduler: bool = True) -> Optional[Dict[str, float]]:
        """Returns the batch's loss parts + accuracy, or None if the batch was skipped (non-finite loss)."""
        wav = batch["waveform"].to(self.device, non_blocking=True)
        target = {k: batch[k].to(self.device, non_blocking=True) for k in ("label", "vocoder_id")}
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(wav)
        loss, parts = total_loss(out, target, self.beta, self.mu)  # losses are computed in float32
        if not torch.isfinite(loss):
            self.nonfinite_skipped += 1
            self._consecutive_bad += 1
            logger.warning("non-finite loss (%s): batch skipped (%d total)", loss.item(), self.nonfinite_skipped)
            if self._consecutive_bad > self.max_nonfinite:
                raise RuntimeError(f"more than {self.max_nonfinite} consecutive non-finite losses: stopping")
            self.optimizer.zero_grad(set_to_none=True)
            return None
        self._consecutive_bad = 0
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._params, self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if step_scheduler:
            self.scheduler.step()
        parts = dict(parts)
        parts["loss"] = float(loss.detach())
        parts["acc"] = float((out["logits_bin"].detach().argmax(1) == target["label"]).float().mean())
        return parts

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """One pass over the training loader. Returns batch-size-weighted means of loss / bin / voc /
        cons / acc, the number of skipped batches, and the current learning rate of each group."""
        self.model.train()
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        skipped = 0
        for batch in self.train_loader:
            n = int(batch["waveform"].shape[0])
            stats = self._train_step(batch)
            if stats is None:
                skipped += 1
                continue
            for k, v in stats.items():
                sums[k] += v * n
                counts[k] += n
        if not counts:
            raise RuntimeError(f"epoch {epoch}: every batch was skipped")
        result = {k: sums[k] / counts[k] for k in sums}
        result["skipped_batches"] = float(skipped)
        for g in self.optimizer.param_groups:
            result[f"lr_{g.get('name', 'group')}"] = float(g["lr"])
        return result

    def evaluate_dev(self) -> Dict[str, float]:
        vocoders = (self.cfg.get("data") or {}).get("vocoders")
        m = evaluate(self.model, self.dev_loader, str(self.device), vocoders=vocoders)
        scalars = {"dev_eer": m["eer"], "dev_auc": m["auc"], "dev_accuracy": m["accuracy"],
                   "dev_accuracy_at_eer": m["accuracy_at_eer"]}
        for name, v in m.get("per_vocoder", {}).items():
            scalars[f"dev_eer_{name}"] = v["eer"]
        return scalars

    # ---- checkpoints -----------------------------------------------------------------------
    def _save(self, path: str, epoch: int) -> None:
        save_checkpoint(
            path, self.model, self.optimizer, epoch, self.stopper.best, self.cfg,
            scheduler=self.scheduler, scaler=self.scaler,
            extra={
                "early_stopping": self.stopper.state_dict(),
                "select_metric": self.stopper.metric,
                "phase_branch_is_dummy": bool(getattr(self.model, "phase_branch_is_dummy", False)),
            },
        )

    def resume(self, path: str) -> None:
        """Restore model, optimizer, scheduler, scaler, epoch counter and best-metric tracking."""
        ckpt = load_checkpoint(path, self.model, self.optimizer, map_location=self.device)
        if ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        es_state = (ckpt.get("extra") or {}).get("early_stopping")
        if es_state:
            self.stopper.load_state_dict(es_state)
        self.start_epoch = int(ckpt["epoch"]) + 1
        logger.info("resumed from %s at epoch %d (git %s)", path, self.start_epoch, ckpt.get("git_hash"))

    # ---- main loops -----------------------------------------------------------------------
    def fit(self) -> str:
        """Run training; return path of best checkpoint (by dev EER, or the configured metric).
        Supports: AMP (the phase stream stays fp32 inside the model), grad clipping, resume,
        deterministic seeding, early stopping and `train.overfit_batch` (see `overfit_batch`)."""
        if self.cfg["train"].get("overfit_batch"):
            result = self.overfit_batch()
            logger.info("overfit_batch: loss %.4f -> %.4f", result["initial_loss"], result["final_loss"])
            self._save(self.last_path, 0)
            return self.last_path
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(os.path.join(self.ckpt_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            stats = self.train_epoch(epoch)
            scalars = {f"train_{k}" if not k.startswith("lr_") else k: v for k, v in stats.items()}
            if self.dev_loader is not None:
                scalars.update(self.evaluate_dev())
            scalars["epoch_seconds"] = time.time() - t0
            self.logger.log(epoch, scalars)
            logger.info("epoch %d: %s", epoch, {k: round(v, 4) for k, v in scalars.items() if not k.startswith("dev_eer_")})
            if self.stopper.metric not in scalars:
                raise KeyError(f"early-stopping metric {self.stopper.metric!r} is not among {sorted(scalars)}")
            stop = self.stopper.step(scalars[self.stopper.metric])
            if self.save_last:
                self._save(self.last_path, epoch)
            if self.save_best and self.stopper.is_best:
                self._save(self.best_path, epoch)
            if stop and self.early_stopping_enabled:
                logger.info("early stopping after epoch %d (no %s improvement for %d epochs)",
                            epoch, self.stopper.metric, self.stopper.patience)
                break
        self.logger.close()
        if self.save_best and os.path.exists(self.best_path):
            return self.best_path
        return self.last_path if os.path.exists(self.last_path) else ""

    def overfit_batch(self, num_steps: Optional[int] = None) -> Dict:
        """Debug mode: train on ONE fixed batch (the first of the training loader) for `num_steps`
        steps (train.overfit_steps, default 100), constant LR (train.overfit_lr overrides every
        group's LR), no dev evaluation. A healthy model drives the loss to ~0.
        Returns {'initial_loss', 'final_loss', 'min_loss', 'losses', 'accuracy'}."""
        tc = self.cfg["train"]
        steps = int(num_steps or tc.get("overfit_steps", 100))
        if tc.get("overfit_lr"):
            for g in self.optimizer.param_groups:
                g["lr"] = float(tc["overfit_lr"])
        batch = next(iter(self.train_loader))
        self.model.train()
        losses: List[float] = []
        acc = 0.0
        for _ in range(steps):
            stats = self._train_step(batch, step_scheduler=False)
            if stats is not None:
                losses.append(stats["loss"])
                acc = stats["acc"]
        if not losses:
            raise RuntimeError("every overfit step produced a non-finite loss")
        return {"initial_loss": losses[0], "final_loss": losses[-1], "min_loss": min(losses),
                "losses": losses, "accuracy": acc}
