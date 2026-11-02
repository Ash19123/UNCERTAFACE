"""Training loop with checkpointing, resume, early stopping and health monitors.

Usage:
  python -m src.training.trainer --config configs/training.yaml --model configs/model.yaml \
      --set run.name=prob optim.epochs=40 model.probabilistic=true

Outputs in ``<out_dir>/<name>/``: config.json, experiment.json, history.json,
metrics.csv, latest.pt, best.pt (+ tb/ if TensorBoard is installed).

Validation numbers are in millimetres (converted with the train-fitted
Normalizer). "Convergence" is never inferred from the training loss alone:
validation metrics, uncertainty statistics and warnings are all recorded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data import dataset as ds
from ..data import preprocessing as pp
from ..losses.composite import CompositeLoss
from ..losses.regularization import MeshOps
from ..models.uncertaface import UncertaFace
from ..uncertainty.estimation import gaussian_to_mm
from . import checkpoint as ck
from .metrics_logger import MetricsLogger


class TrainingDiverged(RuntimeError):
    """Raised when the loss stays non-finite for too many consecutive steps."""


def set_seed(seed: int, deterministic: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a nested dict copy."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def hardware_info(device: torch.device) -> dict:
    info = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                device=str(device))
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    return info


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        r = cfg["run"]
        self.run_dir = Path(r["out_dir"]) / r["name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if (r["device"] == "auto" and torch.cuda.is_available())
                                   else ("cpu" if r["device"] == "auto" else r["device"]))
        set_seed(r["seed"])
        root = Path(r["data_root"])
        nfile = root / "metadata/normalizer.json"
        self.norm = pp.Normalizer.load(nfile) if nfile.exists() else pp.fit_normalizer_from_train(root)
        self.root = root
        self.norm.save(self.run_dir / "normalizer.json")          # each run carries the transform it was trained with
        d, a = cfg["data"], cfg["augmentation"]
        mk = lambda split, aug: ds.UncertafaceDataset(
            root, split, self.norm, d["n_points"], d["knn_k"], a, aug, d["use_missing_info"], r["seed"])
        self.train_ds, self.val_ds = mk("train", True), mk("validation", False)
        template = ds.fit_template(root, self.norm, pp.read_split(root, "train")["base_ids"])
        faces = np.load(root / "processed/correspondence/correspondence.npz")["face_faces"]
        mcfg = copy.deepcopy(cfg["model"])
        mcfg["use_missing_info"] = bool(mcfg["use_missing_info"] and d["use_missing_info"])
        self.model = UncertaFace(mcfg, template, self.norm.scale).to(self.device)
        self.init_info = None
        if r.get("init_from"):
            # warm start: weights only (no optimiser/epoch state); the variance head, absent
            # from a deterministic checkpoint, keeps its fresh initialisation
            state = ck.load_checkpoint(r["init_from"], self.device)["model"]
            res = self.model.load_state_dict(state, strict=False)
            bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if not k.startswith("head.raw")]
            if bad:
                raise ValueError(f"init_from checkpoint does not match the model: {bad}")
            self.init_info = dict(path=str(r["init_from"]), fresh_params=list(res.missing_keys))
        self.loss_fn = CompositeLoss(cfg["loss"], MeshOps(faces), self.norm.scale).to(self.device)
        o = cfg["optim"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
        self.steps_per_epoch = math.ceil(len(self.train_ds) / o["batch_size"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.amp = bool(o["amp"] and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.logger = MetricsLogger(self.run_dir, r["tensorboard"])
        self.history: list[dict] = []
        self.epoch, self.best, self.best_epoch, self.bad_epochs, self.status = 0, float("inf"), -1, 0, "new"
        self.std_lo, self.std_hi = cfg["model"]["head"]["std_bounds_mm"]

    # ---- schedules --------------------------------------------------------
    def _lr_factor(self, step: int) -> float:
        o = self.cfg["optim"]
        warm = o["warmup_epochs"] * self.steps_per_epoch
        total = o["epochs"] * self.steps_per_epoch
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return o["min_lr_frac"] + (1 - o["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p))

    def nll_factor(self, epoch: int) -> float:
        w = self.cfg["loss"]["nll_warmup_epochs"]
        return 1.0 if w <= 0 else min(1.0, epoch / w)

    # ---- data -------------------------------------------------------------
    def _to_device(self, b: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def _loader(self, d, shuffle: bool, epoch: int) -> DataLoader:
        o, seed = self.cfg["optim"], self.cfg["run"]["seed"]
        return ds.make_loader(d, o["batch_size"], shuffle, seed=seed * 100003 + epoch, num_workers=o["num_workers"])

    # ---- one epoch --------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_ds.set_epoch(epoch)
        mon, clip = self.cfg["monitors"], self.cfg["optim"]["grad_clip"]
        acc: dict[str, float] = {}
        n_batches = skipped = bad_run = big_grad = 0
        gmax, gsum = 0.0, 0.0
        for batch in self._loader(self.train_ds, True, epoch):
            b = self._to_device(batch)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                out = self.model.forward_batch(b)
            loss, terms = self.loss_fn(out, b["face"], self.nll_factor(epoch))
            if not torch.isfinite(loss):
                bad_run += 1
                skipped += 1
                self.opt.zero_grad(set_to_none=True)
                if bad_run > mon["max_nonfinite_steps"]:
                    raise TrainingDiverged(f"non-finite loss for {bad_run} consecutive steps (epoch {epoch})")
                continue
            bad_run = 0
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            g = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
            if not math.isfinite(g):                      # overflow (AMP) or bad gradient: skip the step
                skipped += 1
                self.scaler.update()
                continue
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            gmax, gsum = max(gmax, g), gsum + g
            big_grad += g > mon["grad_norm_warn"]
            n_batches += 1
            acc["loss"] = acc.get("loss", 0.0) + float(loss)
            for k, v in terms.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        if n_batches == 0:
            raise TrainingDiverged(f"epoch {epoch}: no successful step, {skipped} skipped (non-finite loss/gradient)")
        out = {f"train_{k}": v / max(1, n_batches) for k, v in acc.items()}
        out.update(train_steps=n_batches, train_skipped_steps=skipped, grad_norm_mean=gsum / max(1, n_batches),
                   grad_norm_max=gmax, grad_norm_over_warn=int(big_grad), lr=self.opt.param_groups[0]["lr"])
        return out

    @torch.no_grad()
    def evaluate(self, d=None) -> dict:
        """Validation metrics in mm. Loss uses the FULL NLL weight (no warm-up ramp)."""
        self.model.eval()
        d = d or self.val_ds
        tot, n = {}, 0
        for batch in self._loader(d, False, 0):
            b = self._to_device(batch)
            out = self.model.forward_batch(b)
            loss, terms = self.loss_fn(out, b["face"], 1.0)
            B = len(b["face"])
            phys = gaussian_to_mm(out["mean"].float(), None if out["logvar"] is None else out["logvar"].float(), self.norm)
            err = phys["mean_mm"] - b["face_mm"]
            # "rmse_mm" accumulates the mean squared 3D vertex error; sqrt is taken after averaging
            add = {"loss": float(loss), "rmse_mm": float(err.pow(2).sum(-1).mean()),
                   "mae_mm": float(err.abs().mean()), "max_vertex_err_mm": float(err.norm(dim=-1).max())}
            for k, v in terms.items():
                add[k] = float(v)
            if out["logvar"] is not None:
                std = phys["std_mm"]
                add.update(nll_mm=float((0.5 * (torch.log(phys["var_mm2"]) + err ** 2 / phys["var_mm2"])).mean()),
                           pos_std_mm=float(phys["pos_std_mm"].mean()),
                           z2=float((err ** 2 / phys["var_mm2"]).mean()),
                           frac_std_at_lower=float((std < self.std_lo * 1.05).float().mean()),
                           frac_std_at_upper=float((std > self.std_hi * 0.95).float().mean()))
            for k, v in add.items():
                tot[k] = tot.get(k, 0.0) + v * B
            n += B
        m = {f"val_{k}": v / n for k, v in tot.items()}
        m["val_rmse_mm"] = math.sqrt(m["val_rmse_mm"])
        return m

    # ---- health checks ----------------------------------------------------
    def check_health(self, row: dict) -> list[str]:
        mon, w = self.cfg["monitors"], []
        if row["train_steps"] == 0:
            w.append("no successful training steps this epoch")
        if row["train_skipped_steps"]:
            w.append(f"{row['train_skipped_steps']} steps skipped (non-finite loss/gradient)")
        if row["grad_norm_over_warn"]:
            w.append(f"exploding gradients: {row['grad_norm_over_warn']} steps with norm > {mon['grad_norm_warn']}")
        if not math.isfinite(row.get("val_loss", 0.0)):
            w.append("non-finite validation loss")
        if row.get("val_frac_std_at_lower", 0) > mon["collapse_frac"]:
            w.append(f"uncertainty collapse: {row['val_frac_std_at_lower']:.0%} of stds at lower bound")
        if row.get("val_frac_std_at_upper", 0) > mon["inflation_frac"]:
            w.append(f"uncertainty inflation: {row['val_frac_std_at_upper']:.0%} of stds at upper bound")
        k = mon["overfit_patience"]
        if len(self.history) >= k:
            v = [h["val_loss"] for h in self.history[-k:]] + [row["val_loss"]]
            t = [h["train_loss"] for h in self.history[-k:]] + [row["train_loss"]]
            if all(b > a for a, b in zip(v, v[1:])) and t[-1] < t[0]:
                w.append(f"possible overfitting: val_loss rose {k} epochs while train_loss fell")
        return w

    # ---- checkpoint -------------------------------------------------------
    def _state(self, epoch_done: int) -> dict:
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), sched=self.sched.state_dict(),
                    scaler=self.scaler.state_dict(), epoch=epoch_done, history=self.history, best=self.best,
                    best_epoch=self.best_epoch, bad_epochs=self.bad_epochs, status=self.status,
                    rng=ck.rng_state(), config=self.cfg)

    def _resume(self) -> bool:
        p = self.run_dir / "latest.pt"
        if not (self.cfg["run"]["resume"] and p.exists()):
            return False
        s = ck.load_checkpoint(p, self.device)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.scaler.load_state_dict(s["scaler"])
        self.history, self.best, self.best_epoch = s["history"], s["best"], s["best_epoch"]
        self.bad_epochs, self.epoch, self.status = s["bad_epochs"], s["epoch"], s["status"]
        ck.set_rng_state(s["rng"])
        return True

    # ---- main loop --------------------------------------------------------
    def fit(self, log=print) -> dict:
        cfg, es = self.cfg, self.cfg["early_stopping"]
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        resumed = self._resume()
        if resumed:
            log(f"resumed from epoch {self.epoch} (status {self.status})")
        t0, prior = time.time(), (self.history[-1].get("elapsed_s", 0.0) if self.history else 0.0)
        # early stopping is final; "completed" only counts if the configured epoch budget was reached
        if self.status == "early_stopped" or (self.status == "completed" and self.epoch >= cfg["optim"]["epochs"]):
            log(f"already {self.status}; nothing to do")
        else:
            self.status = "running"
            try:
                while self.epoch < cfg["optim"]["epochs"]:
                    e = self.epoch
                    row = dict(epoch=e, **self.train_epoch(e), **self.evaluate())
                    row["elapsed_s"] = prior + time.time() - t0
                    row["warnings"] = self.check_health(row)
                    self.history.append(row)
                    self.epoch = e + 1
                    cur = row[es["monitor"]]
                    if cur < self.best - es["min_delta"]:
                        self.best, self.best_epoch, self.bad_epochs = cur, e, 0
                        ck.save_checkpoint(self.run_dir / "best.pt", self._state(self.epoch))
                    else:
                        self.bad_epochs += 1
                    if self.bad_epochs >= es["patience"]:
                        self.status = "early_stopped"
                    elif self.epoch >= cfg["optim"]["epochs"]:
                        self.status = "completed"
                    ck.save_checkpoint(self.run_dir / "latest.pt", self._state(self.epoch))
                    self.logger.log(e, row)
                    self.logger.write_history(self.history)
                    log(f"ep {e:3d} train {row['train_loss']:.4f} val {row['val_loss']:.4f} "
                        f"rmse {row['val_rmse_mm']:.2f}mm lr {row['lr']:.2e}"
                        + (f" std {row['val_pos_std_mm']:.2f}mm" if "val_pos_std_mm" in row else "")
                        + (f"  WARN {row['warnings']}" if row["warnings"] else ""))
                    if self.status == "early_stopped":
                        break
            except TrainingDiverged as ex:
                self.status = "diverged"
                self.history.append(dict(epoch=self.epoch, warnings=[str(ex)], train_loss=float("nan"), val_loss=float("nan")))
                self.logger.write_history(self.history)
                log(f"DIVERGED: {ex}")
                ck.save_checkpoint(self.run_dir / "latest.pt", self._state(self.epoch))
        final = self._finalise(time.time() - t0 + prior)
        self.logger.close()
        return final

    def _finalise(self, duration: float) -> dict:
        best_p = self.run_dir / "best.pt"
        final_val = None
        if best_p.exists():
            self.model.load_state_dict(ck.load_checkpoint(best_p, self.device)["model"])
            final_val = self.evaluate()
        rep = self.root / "dataset_report.json"
        ds_info = {}
        if rep.exists():
            r = json.loads(rep.read_text())
            ds_info = dict(version=r.get("dataset_version"), config_sha256=r.get("config_sha256"),
                           counts=r["counts"], synthetic=r["synthetic"])
        exp = dict(name=self.cfg["run"]["name"], status=self.status, seed=self.cfg["run"]["seed"],
                   epochs_run=self.epoch, best_epoch=self.best_epoch, best_monitor=self.best,
                   monitor=self.cfg["early_stopping"]["monitor"], duration_s=duration,
                   hardware=hardware_info(self.device), dataset=ds_info, splits_dir=str(self.root / "splits"),
                   model_params=self.model.num_parameters(), model=self.cfg["model"], loss=self.cfg["loss"],
                   optim=self.cfg["optim"], augmentation=self.cfg["augmentation"], data=self.cfg["data"],
                   checkpoint=dict(best=str(best_p), latest=str(self.run_dir / "latest.pt")),
                   normalizer=self.norm.to_dict(), best_validation_metrics=final_val, init_from=self.init_info,
                   all_warnings=[dict(epoch=h["epoch"], warnings=h["warnings"]) for h in self.history if h.get("warnings")])
        (self.run_dir / "experiment.json").write_text(json.dumps(exp, indent=2))
        return exp


def load_config(training_path, model_path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(training_path).read_text())
    cfg.update(yaml.safe_load(Path(model_path).read_text()))
    return apply_overrides(cfg, overrides)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--model", default="configs/model.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.epochs=10")
    a = ap.parse_args()
    exp = Trainer(load_config(a.config, a.model, a.set)).fit()
    print(json.dumps({k: exp[k] for k in ("name", "status", "epochs_run", "best_epoch", "best_monitor", "duration_s")}, indent=2))


if __name__ == "__main__":
    main()
