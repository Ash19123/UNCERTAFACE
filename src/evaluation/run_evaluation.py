"""Full evaluation of one trained run.

Split roles (never mixed):  validation = model/checkpoint selection (already used in training),
calibration = fitting conformal calibrators ONLY, test = final reporting ONLY.

Usage:
  python -m src.evaluation.run_evaluation --run experiments/runs/probabilistic_warm [--no-sweeps]

Writes ``reports/results/<run>/results.json`` and figures next to it. All numbers are
SYNTHETIC-data results in millimetres, computed here from the saved checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from ..data import dataset as ds
from ..uncertainty import calibration as cal
from ..visualization import uncertainty_viewer as uv
from ..visualization.results import load_run
from . import reconstruction_metrics as rm
from . import sparsification as sp
from . import stress_tests as st
from . import uncertainty_metrics as um
from .predict import collect

def _eval_cfg(path="configs/evaluation.yaml") -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


_EC = _eval_cfg()
LEVELS = tuple(_EC.get("confidence_levels", (0.5, 0.8, 0.9, 0.95)))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def _dataset(root, split, norm, cfg):
    return ds.UncertafaceDataset(root, split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"],
                                 use_missing_info=cfg["data"]["use_missing_info"], seed=cfg["run"]["seed"])


def uncertainty_vs_missingness(p: dict) -> dict:
    """Does predicted uncertainty rise with missingness? Reported with the actual error trend."""
    err = np.linalg.norm(p["mean"] - p["gt"], axis=-1)                      # (N,V)
    out = dict(spearman_missing_vs_error=float(spearmanr(p["missing_fraction"], np.sqrt((err ** 2).mean(1)))[0]))
    if p["std"] is not None:
        unc = np.sqrt((p["std"] ** 2).sum(-1))                              # (N,V)
        out["spearman_missing_vs_pred_std"] = float(spearmanr(p["missing_fraction"], unc.mean(1))[0])
        per = [spearmanr(err[i], unc[i])[0] for i in range(len(err))]
        out["within_sample_spearman_err_vs_std_mean"] = float(np.nanmean(per))
        out["pooled_spearman_err_vs_std"] = float(spearmanr(err.ravel(), unc.ravel())[0])
    return out


def evaluate_run(run_dir, out_dir=None, eval_split="test", device="cpu", sweeps=True, n_sweep_bases=30,
                 dataset_cfg_path="configs/dataset.yaml") -> dict:
    run_dir = Path(run_dir)
    model, norm, cfg, root = load_run(run_dir, device)
    out_dir = Path(out_dir or Path("reports/results") / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    faces = np.load(root / "processed/correspondence/correspondence.npz")["face_faces"]
    template_mm = norm.inverse(model.decoder.template.cpu().numpy())
    groups = cal.make_vertex_groups(template_mm, 6, 0)

    cal_p = collect(model, norm, _dataset(root, "calibration", norm, cfg), device)
    test_p = collect(model, norm, _dataset(root, eval_split, norm, cfg), device)
    res: dict = dict(run=run_dir.name, eval_split=eval_split, synthetic=True, units="mm",
                     probabilistic=test_p["std"] is not None, n_test=len(test_p["gt"]),
                     n_calibration=len(cal_p["gt"]))

    per = rm.evaluate_batch(test_p["mean"], test_p["gt"], faces)
    res["reconstruction"] = rm.summarise(per)
    res["reconstruction_by_severity"] = {
        sv: rm.summarise({k: v[np.array(test_p["severity"]) == sv] for k, v in per.items()})
        for sv in ("mild", "moderate", "severe", "extreme") if (np.array(test_p["severity"]) == sv).any()}
    err = np.linalg.norm(test_p["mean"] - test_p["gt"], axis=-1)
    res["missingness_relationship"] = uncertainty_vs_missingness(test_p)

    if test_p["std"] is not None:
        m, s, y = test_p["mean"], test_p["std"], test_p["gt"]
        unc = np.sqrt((s ** 2).sum(-1))
        res["uncertainty_raw"] = um.summary(m, s, y, LEVELS)
        res["calibration"] = {}
        rel = {"raw": um.reliability_curve(m, s, y)}
        for scope in cal.SCOPES:
            c = cal.ConformalCalibrator(scope, groups=groups if scope == "per_group" else None, levels=LEVELS)
            c.fit(cal_p["mean"], cal_p["std"], cal_p["gt"], base_id=cal_p["base_id"])
            c.save(out_dir / f"calibrator_{scope}.json")
            res["calibration"][scope] = dict(
                n_scores=c.n_scores, q={str(k): np.asarray(v).tolist() for k, v in c.q.items()},
                test=cal.evaluate_intervals(m, s, y, c, LEVELS, groups))
            if scope == "global":
                lv = np.linspace(0.05, 0.95, 19)
                c19 = cal.ConformalCalibrator("global", levels=tuple(lv)).fit(cal_p["mean"], cal_p["std"], cal_p["gt"])
                rel["global conformal"] = (lv, np.array([cal.evaluate_intervals(m, s, y, c19, (p,))[f"{int(round(p*100))}"]["pointwise"] for p in lv]))
        res["calibration"]["raw"] = cal.evaluate_intervals(m, s, y, None, LEVELS, groups)
        sc = sp.sparsification_curves(err, unc)
        res["sparsification"] = {k: v for k, v in sc.items() if not isinstance(v, np.ndarray)}
        res["risk_coverage"] = sp.risk_coverage(err, unc)
        res["best_of_n"] = sp.best_of_n(m, s, y, int(_EC.get("best_of_n_samples", 10)))
        res["best_of_n"]["note"] = "oracle-selected using ground truth; NOT a single-prediction result"
        uv.reliability_plot(rel, out_dir / "reliability.png")
        uv.sparsification_plot(sc, out_dir / "sparsification.png", run_dir.name)
        uv.error_vs_uncertainty_plot(err, unc, out_dir / "error_vs_uncertainty.png")
        uv.spatial_map_plot(template_mm, err.mean(0), unc.mean(0), out_dir / "spatial_maps.png")
        res["spatial_map_corr_err_vs_std"] = float(np.corrcoef(err.mean(0), unc.mean(0))[0, 1])

    tables = {"severity": st.group_table(test_p, "severity"), "pattern": st.group_table(test_p, "pattern"),
              "n_points": st.group_table(test_p, "n_points"), "sampler": st.group_table(test_p, "sampler"),
              "noise_mm": st.group_table(test_p, "noise_sigma_mm", _EC.get("grouped_noise_bins_mm", [0, 0.5, 1.0, 1.51]))}
    res["stress_grouped"] = tables
    if sweeps:
        dcfg = yaml.safe_load(Path(dataset_cfg_path).read_text())
        bases = sorted(set(test_p["base_id"].tolist()))[:n_sweep_bases]
        sw = {}
        for factor, values in {k: _EC.get("controlled_sweeps", {}).get({"noise": "noise_mm"}.get(k, k), v) for k, v in dict(
                noise=[0.0, 0.5, 1.0, 2.0, 3.0], density=[256, 512, 1024, 2048, 4096],
                sampler=["area", "fps"], deformation=[0.0, 0.03, 0.06, 0.10]).items()}.items():
            sw[factor] = st.controlled_sweep(model, norm, cfg, root, bases, factor, values, dcfg, cfg["data"]["knn_k"], device)
        sw["missingness"] = st.controlled_sweep(
            model, norm, cfg, root, bases, "missingness",
            [(sv, pt) for sv in ("mild", "moderate", "severe", "extreme") for pt in dcfg["observation"]["patterns"]],
            dcfg, cfg["data"]["knn_k"], device)
        res["stress_controlled"] = sw
        uv.grouped_bar_plot({k: v for k, v in sw.items() if k != "missingness"}, out_dir / "stress_sweeps.png",
                            title=f"{run_dir.name}: controlled sweeps (vertex RMSE, mm)")
    uv.grouped_bar_plot({k: v for k, v in tables.items()}, out_dir / "stress_grouped.png",
                        title=f"{run_dir.name}: test-set breakdown (vertex RMSE, mm)")
    hist = json.loads((run_dir / "history.json").read_text())
    uv.training_curves(hist, out_dir / "training_curves.png")
    res["training"] = dict(epochs_run=len(hist), warnings_epochs=sum(bool(h.get("warnings")) for h in hist))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(res), indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-sweeps", action="store_true")
    a = ap.parse_args()
    r = evaluate_run(a.run, eval_split=a.split, device=a.device, sweeps=not a.no_sweeps)
    print(json.dumps(_jsonable({k: r[k] for k in ("run", "reconstruction", "missingness_relationship")}), indent=2))


if __name__ == "__main__":
    main()
