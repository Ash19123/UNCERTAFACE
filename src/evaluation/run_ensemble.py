"""Evaluate a deep ensemble built from independently trained runs.

  python -m src.evaluation.run_ensemble --runs experiments/runs/abl_full experiments/runs/abl_full_seed1 \
        experiments/runs/abl_full_seed2 --name ensemble3 [--split test]

Reports single-member vs ensemble reconstruction, and for the three uncertainty kinds kept apart:
aleatoric (mean predicted variance), epistemic (variance of member means) and combined (their sum):
raw 90% pointwise coverage, CRPS, sparsification quality, and the correlation of each with the
actual error. Calibration is fitted on the calibration split with the COMBINED std only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..uncertainty import calibration as cal
from ..uncertainty.ensemble import combine_members
from ..visualization.results import load_run
from . import reconstruction_metrics as rm
from . import sparsification as sp
from . import uncertainty_metrics as um
from .predict import collect
from .run_evaluation import LEVELS, _dataset, _jsonable


def evaluate_ensemble(run_dirs, name="ensemble", eval_split="test", device="cpu", out_root="reports/results") -> dict:
    members_cal, members_test, norm0 = [], [], None
    for rd in run_dirs:
        model, norm, cfg, root = load_run(rd, device)
        members_cal.append(collect(model, norm, _dataset(root, "calibration", norm, cfg), device))
        members_test.append(collect(model, norm, _dataset(root, eval_split, norm, cfg), device))
    if any(m["std"] is None for m in members_test):
        raise ValueError("all ensemble members must be probabilistic")
    faces = np.load(root / "processed/correspondence/correspondence.npz")["face_faces"]
    gt, err_of = members_test[0]["gt"], lambda mean: np.linalg.norm(mean - gt, axis=-1)
    ens, ens_cal = combine_members(members_test), combine_members(members_cal)
    out = dict(name=name, members=[Path(r).name for r in run_dirs], n_members=len(run_dirs), eval_split=eval_split,
               synthetic=True, units="mm")
    out["single_member_rmse_mm"] = {Path(r).name: float(np.sqrt((err_of(m["mean"]) ** 2).mean()))
                                    for r, m in zip(run_dirs, members_test)}
    per = rm.evaluate_batch(ens["mean"], gt, faces)
    out["ensemble_reconstruction"] = rm.summarise(per)
    err = err_of(ens["mean"])
    kinds = dict(aleatoric=ens["std_aleatoric"], epistemic=ens["std_epistemic"], combined=ens["std_total"])
    out["uncertainty"] = {}
    for k, std in kinds.items():
        s = np.maximum(std, 1e-6)
        pos = np.sqrt((std ** 2).sum(-1))
        sc = sp.sparsification_curves(err, pos)
        out["uncertainty"][k] = dict(
            mean_std_mm=float(std.mean()), cov90_raw=um.coverage(ens["mean"], s, gt, 0.9),
            crps_mm=float(um.crps_gaussian(ens["mean"], s, gt).mean()),
            pooled_spearman_err_vs_std=float(__import__("scipy.stats").stats.spearmanr(err.ravel(), pos.ravel())[0]),
            normalised_sparsification_error=sc["normalised_se"], aurc_mm=sc["aurc_mm"])
    out["epistemic_share_of_total_variance"] = float(ens["var_epistemic"].mean() / ens["var_total"].mean())
    c = cal.ConformalCalibrator("global", levels=LEVELS).fit(ens_cal["mean"], ens_cal["std_total"], members_cal[0]["gt"])
    out["combined_calibrated_global"] = cal.evaluate_intervals(ens["mean"], ens["std_total"], gt, c, LEVELS)
    out["combined_raw"] = cal.evaluate_intervals(ens["mean"], ens["std_total"], gt, None, LEVELS)
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(_jsonable(out), indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    r = evaluate_ensemble(a.runs, a.name, a.split)
    print(json.dumps(_jsonable({k: r[k] for k in ("single_member_rmse_mm", "ensemble_reconstruction", "uncertainty",
                                                   "epistemic_share_of_total_variance")}), indent=2)[:3500])


if __name__ == "__main__":
    main()
