"""Bootstrap confidence intervals over TEST SKULLS for the headline numbers.

  python scripts/bootstrap_ci.py --run experiments/runs/face_v3_pn

The test split has only a handful of independent skulls, so a point estimate hides how uncertain it is. Observations of the same skull are
not independent, so the resampling unit is the SKULL (base geometry): each bootstrap draw resamples skulls with replacement and averages
their per-skull values. Reports 95% percentile intervals for: model vertex RMSE, mean-face baseline RMSE, their difference and relative gain,
and per-coordinate coverage of the calibrated 90% intervals. With 6 skulls the intervals are wide; that is the honest message.

Writes reports/results/<run>/bootstrap_ci.json. Test targets are used for reporting only, never for fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import dataset as ds
from src.evaluation.predict import collect
from src.uncertainty.calibration import ConformalCalibrator
from src.visualization.results import load_run


def percentile_ci(x: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/face_v3_pn")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, norm, cfg, root = load_run(a.run, a.device)
    d = ds.UncertafaceDataset(root, "test", norm, cfg["data"]["n_points"], cfg["data"]["knn_k"], cache=False)
    o = collect(model, norm, d, a.device)
    name = Path(a.run).name
    q90 = float(ConformalCalibrator.load(Path("reports/results") / name / "calibrator_global.json").scale(0.9, o["mean"].shape))
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    rmse = np.sqrt(((o["mean"] - o["gt"]) ** 2).sum(-1).mean(-1))                   # (N,) per view
    base = np.sqrt(((tpl[None] - o["gt"]) ** 2).sum(-1).mean(-1))
    cov = (np.abs(o["mean"] - o["gt"]) <= q90 * o["std"]).mean((1, 2))             # per-view per-coordinate coverage
    ids = np.unique(o["base_id"])
    per = lambda x: np.array([x[o["base_id"] == b].mean() for b in ids])            # per-skull means
    S = {"model_rmse_mm": per(rmse), "baseline_rmse_mm": per(base), "coverage90_calibrated": per(cov)}
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(ids), size=(a.n_boot, len(ids)))
    out = dict(run=name, n_test_skulls=int(len(ids)), n_test_views=int(len(rmse)), n_boot=a.n_boot, resample_unit="skull", level=0.95)
    for k, v in S.items():
        out[k] = dict(estimate=float(v.mean()), ci95=percentile_ci(v[idx].mean(1)), per_skull=[float(x) for x in v])
    diff = S["baseline_rmse_mm"][idx].mean(1) - S["model_rmse_mm"][idx].mean(1)
    gain = diff / S["baseline_rmse_mm"][idx].mean(1)
    out["baseline_minus_model_mm"] = dict(estimate=float(S["baseline_rmse_mm"].mean() - S["model_rmse_mm"].mean()), ci95=percentile_ci(diff))
    out["relative_gain_over_baseline"] = dict(estimate=float(1 - S["model_rmse_mm"].mean() / S["baseline_rmse_mm"].mean()), ci95=percentile_ci(gain),
                                              share_of_draws_model_better=float((diff > 0).mean()))
    out["note"] = "skulls are resampled, not views; with this few skulls the intervals are wide and should be read as a range, not a precise error bar"
    p = Path("reports/results") / name / "bootstrap_ci.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k not in ("note",)}, indent=2))


if __name__ == "__main__":
    main()
