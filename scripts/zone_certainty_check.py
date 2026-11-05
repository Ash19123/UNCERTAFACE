"""Do the zone ranking and the sampled faces mean anything? Evaluated on the TEST split only (never used for fitting).

  python scripts/zone_certainty_check.py --run experiments/runs/face_v3_pn

Writes reports/results/<run>/zone_certainty.json:
  - per zone: mean predicted calibrated 90% half-width vs mean actual error (mm), and the Spearman rank agreement of the
    two zone rankings (is "most certain zone" really the most accurate zone?);
  - per vertex: Spearman correlation of half-width with error; error in the 25% most vs 25% least certain vertices;
  - sampled faces (K smooth samples): mean sample RMSE, best-of-K RMSE (oracle-selected, NOT a single-prediction result),
    and the mean-prediction RMSE for reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.data import dataset as ds
from src.evaluation.predict import collect
from src.uncertainty import plausible_faces as pf
from src.uncertainty.calibration import ConformalCalibrator
from src.visualization.results import load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/face_v3_pn")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--corr-mm", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, norm, cfg, root = load_run(a.run, a.device)
    d = ds.UncertafaceDataset(root, "test", norm, cfg["data"]["n_points"], cfg["data"]["knn_k"], cache=False)
    o = collect(model, norm, d, a.device)
    name = Path(a.run).name
    q90 = float(ConformalCalibrator.load(Path("reports/results") / name / "calibrator_global.json").scale(0.9, o["mean"].shape))
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    zone = pf.zone_ids(tpl[:, 0], tpl[:, 2])
    W = pf.smoothing_matrix(tpl[:, [0, 2]], a.corr_mm)
    err = np.linalg.norm(o["mean"] - o["gt"], axis=-1)                     # (N, V) mm
    hw = q90 * np.sqrt((o["std"] ** 2).mean(-1))                            # (N, V) calibrated PER-COORDINATE half-width (RMS over axes), mm
    res = dict(run=name, n_test=int(len(err)), q90=q90, corr_mm=a.corr_mm, k=a.k, split="test")
    rows = []
    for zi, zn in enumerate(pf.ZONES):
        m = zone == zi
        if m.any():
            rows.append(dict(zone=zn, n_vertices=int(m.sum()), mean_halfwidth_mm=float(hw[:, m].mean()), mean_error_mm=float(err[:, m].mean())))
    res["zones"] = sorted(rows, key=lambda r: r["mean_halfwidth_mm"])
    res["zone_rank_spearman_halfwidth_vs_error"] = float(spearmanr([r["mean_halfwidth_mm"] for r in rows], [r["mean_error_mm"] for r in rows])[0])
    res["vertex_spearman_halfwidth_vs_error"] = float(spearmanr(hw.ravel(), err.ravel())[0])
    lo, hi = np.quantile(hw, [0.25, 0.75])
    res["error_in_25pct_most_certain_mm"] = float(err[hw <= lo].mean())
    res["error_in_25pct_least_certain_mm"] = float(err[hw >= hi].mean())
    res["error_all_mm"] = float(err.mean())
    ax_err = np.abs(o["mean"] - o["gt"])                                      # (N, V, 3)
    res["coverage_per_coordinate_inside_halfwidth"] = float((ax_err <= hw[..., None]).mean())
    res["coverage_note"] = "share of true face coordinates (x, y, z) within +- the displayed half-width; nominal 0.90; fitted on the calibration split only"
    rm = lambda p: float(np.sqrt(((p - o["gt"]) ** 2).sum(-1).mean(-1)).mean())
    rmse_mean = rm(o["mean"])
    S = [pf.sample_faces(o["mean"][i], o["std"][i], W, q90, a.k, seed=i) for i in range(len(err))]
    per = np.array([[np.sqrt(((s[j] - o["gt"][i]) ** 2).sum(-1).mean()) for j in range(a.k)] for i, s in enumerate(S)])
    res["samples"] = dict(mean_prediction_rmse_mm=rmse_mean, mean_sample_rmse_mm=float(per.mean()), best_of_k_rmse_mm=float(per.min(1).mean()),
                          note="best-of-K is oracle-selected using the true face; NOT a single-prediction result")
    out = Path("reports/results") / name / "zone_certainty.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
