"""Sparsification and risk-coverage analysis.

Units are individual vertex locations (pooled over samples). ``err`` is the 3D
vertex error (mm) and ``unc`` the predicted positional std (mm).

SPARSIFICATION  For removed fraction f in [0, 1): drop the f most uncertain locations,
                report the mean error of those retained. Three rankings:
                  uncertainty : by predicted ``unc`` (descending removed first)
                  oracle      : by the true ``err`` (the best possible ranking)
                  random      : by chance (expected value = overall mean error, flat line)
                Sparsification error  SE = mean_f [ curve_unc(f) - curve_oracle(f) ]
                (area between the uncertainty and oracle curves, mm; 0 = perfect ranking).
                Normalised SE = SE / mean_f [ random - oracle ] (1 = no better than random).

RISK-COVERAGE   Same curve read the other way: coverage = retained fraction (1 - f),
                risk = mean retained error. AURC = mean over coverage of risk (mm; lower is better).
"""
from __future__ import annotations

import numpy as np


def sparsification_curves(err: np.ndarray, unc: np.ndarray, n_points: int = 100) -> dict:
    """Curves of retained-mean error vs removed fraction for uncertainty / oracle / random rankings."""
    err, unc = np.asarray(err).ravel(), np.asarray(unc).ravel()
    n = len(err)
    fr = np.linspace(0.0, 0.99, n_points)

    def curve(order):                       # order: indices, most trustworthy first
        cs = np.cumsum(err[order])
        keep = np.maximum((np.round((1 - fr) * n)).astype(int), 1)
        return cs[keep - 1] / keep

    by_unc = curve(np.argsort(unc, kind="stable"))
    oracle = curve(np.argsort(err, kind="stable"))
    random = np.full(n_points, err.mean())
    se = float(np.mean(by_unc - oracle))
    denom = float(np.mean(random - oracle))
    return dict(removed_fraction=fr, uncertainty=by_unc, oracle=oracle, random=random,
                sparsification_error_mm=se, normalised_se=se / denom if denom > 0 else float("nan"),
                aurc_mm=float(np.mean(by_unc)), aurc_oracle_mm=float(np.mean(oracle)),
                aurc_random_mm=float(err.mean()))


def risk_coverage(err: np.ndarray, unc: np.ndarray, coverages=(1.0, 0.9, 0.75, 0.5, 0.25, 0.1)) -> dict:
    """Mean error (risk) at selected retained fractions, ranking by predicted uncertainty."""
    err, unc = np.asarray(err).ravel(), np.asarray(unc).ravel()
    order = np.argsort(unc, kind="stable")
    cs = np.cumsum(err[order])
    out = {}
    for c in coverages:
        k = max(int(round(c * len(err))), 1)
        out[f"{c:g}"] = float(cs[k - 1] / k)
    return out


def best_of_n(mean, std, gt, n_samples: int = 10, seed: int = 0) -> dict:
    """Best-of-N over candidate faces sampled from the predicted diagonal Gaussian.

    For each test sample draw ``n_samples`` surfaces ``mean + std * eps`` and keep the
    one with the lowest vertex RMSE. Reported SEPARATELY from the single-prediction
    result: it uses the ground truth to choose, so it is an optimistic oracle-selected
    number and says nothing about a deployable single reconstruction.
    """
    rng = np.random.default_rng(seed)
    rmse = lambda x: np.sqrt(((x - gt) ** 2).sum(-1).mean(-1))           # (N,)
    single = rmse(mean)
    draws = np.stack([rmse(mean + std * rng.normal(size=mean.shape)) for _ in range(n_samples)])  # (n,N)
    return dict(n_samples=n_samples, single_prediction_rmse_mm=float(np.sqrt((single ** 2).mean())),
                best_of_n_rmse_mm=float(np.sqrt((draws.min(0) ** 2).mean())),
                mean_sampled_rmse_mm=float(np.sqrt((draws ** 2).mean())))
