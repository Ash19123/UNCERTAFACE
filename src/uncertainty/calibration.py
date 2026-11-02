"""Split conformal calibration of predicted Gaussian intervals.

Nonconformity score (per coordinate): ``s = |y - mu| / sigma`` computed on the held-out
CALIBRATION split only. The calibrated interval for nominal level ``p`` is
``mu +/- q_p * sigma`` with ``q_p`` the ``ceil((n+1) p) / n`` empirical quantile of the
scores (the standard split-conformal finite-sample rule; ``q = inf`` if ``ceil((n+1)p) > n``).

SCOPE decides which scores are pooled, i.e. what the guarantee is about:

  global          one ``q`` for every vertex/axis. Guarantee: pointwise MARGINAL coverage
                  averaged over the pooled population. Scores of vertices of one subject
                  are dependent, so the guarantee is approximate in practice.
  per_coordinate  one ``q`` per axis (x, y, z): marginal coverage per axis.
  per_group       one ``q`` per vertex group (e.g. k-means regions of the template):
                  marginal coverage inside each region.
  subject_max     ONE score per BASE GEOMETRY = max over all its vertices, axes and
                  observations. Gives SIMULTANEOUS whole-surface coverage (all vertices of a
                  new subject covered at once) under exchangeability of base geometries.
                  Conservative (the max is taken over the base's observations) and needs
                  >= ceil(1/(1-p)) - 1 calibration bases (otherwise ``q = inf``).

Pointwise coverage must never be reported as simultaneous coverage. The calibrator never
sees test targets: it is fitted from calibration arrays only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCOPES = ("global", "per_coordinate", "per_group", "subject_max")


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile; ``inf`` when ``ceil((n+1) level) > n``."""
    s = np.sort(np.asarray(scores).ravel())
    n = len(s)
    k = math.ceil((n + 1) * level)
    return float("inf") if k > n else float(s[k - 1])


def make_vertex_groups(template: np.ndarray, k: int = 6, seed: int = 0) -> np.ndarray:
    """Deterministic k-means grouping of template vertices (V,) for ``per_group`` calibration."""
    from scipy.cluster.vq import kmeans2
    _, lab = kmeans2(np.asarray(template, dtype=np.float64), k, minit="++", seed=seed)
    return lab.astype(np.int64)


class ConformalCalibrator:
    def __init__(self, scope: str = "global", groups: np.ndarray | None = None,
                 levels=(0.5, 0.8, 0.9, 0.95)):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if scope == "per_group" and groups is None:
            raise ValueError("per_group scope needs vertex groups")
        self.scope, self.groups, self.levels = scope, groups, tuple(levels)
        self.q: dict[float, np.ndarray] = {}
        self.n_scores = 0

    def fit(self, mean: np.ndarray, std: np.ndarray, gt: np.ndarray, base_id: np.ndarray | None = None):
        """Fit on calibration arrays ``(N,V,3)`` (mm). ``base_id (N,)`` needed for ``subject_max``."""
        s = np.abs(gt - mean) / std
        for p in self.levels:
            if self.scope == "global":
                q = np.array(conformal_quantile(s, p))
                self.n_scores = s.size
            elif self.scope == "per_coordinate":
                q = np.array([conformal_quantile(s[..., k], p) for k in range(3)])
                self.n_scores = s[..., 0].size
            elif self.scope == "per_group":
                q = np.array([conformal_quantile(s[:, self.groups == g], p) for g in range(int(self.groups.max()) + 1)])
                self.n_scores = s.shape[0]
            else:
                if base_id is None:
                    raise ValueError("subject_max needs base_id")
                per_obs = s.reshape(len(s), -1).max(1)
                per_base = np.array([per_obs[base_id == b].max() for b in np.unique(base_id)])
                q = np.array(conformal_quantile(per_base, p))
                self.n_scores = len(per_base)
            self.q[p] = q
        return self

    def scale(self, level: float, shape: tuple) -> np.ndarray:
        """Multiplier of ``sigma`` for the half-width at ``level``, broadcastable to ``shape (N,V,3)``."""
        q = self.q[level]
        if self.scope in ("global", "subject_max"):
            return np.asarray(float(q))
        if self.scope == "per_coordinate":
            return q.reshape(1, 1, 3)
        return q[self.groups].reshape(1, -1, 1)

    def interval(self, mean, std, level: float):
        """Calibrated ``(lower, upper)`` bounds in mm."""
        h = self.scale(level, mean.shape) * std
        return mean - h, mean + h

    def to_dict(self) -> dict:
        return dict(scope=self.scope, levels=list(self.levels), n_scores=self.n_scores,
                    groups=None if self.groups is None else self.groups.tolist(),
                    q={str(p): np.asarray(v).tolist() for p, v in self.q.items()})

    @staticmethod
    def from_dict(d: dict) -> "ConformalCalibrator":
        c = ConformalCalibrator(d["scope"], None if d["groups"] is None else np.array(d["groups"]), tuple(d["levels"]))
        c.q = {float(p): np.array(v) for p, v in d["q"].items()}
        c.n_scores = d["n_scores"]
        return c

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path) -> "ConformalCalibrator":
        return ConformalCalibrator.from_dict(json.loads(Path(path).read_text()))


def evaluate_intervals(mean, std, gt, cal: ConformalCalibrator | None, levels=(0.5, 0.8, 0.9, 0.95),
                       groups: np.ndarray | None = None) -> dict:
    """Coverage of (calibrated or raw) intervals on a TEST set.

    Reports, per level: POINTWISE per-coordinate coverage (the usual marginal number),
    per-axis coverage, optional per-group coverage, the SIMULTANEOUS rate (share of test
    samples with every vertex and axis inside its interval) and the mean half-width in mm.
    ``cal=None`` evaluates the raw model intervals (scale 1).
    """
    from ..evaluation.uncertainty_metrics import z_for
    out = {}
    for p in levels:
        scale = np.asarray(1.0) if cal is None else cal.scale(p, mean.shape)
        half = scale * z_for(p) * std if cal is None else scale * std
        inside = np.abs(gt - mean) <= half
        r = dict(nominal=p, pointwise=float(inside.mean()), per_axis=[float(inside[..., k].mean()) for k in range(3)],
                 simultaneous=float(inside.reshape(len(inside), -1).all(1).mean()),
                 mean_half_width_mm=float(np.where(np.isfinite(half), half, np.nan).mean()))
        if groups is not None:
            r["per_group"] = [float(inside[:, groups == g].mean()) for g in range(int(groups.max()) + 1)]
        out[f"{int(round(p * 100))}"] = r
    return out
