"""Several plausible faces for one broken skull, and which face regions are most certain.

Two products built on the model's calibrated predictive distribution (per vertex, per axis Gaussian, in mm):

1. ``sample_faces``: K faces drawn from N(mean, (c * sigma)^2), where ``c = q90 / 1.645`` rescales the raw sigma by the
   split-conformal factor of the run, so the sampled spread matches the calibrated 90% intervals.
   ASSUMPTION (not learned, not validated): the model predicts independent per-vertex Gaussians, so drawing vertices
   independently would give a rough, implausible surface. We instead draw white noise and smooth it over the face with a
   Gaussian kernel of ``corr_mm``, rescaled so each vertex keeps its calibrated variance. The correlation length is a
   display choice. The samples show the SIZE of the model's uncertainty as different faces; they are not draws from a
   validated joint distribution, and they are synthetic-face samples, not candidate real faces.

2. ``zone_table``: the face split into fixed anatomical-style zones in the reference frame (forehead, eyes, nose, cheeks,
   mouth, chin and jaw), each with its mean calibrated 90% half-width (mm), so the most and least certain zones can be
   listed. Zone borders are rough bands chosen on the synthetic template, not anatomical landmarks.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ZONES = ["forehead", "eyes", "nose", "cheeks", "mouth", "chin and jaw"]
Z_FOREHEAD, Z_EYE, Z_NOSE, Z_MOUTH_TOP, Z_MOUTH_BOT = -28.0, -52.0, -75.0, -75.0, -95.0
Z_X_MID = 16.0
Z90_NOMINAL = 1.6448536269514722


def zone_ids(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Zone index (into ``ZONES``) for face points with lateral ``x`` and vertical ``z`` in the reference frame (mm).
    Every point gets exactly one zone."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    out = np.full(x.shape, ZONES.index("cheeks"), dtype=int)
    out[(z > Z_EYE) & (z <= Z_FOREHEAD) & (np.abs(x) >= Z_X_MID)] = ZONES.index("eyes")
    out[(z > Z_NOSE) & (z <= Z_FOREHEAD) & (np.abs(x) < Z_X_MID)] = ZONES.index("nose")
    out[(z > Z_MOUTH_BOT) & (z <= Z_MOUTH_TOP)] = ZONES.index("mouth")
    out[z <= Z_MOUTH_BOT] = ZONES.index("chin and jaw")
    out[z > Z_FOREHEAD] = ZONES.index("forehead")
    return out


def smoothing_matrix(xz: np.ndarray, corr_mm: float = 12.0) -> csr_matrix:
    """Row-normalised Gaussian smoothing weights over face points ``xz`` (V, 2), in mm. Rows have unit L2 norm, so
    smoothing unit white noise returns noise whose per-vertex variance is exactly 1 (correlated between neighbours)."""
    tree = cKDTree(xz)
    pairs = tree.query_ball_point(xz, 3.0 * corr_mm)
    rows, cols, vals = [], [], []
    for i, nb in enumerate(pairs):
        d2 = ((xz[nb] - xz[i]) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / corr_mm ** 2)
        w /= np.linalg.norm(w)
        rows += [i] * len(nb)
        cols += list(nb)
        vals += list(w)
    return csr_matrix((vals, (rows, cols)), shape=(len(xz), len(xz)))


def calibrated_std(std_mm: np.ndarray, q90: float) -> np.ndarray:
    """Raw per-axis std (mm) rescaled so that +-1.645 std is the conformal-calibrated 90% half-width."""
    return np.asarray(std_mm, float) * (q90 / Z90_NOMINAL)


def sample_faces(mean_mm: np.ndarray, std_mm: np.ndarray, W: csr_matrix, q90: float, k: int = 4, seed: int = 0) -> np.ndarray:
    """K spatially smooth face samples (K, V, 3) in mm. See module docstring for the correlation assumption."""
    rng = np.random.default_rng(seed)
    sd = calibrated_std(std_mm, q90)
    z = rng.standard_normal((k, len(mean_mm), 3))
    return np.stack([mean_mm + sd * (W @ z[i]) for i in range(k)])


def zone_table(zone: np.ndarray, halfwidth_mm: np.ndarray, spread_mm: np.ndarray | None = None) -> list[dict]:
    """Per-zone mean calibrated half-width (mm), most certain first. ``halfwidth_mm`` is per vertex; ``spread_mm``
    (optional) is the per-vertex spread between sampled faces."""
    rows = []
    for zi, name in enumerate(ZONES):
        m = zone == zi
        if not m.any():
            continue
        r = dict(zone=name, n_vertices=int(m.sum()), halfwidth_mm=float(np.mean(halfwidth_mm[m])))
        if spread_mm is not None:
            r["sample_spread_mm"] = float(np.mean(spread_mm[m]))
        rows.append(r)
    return sorted(rows, key=lambda r: r["halfwidth_mm"])
