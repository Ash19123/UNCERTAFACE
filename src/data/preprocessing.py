"""Normalisation, sample loading, fixed-size batching and neighbourhoods.

Normalisation is ONE isotropic similarity transform shared by inputs, targets
and predictions:

    x_n = (x_mm - c) / s            (c in R^3, s a positive scalar in mm)

fitted on the training base geometries only. Because ``s`` is a scalar, a
diagonal Gaussian stays diagonal and physical units are recovered exactly:

    mean:      mu_mm     = s * mu_n + c
    std:       sigma_mm  = s * sigma_n
    variance:  var_mm2   = s**2 * var_n
    log-var:   logvar_mm = logvar_n + 2 * log(s)

Anisotropic scaling is deliberately NOT used: it would make per-axis variances
non-interchangeable. Uncertainty must be converted with these methods before
it is reported; normalised values are never millimetres.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import mesh_utils as mu
from .build_dataset import SPLIT_NAMES, base_name


@dataclass(frozen=True)
class Normalizer:
    """Isotropic similarity transform ``x_n = (x_mm - center) / scale``."""
    center: tuple[float, float, float]
    scale: float
    fitted_on: str = ""

    def __post_init__(self):
        if not (np.isfinite(self.scale) and self.scale > 0):
            raise ValueError("scale must be positive and finite")

    def forward(self, x_mm):
        """Millimetres -> normalised coordinates."""
        return (np.asarray(x_mm) - np.asarray(self.center)) / self.scale

    def inverse(self, x_n):
        """Normalised coordinates -> millimetres (positions, i.e. means)."""
        return np.asarray(x_n) * self.scale + np.asarray(self.center)

    def std_to_mm(self, sigma_n):
        """Standard deviation (or any length offset, e.g. interval half-width) -> mm."""
        return np.asarray(sigma_n) * self.scale

    def var_to_mm2(self, var_n):
        """Variance -> mm^2."""
        return np.asarray(var_n) * self.scale ** 2

    def logvar_to_mm(self, logvar_n):
        """Log-variance in normalised units -> log(mm^2)."""
        return np.asarray(logvar_n) + 2.0 * np.log(self.scale)

    def logvar_from_mm(self, logvar_mm2):
        """Log(mm^2) -> normalised log-variance (e.g. to set bounds in mm)."""
        return np.asarray(logvar_mm2) - 2.0 * np.log(self.scale)

    def to_dict(self) -> dict:
        return dict(type="isotropic_similarity", center=list(self.center), scale=self.scale,
                    units_of_scale="mm", fitted_on=self.fitted_on)

    @staticmethod
    def from_dict(d: dict) -> "Normalizer":
        return Normalizer(tuple(d["center"]), float(d["scale"]), d.get("fitted_on", ""))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path) -> "Normalizer":
        return Normalizer.from_dict(json.loads(Path(path).read_text()))


def read_split(root, split: str) -> dict:
    """Read ``splits/<split>.json`` -> ``{'base_ids': [...], 'sample_ids': [...]}``."""
    return json.loads((Path(root) / "splits" / f"{split}.json").read_text())


def fit_normalizer(root, base_ids) -> Normalizer:
    """Fit centre and scale from the complete skulls of ``base_ids`` (use TRAIN only).

    centre = mean vertex position; scale = RMS distance of vertices to centre.
    """
    root = Path(root)
    verts = np.concatenate([mu.load_mesh(str(root / f"raw/generated/cranial_meshes/{base_name(b)}.ply"))[0]
                            for b in base_ids])
    c = verts.mean(0)
    s = float(np.sqrt(((verts - c) ** 2).sum(1).mean()))
    return Normalizer(tuple(float(v) for v in c), s, fitted_on=f"{len(list(base_ids))} base geometries")


def fit_normalizer_from_train(root) -> Normalizer:
    """Convenience: fit on the train split recorded on disk."""
    return fit_normalizer(root, read_split(root, "train")["base_ids"])


def load_sample(root, record: dict, norm: Normalizer | None = None) -> dict:
    """Load one observation and its target (float32).

    Returns raw-mm arrays, plus normalised copies when ``norm`` is given:
    ``points``, ``face``, ``vertex_visible``, ``missing_fraction``, and
    ``face_vert_index`` (skull vertex index of each face vertex).
    """
    root = Path(root)
    pc = np.load(root / record["pointcloud"])
    face = np.load(root / record["face_target"])["face_verts"]
    corr = np.load(root / record["correspondence"])
    out = dict(sample_id=record["sample_id"], points_mm=pc["points"].astype(np.float32),
               face_mm=face.astype(np.float32), vertex_visible=pc["vertex_visible"],
               face_vert_index=corr["face_vert_index"],
               missing_fraction=float(record["observation"]["measured_missing_fraction"]))
    if norm is not None:
        out["points"] = norm.forward(out["points_mm"]).astype(np.float32)
        out["face"] = norm.forward(out["face_mm"]).astype(np.float32)
    return out


def to_fixed_size(points: np.ndarray, n: int, rng: np.random.Generator | None = None):
    """Make a cloud exactly ``n`` points: random subsample if larger, zero-pad if smaller.

    Returns ``(points (n,3), mask (n,) bool)`` where ``mask`` is True for real
    points. Padded rows are exactly zero and must be ignored via the mask.
    """
    rng = rng or np.random.default_rng(0)
    m = len(points)
    if m >= n:
        idx = rng.choice(m, n, replace=False) if m > n else np.arange(m)
        return points[idx], np.ones(n, dtype=bool)
    out = np.zeros((n, points.shape[1]), dtype=points.dtype)
    out[:m] = points
    mask = np.zeros(n, dtype=bool)
    mask[:m] = True
    return out, mask


def knn_indices(points: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices (n, k) of the ``k`` nearest REAL neighbours of each point (self included).

    If ``mask`` is given, padded points are excluded from the search and their
    own rows point to themselves. Requires at least ``k`` real points.
    """
    real = np.arange(len(points)) if mask is None else np.where(mask)[0]
    if len(real) < k:
        raise ValueError(f"need at least {k} real points, got {len(real)}")
    _, nn = cKDTree(points[real]).query(points[real], k=k)
    out = np.tile(np.arange(len(points))[:, None], (1, k))
    out[real] = real[nn]
    return out
