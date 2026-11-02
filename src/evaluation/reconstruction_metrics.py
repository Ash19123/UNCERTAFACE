"""Reconstruction metrics in millimetres.

Three families are kept distinct because they answer different questions:

  VERTEX-WISE (uses the known correspondence): ``vertex_mae``, ``vertex_rmse``.
  POINT-SET (ignores correspondence, treats vertices as points): ``chamfer``, ``hausdorff``.
  SURFACE-LEVEL (distance to the other mesh's triangles): ``point_to_surface`` and the
  symmetric ``surface_error``.

Inputs are ``pred (V,3)`` and ``gt (V,3)`` in mm and the shared face triangulation ``faces (F,3)``.
Batched helpers average per-sample values.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def vertex_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Euclidean error per corresponding vertex, ``(..., V)``."""
    return np.linalg.norm(pred - gt, axis=-1)


def vertex_mae(pred, gt) -> float:
    """Mean absolute error over all coordinates (mm)."""
    return float(np.abs(pred - gt).mean())


def vertex_rmse(pred, gt) -> float:
    """sqrt(mean squared 3D vertex distance) (mm)."""
    return float(np.sqrt((vertex_errors(pred, gt) ** 2).mean()))


def chamfer(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Chamfer distance (mean of the two mean nearest-neighbour distances), mm."""
    d1 = cKDTree(gt).query(pred)[0]
    d2 = cKDTree(pred).query(gt)[0]
    return float(0.5 * (d1.mean() + d2.mean()))


def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff distance between the vertex sets (max nearest-neighbour distance), mm."""
    return float(max(cKDTree(gt).query(pred)[0].max(), cKDTree(pred).query(gt)[0].max()))


def _point_triangle_sq_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared distance from points ``p (M,3)`` to triangles ``(a,b,c)`` each ``(M,3)``.

    Ericson, Real-Time Collision Detection, closest point on a triangle (vectorised).
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    closest = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def put(mask, pt):
        nonlocal done
        m = mask & ~done
        closest[m] = pt[m]
        done |= m

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    put((d6 >= 0) & (d5 <= d6), c)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + u[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        vv, ww = vb * denom, vc * denom
    put(np.ones(len(p), dtype=bool), a + vv[:, None] * ab + ww[:, None] * ac)
    return ((p - closest) ** 2).sum(1)


def point_to_surface(points: np.ndarray, verts: np.ndarray, faces: np.ndarray, k: int = 12) -> np.ndarray:
    """Distance from each point to the surface of the mesh ``(verts, faces)``, mm.

    Candidate triangles are the ``k`` with nearest centroids (KD-tree); the exact
    point-triangle distance is minimised over them. With k nearest centroids on
    these smooth, evenly tessellated patches this equals the true minimum
    (verified against a brute-force search in the tests).
    """
    tri = verts[faces]                                              # (F,3,3)
    _, idx = cKDTree(tri.mean(1)).query(points, k=min(k, len(faces)))
    idx = idx.reshape(len(points), -1)
    best = np.full(len(points), np.inf)
    for j in range(idx.shape[1]):
        t = tri[idx[:, j]]
        best = np.minimum(best, _point_triangle_sq_dist(points, t[:, 0], t[:, 1], t[:, 2]))
    return np.sqrt(best)


def surface_error(pred: np.ndarray, gt: np.ndarray, faces: np.ndarray) -> float:
    """Symmetric mean point-to-surface distance: 0.5 * (pred->gt surface + gt->pred surface), mm."""
    return float(0.5 * (point_to_surface(pred, gt, faces).mean() + point_to_surface(gt, pred, faces).mean()))


def evaluate_batch(pred: np.ndarray, gt: np.ndarray, faces: np.ndarray) -> dict:
    """Per-sample metrics for ``pred, gt (N,V,3)``; returns arrays of length N (all mm)."""
    n = len(pred)
    cols = {k: np.empty(n) for k in ("vertex_mae", "vertex_rmse", "chamfer", "hausdorff", "point_to_surface", "surface_error")}
    for i in range(n):
        cols["vertex_mae"][i] = vertex_mae(pred[i], gt[i])
        cols["vertex_rmse"][i] = vertex_rmse(pred[i], gt[i])
        cols["chamfer"][i] = chamfer(pred[i], gt[i])
        cols["hausdorff"][i] = hausdorff(pred[i], gt[i])
        cols["point_to_surface"][i] = point_to_surface(pred[i], gt[i], faces).mean()
        cols["surface_error"][i] = surface_error(pred[i], gt[i], faces)
    return cols


def summarise(per_sample: dict) -> dict:
    """Mean and standard deviation across samples of each per-sample metric."""
    return {k: dict(mean=float(v.mean()), std=float(v.std())) for k, v in per_sample.items()}
