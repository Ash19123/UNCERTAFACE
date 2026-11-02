"""Stress tests: grouped breakdowns of stored observations and controlled sweeps.

``group_table``     re-uses predictions already collected on a split and breaks the error
                    (and predicted uncertainty) down by severity / pattern / point count / sampler /
                    noise bin. Nothing is regenerated, so factors are confounded exactly as in the data.
``controlled_sweep`` regenerates observations of the SAME base geometries while changing ONE factor
                    (noise level, point density, missingness x pattern, sampler, geometric
                    deformation) and re-predicts, so effects are isolated.

Rows report vertex RMSE (mm), mean predicted positional std (mm, if probabilistic) and the raw
(uncalibrated) 90% pointwise coverage. All values are millimetres on SYNTHETIC data.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from ..data import mesh_utils as mu
from ..data import pointcloud as pc
from .predict import predict_cloud
from .uncertainty_metrics import z_for


def _row(label, pred, gt, std=None) -> dict:
    err = np.linalg.norm(pred - gt, axis=-1)
    r = dict(group=label, n=int(len(pred)), rmse_mm=float(np.sqrt((err ** 2).mean())), mean_error_mm=float(err.mean()))
    if std is not None:
        r["mean_pos_std_mm"] = float(np.sqrt((std ** 2).sum(-1)).mean())
        r["cov90_raw"] = float((np.abs(gt - pred) <= z_for(0.9) * std).mean())
    return r


def group_table(p: dict, key: str, bins: list | None = None) -> list[dict]:
    """Break collected predictions ``p`` (see ``predict.collect``) down by metadata ``key``.

    ``key`` is one of severity, pattern, n_points, sampler, noise_sigma_mm (numeric keys may
    be binned by passing ``bins`` = increasing edges).
    """
    vals = np.asarray(p[key])
    if bins is not None:
        idx = np.digitize(vals, bins[1:-1])
        labels = [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)]
        groups = {labels[i]: np.where(idx == i)[0] for i in range(len(labels))}
    else:
        groups = {str(v): np.where(vals == v)[0] for v in sorted(set(vals.tolist()), key=str)}
    rows = []
    for label, ix in groups.items():
        if len(ix):
            rows.append(_row(label, p["mean"][ix], p["gt"][ix], None if p["std"] is None else p["std"][ix]))
    return rows


def load_geometry(root, base_id: int):
    """Complete skull mesh, region labels and ground-truth face of one base geometry (mm)."""
    root = Path(root)
    v, f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{base_id:04d}.ply"))
    t = np.load(root / f"processed/facial_targets/B{base_id:04d}.npz")
    return v, f, t["skull_region"].astype(np.int64), t["face_verts"]


def _deformation(delta: float, seed: int) -> np.ndarray:
    m = np.random.default_rng(seed).normal(size=(3, 3)) / np.sqrt(3)
    return np.eye(3) + delta * m


def controlled_sweep(model, norm, cfg: dict, root, base_ids, factor: str, values, dataset_cfg: dict,
                     knn_k: int = 8, device="cpu", seed: int = 0, reps: int = 1) -> list[dict]:
    """Vary one ``factor`` over ``values`` on the given bases and re-predict.

    factor: ``noise`` (sigma mm), ``density`` (points in the cloud), ``missingness`` (values are
    ``(severity, pattern)``), ``sampler`` ('area'|'fps') or ``deformation`` (relative size delta of
    a random affine map applied identically to the input skull points and the ground truth, so
    the test probes robustness to out-of-distribution skull geometry/pose, not label noise).
    Other factors are held at moderate severity, 1024 points, 0.3 mm noise.
    """
    obs_cfg, bins = dataset_cfg["observation"], dataset_cfg["missingness_bins"]
    rows = []
    geos = {b: load_geometry(root, b) for b in base_ids}
    patterns = obs_cfg["patterns"]
    for val in values:
        preds, gts, stds = [], [], []
        for bi, b in enumerate(base_ids):
            v, f, region, gt = geos[b]
            for r in range(reps):
                rng = np.random.default_rng(np.random.SeedSequence([seed, b, r, zlib.crc32(factor.encode()) % 100000]))
                kw = dict(severity="moderate", pattern=patterns[bi % len(patterns)], n_points=1024,
                          noise_sigma=0.3, sampler="area")
                if factor == "noise":
                    kw["noise_sigma"] = float(val)
                elif factor == "density":
                    kw["n_points"] = int(val)
                elif factor == "missingness":
                    kw["severity"], kw["pattern"] = val
                elif factor == "sampler":
                    kw["sampler"] = str(val)
                elif factor != "deformation":
                    raise ValueError(f"unknown factor {factor!r}")
                o = pc.simulate_observation(v, f, region, kw["severity"], kw["pattern"], kw["n_points"], rng,
                                            obs_cfg, bins, noise_sigma=kw["noise_sigma"], sampler=kw["sampler"])
                pts, target = o.points, gt
                if factor == "deformation":
                    A = _deformation(float(val), seed + b)
                    pts, target = pts @ A.T, gt @ A.T
                out = predict_cloud(model, norm, pts, o.meta["measured_missing_fraction"], knn_k, device)
                preds.append(out["mean_mm"][0].cpu().numpy())
                gts.append(target)
                if "std_mm" in out:
                    stds.append(out["std_mm"][0].cpu().numpy())
        label = "-".join(map(str, val)) if isinstance(val, tuple) else str(val)
        row = _row(label, np.stack(preds), np.stack(gts), np.stack(stds) if stds else None)
        row["factor"] = factor
        rows.append(row)
    return rows
