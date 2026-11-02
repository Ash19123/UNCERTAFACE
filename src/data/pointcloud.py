"""Incomplete, noisy point-cloud observations of a complete synthetic skull.

An observation is defined by a *visibility function* on the skull surface:
a scalar score ``s(p)`` and threshold ``tau``; a surface point is REMOVED iff
``s(p) > tau``. Using a function on the continuous surface (rather than a
vertex subset) means any number of points can be sampled from the visible part.

``tau`` is chosen as a quantile of ``s`` on a dense area-uniform cloud so the
removed area fraction hits a random target inside the requested severity bin.
The reported ``measured_missing_fraction`` is then computed on a SECOND,
independent dense cloud (Monte-Carlo estimate with standard error), and the
target is redrawn if that measurement falls outside the bin. The fraction is
therefore measured on generated geometry, not assumed.

Patterns (all contiguous or patchy, defined on the direction ``d = p/|p|``;
valid because the synthetic skulls are star-shaped about the origin):
  patchy   random Voronoi cells on the sphere are removed
  cap      a spherical cap around a random centre is removed
  upper    only the upper cranium (tilted around +z) is observed
  lateral  only one side (tilted around +/-x) is observed
  region   whole anatomical regions removed in random order, last one partial
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from . import mesh_utils as mu

PATTERNS = ("patchy", "cap", "upper", "lateral", "region")


def _dirs(p: np.ndarray) -> np.ndarray:
    return p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _angle(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(d @ c, -1.0, 1.0))


def make_score(pattern: str, rng: np.random.Generator, skull_verts: np.ndarray,
               region: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``score(points) -> (n,)``; higher score = removed first."""
    if pattern == "patchy":
        k = int(rng.integers(12, 41))
        seeds = _dirs(rng.normal(size=(k, 3)))
        rank = rng.permutation(k).astype(np.float64)

        def score(p):
            sim = _dirs(p) @ seeds.T
            cell = sim.argmax(1)
            ang = np.arccos(np.clip(sim.max(1), -1, 1))
            return rank[cell] + 0.999 * (1.0 - ang / np.pi)
    elif pattern == "cap":
        c = _unit(rng.normal(size=3))
        score = lambda p: -_angle(_dirs(p), c)
    elif pattern in ("upper", "lateral"):
        base = np.array([0.0, 0, 1]) if pattern == "upper" else np.array([rng.choice([-1.0, 1.0]), 0, 0])
        a = _unit(base + 0.3 * rng.normal(size=3))
        score = lambda p: -(_dirs(p) @ a)
    elif pattern == "region":
        n_reg = int(region.max()) + 1
        rank = rng.permutation(n_reg).astype(np.float64)
        d = _dirs(skull_verts)
        cent = np.stack([_unit(d[region == r].mean(0)) if (region == r).any() else np.array([0.0, 0, 1])
                         for r in range(n_reg)])
        tree = cKDTree(skull_verts)

        def score(p):
            reg = region[tree.query(p)[1]]
            ang = np.arccos(np.clip((_dirs(p) * cent[reg]).sum(1), -1, 1))
            return rank[reg] + 0.999 * (1.0 - ang / np.pi)
    else:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
    return score


@dataclass
class Observation:
    """A simulated partial observation of one skull."""
    points: np.ndarray            # (n, 3) noisy observed points, mm
    points_clean: np.ndarray      # (n, 3) same points before noise/outliers
    vertex_visible: np.ndarray    # (V,) bool, skull vertices inside the observed region
    meta: dict                    # JSON-serialisable description (see simulate_observation)
    visible_fn: Callable = field(repr=False, default=None)  # points -> bool, True = observed


def _sample_visible(verts, faces, visible_fn, n, rng, kappa, axis, sampler):
    """Sample ``n`` visible points: area-uniform, tilted by ``kappa``, optionally FPS."""
    need = n * (4 if sampler == "fps" else 1)
    out, got, tries = [], 0, 0
    while got < need:
        cand = mu.sample_surface(verts, faces, max(4 * need, 4096), rng)
        keep = visible_fn(cand)
        if kappa > 0:
            keep &= rng.random(len(cand)) < np.exp(kappa * (_dirs(cand) @ axis - 1.0))
        out.append(cand[keep])
        got += int(keep.sum())
        tries += 1
        if tries > 200:
            raise RuntimeError("visible region too small to sample from")
    pts = np.concatenate(out)[:need]
    if sampler == "fps":
        pts = pts[mu.farthest_point_sampling(pts, n)]
    return pts


def simulate_observation(skull_verts: np.ndarray, skull_faces: np.ndarray, region: np.ndarray,
                         severity: str, pattern: str, n_points: int, rng: np.random.Generator,
                         cfg_obs: dict, bins: dict, noise_sigma: float = 0.0,
                         outlier_fraction: float = 0.0, density_kappa: float = 0.0,
                         sampler: str = "area") -> Observation:
    """Create one incomplete observation of a complete skull.

    ``bins`` maps severity name -> [low, high] of the *measured* missing area
    fraction. ``meta`` records pattern, severity, target and measured fraction
    (with its Monte-Carlo standard error), noise parameters, density tilt,
    sampler and point count. Coordinates and ``noise_sigma`` are in mm.
    """
    lo, hi = bins[severity]
    m = int(cfg_obs["dense_reference_points"])
    score = make_score(pattern, rng, skull_verts, region)
    dense_fit = mu.sample_surface(skull_verts, skull_faces, m, rng)
    s_fit = score(dense_fit)
    for attempt in range(1, int(cfg_obs["max_tries"]) + 1):
        target = float(rng.uniform(lo, hi))
        tau = float(np.quantile(s_fit, 1.0 - target))
        dense_eval = mu.sample_surface(skull_verts, skull_faces, m, rng)   # independent estimate
        removed = score(dense_eval) > tau
        measured = float(removed.mean())
        if lo <= measured <= hi:
            break
    else:
        raise RuntimeError(f"could not hit bin {severity}={lo, hi} with pattern {pattern}")
    se = float(np.sqrt(measured * (1 - measured) / m))

    visible_fn = lambda p, _s=score, _t=tau: _s(p) <= _t
    axis = _unit(rng.normal(size=3))
    clean = _sample_visible(skull_verts, skull_faces, visible_fn, n_points, rng, density_kappa, axis, sampler)

    noisy = clean + rng.normal(scale=noise_sigma, size=clean.shape) if noise_sigma > 0 else clean.copy()
    n_out = int(round(outlier_fraction * len(clean)))
    if n_out:
        idx = rng.choice(len(clean), n_out, replace=False)
        noisy[idx] += rng.normal(scale=cfg_obs["outlier_scale_mm"], size=(n_out, 3))

    meta = dict(pattern=pattern, severity=severity, target_missing_fraction=target,
                measured_missing_fraction=measured, measured_fraction_mc_se=se, tries=attempt,
                n_points=int(n_points), sampler=sampler, noise_sigma_mm=float(noise_sigma),
                outlier_fraction=float(outlier_fraction), outlier_scale_mm=float(cfg_obs["outlier_scale_mm"]),
                density_kappa=float(density_kappa), density_axis=axis.tolist(), units="mm")
    return Observation(noisy, clean, visible_fn(skull_verts), meta, visible_fn)


def random_observation(skull_verts, skull_faces, region, rng, cfg_obs, bins, point_counts,
                       severity: str | None = None) -> Observation:
    """Draw all observation settings at random from the config, then simulate."""
    sev = severity or str(rng.choice(list(bins)))
    return simulate_observation(
        skull_verts, skull_faces, region, sev, str(rng.choice(cfg_obs["patterns"])),
        int(rng.choice(point_counts)), rng, cfg_obs, bins,
        noise_sigma=float(rng.uniform(*cfg_obs["noise_sigma_mm"])),
        outlier_fraction=float(rng.uniform(*cfg_obs["outlier_fraction"])),
        density_kappa=float(rng.uniform(*cfg_obs["density_kappa"])),
        sampler=str(rng.choice(cfg_obs["samplers"])))
