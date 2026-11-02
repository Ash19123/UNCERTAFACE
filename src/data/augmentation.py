"""Training-time augmentation for (input points, face target) pairs, in millimetres.

Two families, kept apart to protect skull-face correspondence:

  Joint (applied identically to input AND target): small rotation about the
  origin (~skull centre), translation, isotropic scaling. These are rigid or
  similarity maps, so every input-target geometric relation is preserved up to
  the same map and vertex correspondence of the target is untouched.

  Input-only (target untouched): jitter, random point dropout, density change,
  extra cap occlusion. These degrade the observation, which is what the
  model should be robust to, and do not move the target.

An extra occlusion makes the input *more incomplete*, so the recorded
``missing_fraction`` is updated as ``1 - (1 - m)(1 - q)`` with ``q`` the share
of input points removed. That treats point share as area share, which is
approximate for non-uniformly sampled clouds (documented in the output info).
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about unit ``axis`` by ``angle`` radians."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class Augmenter:
    """Callable ``(points_mm, face_mm, missing_fraction, rng) -> (points, face, info)``."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(self, points: np.ndarray, face: np.ndarray, missing_fraction: float,
                 rng: np.random.Generator):
        c = self.cfg
        points, face = points.astype(np.float64), face.astype(np.float64)
        info: dict = {}

        # ---- joint similarity transform -----------------------------------
        angle = np.deg2rad(rng.uniform(0, c["rotation_deg"]))
        R = rotation_matrix(rng.normal(size=3), angle)
        s = 1.0 + rng.uniform(-c["scale"], c["scale"])
        t = rng.uniform(-c["translation_mm"], c["translation_mm"], size=3)
        points = s * points @ R.T + t
        face = s * face @ R.T + t
        info.update(rotation_deg=float(np.rad2deg(angle)), scale=float(s), translation_mm=t.tolist())

        # ---- input-only degradations --------------------------------------
        if c["jitter_mm"] > 0:
            points = points + rng.normal(scale=c["jitter_mm"], size=points.shape)

        keep = np.ones(len(points), dtype=bool)
        drop = rng.uniform(*c["dropout"])
        keep &= rng.random(len(points)) >= drop
        info["dropout_fraction"] = float(drop)

        dens = rng.uniform(*c["density_keep"])
        keep &= rng.random(len(points)) < dens
        info["density_keep"] = float(dens)

        info["occlusion_removed_fraction"] = 0.0
        if rng.random() < c["occlusion_prob"] and keep.sum() > 16:
            q = rng.uniform(*c["occlusion_fraction"])
            idx = np.where(keep)[0]
            centre = points[rng.choice(idx)]
            order = idx[np.argsort(((points[idx] - centre) ** 2).sum(1))]
            n_rm = int(round(q * len(idx)))
            keep[order[:n_rm]] = False
            info["occlusion_removed_fraction"] = n_rm / len(idx)
            missing_fraction = 1.0 - (1.0 - missing_fraction) * (1.0 - info["occlusion_removed_fraction"])

        if keep.sum() < 16:                      # never return an unusable cloud
            keep[rng.choice(len(points), min(16, len(points)), replace=False)] = True
        info.update(missing_fraction=float(missing_fraction), n_points_after=int(keep.sum()),
                    missing_fraction_note="updated assuming point share ~ area share")
        return points[keep], face, info
