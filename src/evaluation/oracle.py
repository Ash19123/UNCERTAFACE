"""Irreducible-error floor of the SYNTHETIC task, computed from the generator.

Face = skull surface displaced along its normal by a thickness
``t = f(position, bone-independent latents)``. A predictor that sees the COMPLETE,
noise-free skull knows everything about the bone; what it cannot know are the
face latents (thickness scale/field, nose, lips, cheeks, unobserved field).
Holding the skull fixed and redrawing only those latents therefore gives, per
face vertex, the unavoidable predictive variance of this synthetic world:

    floor_rmse = sqrt( mean_v  sum_k Var_latents[ y_{v,k} | skull ] )
    oracle_pos_std(v) = sqrt( sum_k Var[ y_{v,k} | skull ] )

No real-anatomy claim: this characterises the generator only. Partial inputs
cannot beat this floor.
"""
from __future__ import annotations

import numpy as np

from ..data import generator as gen

FACE_LATENTS = ("thickness_scale", "thickness_field_amp", "nose_projection", "lip_fullness",
                "cheek_fat", "unobserved_field_amp", "thickness_field_seed", "unobserved_field_seed")


def oracle_variance(base_params: list[dict], cfg: dict, n_draws: int = 200, seed: int = 0,
                    subdivisions: int = 4) -> dict:
    """Monte-Carlo per-vertex 3D variance of the face given the skull, over face latents.

    ``base_params``: generator parameter dicts of skulls to hold fixed.
    Returns ``var (S,V,3)`` per skull/vertex/axis and the pooled floor RMSE (mm).
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in base_params:
        faces = []
        for _ in range(n_draws):
            q = dict(p)
            draw = gen.sample_params(rng, cfg)
            for k in FACE_LATENTS:
                q[k] = draw[k]
            faces.append(gen.build_subject(0, q, cfg, subdivisions).face_verts)
        out.append(np.var(np.stack(faces), axis=0))               # (V,3)
    var = np.stack(out)
    return dict(var=var, floor_rmse_mm=float(np.sqrt(var.sum(-1).mean())),
                pos_std_mm=np.sqrt(var.sum(-1)))                  # (S,V)
