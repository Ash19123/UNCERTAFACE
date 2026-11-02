"""Convert model Gaussian outputs (normalised units) to physical millimetres.

For a diagonal Gaussian ``y_k ~ N(mu_k, sigma_k^2)`` per axis ``k``:
  per-axis std (mm)        sigma_mm_k = s * sigma_n_k
  per-axis variance (mm^2) s^2 * sigma_n_k^2
  positional variance      sum_k sigma_mm_k^2     (expected squared 3D error about the mean)
  positional std           sqrt(sum_k sigma_mm_k^2)  (root-sum-square, NOT the std of the 3D distance)
where ``s`` is the Normalizer scale. These are ALEATORIC (heteroscedastic) values
from a single network; epistemic and combined uncertainty need an ensemble.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.preprocessing import Normalizer


def gaussian_to_mm(mean_n, logvar_n, norm: Normalizer) -> dict:
    """Return mean/uncertainty in mm as torch tensors (inputs: tensors, normalised units)."""
    s = norm.scale
    center = torch.as_tensor(norm.center, dtype=mean_n.dtype, device=mean_n.device)
    out = dict(mean_mm=mean_n * s + center)
    if logvar_n is not None:
        var_mm2 = torch.exp(logvar_n) * s ** 2
        out.update(var_mm2=var_mm2, std_mm=var_mm2.sqrt(),
                   pos_var_mm2=var_mm2.sum(-1), pos_std_mm=var_mm2.sum(-1).sqrt())
    return out
