"""Deep-ensemble epistemic uncertainty and the law-of-total-variance decomposition.

For M independently trained members with Gaussian outputs ``N(mu_m, sigma_m^2)`` (per axis):
    mean               mu_bar = (1/M) sum_m mu_m
    aleatoric variance (1/M) sum_m sigma_m^2        (average predicted data noise)
    epistemic variance (1/M) sum_m (mu_m - mu_bar)^2  (disagreement between members)
    combined variance  aleatoric + epistemic        (variance of the mixture)
The three are always reported separately; "aleatoric" is never labelled total uncertainty.
MC dropout is deliberately not used. With M = 1 the epistemic term is zero by construction
and is flagged as unavailable. Members are independent runs (different seeds => different
initialisation, data order and augmentation); bootstrap resampling is not implemented.
"""
from __future__ import annotations

import numpy as np


def combine_members(members: list[dict]) -> dict:
    """``members``: list of prediction dicts with ``mean (N,V,3)`` and ``std (N,V,3)`` (mm)."""
    means = np.stack([m["mean"] for m in members])          # (M,N,V,3)
    var_a = np.mean(np.stack([m["std"] for m in members]) ** 2, axis=0)
    mu = means.mean(0)
    var_e = ((means - mu) ** 2).mean(0)
    return dict(mean=mu, n_members=len(members), epistemic_available=len(members) > 1,
                var_aleatoric=var_a, var_epistemic=var_e, var_total=var_a + var_e,
                std_aleatoric=np.sqrt(var_a), std_epistemic=np.sqrt(var_e), std_total=np.sqrt(var_a + var_e))
