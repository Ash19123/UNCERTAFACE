"""Uncertainty metrics for diagonal-Gaussian predictions (all in millimetres).

Notation: residual r = y - mu, predicted std sigma, z = r / sigma, per coordinate.

COVERAGE        c(p) = mean over (sample, vertex, axis) of 1[ |r| <= z_p * sigma ],
                z_p = Phi^-1(0.5 + p/2). Pointwise marginal coverage, NOT simultaneous
                coverage of a whole surface.
CALIBRATION ERROR  ECE_cov = mean_p |c(p) - p| over the given nominal levels (plus the
                signed gap c(p) - p per level). 0 is perfect; positive gap = over-coverage.
CRPS (Gaussian) CRPS(mu, sigma; y) = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],
                computed per coordinate and averaged over coordinates, vertices and
                samples (mm). Lower is better; it rewards sharpness AND calibration.
NLL (mm)        mean of 0.5 [ log(2 pi sigma^2) + z^2 ]  with sigma in mm.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm as _norm

LEVELS = (0.5, 0.8, 0.9, 0.95)


def z_for(level: float) -> float:
    return float(_norm.ppf(0.5 + level / 2.0))


def coverage(mean, std, gt, level: float, scale: float | np.ndarray = 1.0) -> float:
    """Empirical per-coordinate coverage of ``mean +/- scale * z_level * std``."""
    return float((np.abs(gt - mean) <= scale * z_for(level) * std).mean())


def coverage_table(mean, std, gt, levels=LEVELS, scale: float | np.ndarray = 1.0) -> dict:
    """Nominal vs observed coverage, signed gap and interval half-width (mm) for each level."""
    out = {}
    for p in levels:
        c = coverage(mean, std, gt, p, scale)
        out[f"{int(round(p * 100))}"] = dict(nominal=p, observed=c, gap=c - p,
                                             mean_half_width_mm=float((scale * z_for(p) * std).mean()))
    out["ece_coverage"] = float(np.mean([abs(out[f"{int(round(p * 100))}"]["gap"]) for p in levels]))
    return out


def reliability_curve(mean, std, gt, n_levels: int = 19):
    """Nominal levels 5%..95% and observed coverage, for a reliability diagram."""
    lv = np.linspace(0.05, 0.95, n_levels)
    return lv, np.array([coverage(mean, std, gt, p) for p in lv])


def crps_gaussian(mean, std, gt) -> np.ndarray:
    """Per-coordinate Gaussian CRPS (same shape as inputs, mm)."""
    z = (gt - mean) / std
    return std * (z * (2 * _norm.cdf(z) - 1) + 2 * _norm.pdf(z) - 1 / math.sqrt(math.pi))


def nll_gaussian(mean, std, gt) -> float:
    z = (gt - mean) / std
    return float((0.5 * (np.log(2 * math.pi * std ** 2) + z ** 2)).mean())


def summary(mean, std, gt, levels=LEVELS, scale: float | np.ndarray = 1.0) -> dict:
    """Coverage table, CRPS, NLL and standardised-residual moments (``scale`` multiplies sigma)."""
    s = std * scale
    z = (gt - mean) / s
    return dict(coverage=coverage_table(mean, std, gt, levels, scale), crps_mm=float(crps_gaussian(mean, s, gt).mean()),
                nll=nll_gaussian(mean, s, gt), z_mean=float(z.mean()), z_std=float(z.std()),
                mean_std_mm=float(s.mean()))
