"""Gaussian output head: mean displacement and bounded per-axis log-variance.

    mu_{v,k}      = template_{v,k} + delta_{v,k}
    logvar_{v,k}  = lo + (hi - lo) * sigmoid(raw_{v,k})

``lo`` and ``hi`` are chosen in MILLIMETRES (std bounds -> log mm^2) and
converted once to normalised units with the data scale, so the variance is
always positive, finite and physically bounded (default std in [0.1, 25] mm).
The cost of the sigmoid is that raw values far from 0 saturate with vanishing
gradient; saturation is monitored in training.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class GaussianHead(nn.Module):
    def __init__(self, d_model: int, std_bounds_mm: tuple[float, float], init_std_mm: float,
                 scale_mm: float, probabilistic: bool = True):
        super().__init__()
        self.probabilistic = probabilistic
        self.mean = nn.Linear(d_model, 3)
        nn.init.normal_(self.mean.weight, std=1e-3)
        nn.init.zeros_(self.mean.bias)
        lo_mm, hi_mm = (2 * math.log(s) for s in std_bounds_mm)             # log(mm^2)
        shift = -2 * math.log(scale_mm)                                      # mm^2 -> normalised
        self.lo, self.hi = lo_mm + shift, hi_mm + shift
        if probabilistic:
            self.raw = nn.Linear(d_model, 3)
            nn.init.normal_(self.raw.weight, std=1e-3)
            frac = (2 * math.log(init_std_mm) - lo_mm) / (hi_mm - lo_mm)
            nn.init.constant_(self.raw.bias, math.log(frac / (1 - frac)))   # logit

    def forward(self, h: torch.Tensor, template: torch.Tensor):
        """Returns ``(mean (B,V,3), logvar (B,V,3) or None)`` in normalised units."""
        mean = template[None] + self.mean(h)
        if not self.probabilistic:
            return mean, None
        return mean, self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw(h))
