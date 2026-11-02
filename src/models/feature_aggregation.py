"""Combine local tokens, global context, position and missingness information."""
from __future__ import annotations

import math

import torch
from torch import nn

from .encoder import mlp


def fourier_features(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Positional encoding ``[sin(2^k pi x), cos(2^k pi x)]`` -> ``(..., 3 * 2 * n_freq)``."""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)


class FeatureAggregation(nn.Module):
    """Local tokens (+ Fourier position) and a context vector from global features.

    ``tokens_out = Linear([token_feat, PE(token_xyz)])``
    ``context    = MLP([global_feat, missing_fraction])``
    With ``use_missing_info=False`` the missing fraction is replaced by zero.
    """

    def __init__(self, token_dim: int, global_dim: int, d_model: int, n_freq: int, use_missing_info: bool):
        super().__init__()
        self.n_freq, self.use_missing_info = n_freq, use_missing_info
        self.token_proj = nn.Sequential(nn.Linear(token_dim + 6 * n_freq, d_model), nn.LayerNorm(d_model))
        self.ctx = mlp([global_dim + 1, d_model, d_model])

    def forward(self, tokens, token_xyz, glob, missing_fraction):
        t = self.token_proj(torch.cat([tokens, fourier_features(token_xyz, self.n_freq)], dim=-1))
        m = missing_fraction.view(-1, 1) * (1.0 if self.use_missing_info else 0.0)
        return t, self.ctx(torch.cat([glob, m], dim=-1))
