"""Reconstruction losses on predicted vs target face vertices (any common length unit)."""
from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all coordinates."""
    return ((pred - target) ** 2).mean()


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all coordinates."""
    return (pred - target).abs().mean()


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric squared Chamfer distance between point sets ``(B,P,3)`` and ``(B,Q,3)``.

    ``mean_p min_q |p-q|^2 + mean_q min_p |p-q|^2``, averaged over the batch.
    Correspondence-free: it ignores vertex identity, so on its own it can let
    vertices slide along the surface.
    """
    d = torch.cdist(pred, target) ** 2
    return (d.min(2).values.mean(1) + d.min(1).values.mean(1)).mean()


RECONSTRUCTION = {"mse": mse_loss, "mae": mae_loss, "chamfer": chamfer_distance}
