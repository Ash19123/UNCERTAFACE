"""Gaussian negative log-likelihood (coordinate-wise, diagonal covariance)."""
from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                 beta: float = 0.0, reduce: bool = True) -> torch.Tensor:
    """``0.5 * [logvar + (y - mu)^2 / exp(logvar)]`` per coordinate (constants dropped).

    The likelihood factorises over the 3 axes of every vertex (diagonal
    covariance); the loss is the mean over batch, vertices and axes. With
    ``beta > 0`` each term is weighted by ``stopgrad(var)^beta`` (beta-NLL), which
    stops high-variance points from starving the mean of gradient; ``beta = 0``
    is the plain NLL.

    The gradient of the NLL is the same whatever length unit ``mean/logvar/target``
    use (a unit change only adds a constant to the value), provided ``logvar`` is
    expressed in the same unit squared.
    """
    nll = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    if beta > 0:
        nll = nll * torch.exp(logvar).detach() ** beta
    return nll.mean() if reduce else nll
