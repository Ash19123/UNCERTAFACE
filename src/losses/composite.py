"""Configurable multi-objective loss.

    L = l_rec * L_rec + l_nll * L_NLL + l_surface * L_surface + l_reg * L_reg

  L_rec      mse | mae | chamfer on the predicted mean
  L_NLL      Gaussian NLL (skipped if the model has no variance head)
  L_surface  w_lap * Laplacian + w_edge * edge + w_normal * normal-consistency
  L_reg      log-variance spatial smoothness

Coordinates are rescaled from normalised units to ``unit_mm`` millimetres
(default 10 mm) so the terms have comparable magnitudes. Weights are chosen on
the validation split only.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .probabilistic import gaussian_nll
from .reconstruction import RECONSTRUCTION
from .regularization import MeshOps, edge_loss, laplacian_loss, logvar_smoothness, normal_loss


class CompositeLoss(nn.Module):
    def __init__(self, cfg: dict, mesh: MeshOps, scale_mm: float):
        """``cfg`` is the ``loss`` config section; ``scale_mm`` the normaliser scale."""
        super().__init__()
        self.cfg, self.mesh = cfg, mesh
        if cfg["reconstruction"] not in RECONSTRUCTION:
            raise ValueError(f"unknown reconstruction loss {cfg['reconstruction']!r}")
        self.rec_fn = RECONSTRUCTION[cfg["reconstruction"]]
        self.k = scale_mm / cfg["unit_mm"]                    # normalised -> loss units
        self.logk2 = 2 * math.log(self.k)

    def forward(self, out: dict, face: torch.Tensor, nll_factor: float = 1.0):
        """Return ``(total, terms)``; ``nll_factor`` in [0,1] ramps the NLL weight (warm-up)."""
        c, mean = self.cfg, out["mean"].float() * self.k
        tgt = face.float() * self.k
        terms: dict[str, torch.Tensor] = {}
        total = mean.new_zeros(())
        if c["lambda_rec"] > 0:
            terms["rec"] = self.rec_fn(mean, tgt)
            total = total + c["lambda_rec"] * terms["rec"]
        if out["logvar"] is not None:
            lv = out["logvar"].float() + self.logk2
            terms["nll"] = gaussian_nll(mean, lv, tgt, beta=c["nll_beta"])
            total = total + c["lambda_nll"] * nll_factor * terms["nll"]
            if c["lambda_reg"] > 0:
                terms["reg"] = logvar_smoothness(self.mesh, lv)
                total = total + c["lambda_reg"] * terms["reg"]
        if c["lambda_surface"] > 0:
            w = c["surface_weights"]
            terms["lap"] = laplacian_loss(self.mesh, mean, tgt)
            terms["edge"] = edge_loss(self.mesh, mean, tgt)
            terms["normal"] = normal_loss(self.mesh, mean, tgt)
            surf = w["laplacian"] * terms["lap"] + w["edge"] * terms["edge"] + w["normal"] * terms["normal"]
            total = total + c["lambda_surface"] * surf
        return total, {k: v.detach() for k, v in terms.items()}
