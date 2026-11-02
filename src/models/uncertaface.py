"""UncertaFace: partial skull point cloud -> face-vertex Gaussian (mean, log-variance).

    encoder -> feature aggregation -> template-conditioned decoder -> Gaussian head

All outputs are in NORMALISED coordinates. Convert with
``src.uncertainty.estimation.gaussian_to_mm`` before reporting anything.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .decoder import TemplateDecoder
from .encoder import build_encoder
from .feature_aggregation import FeatureAggregation
from .probabilistic_head import GaussianHead


class UncertaFace(nn.Module):
    def __init__(self, cfg: dict, template: np.ndarray | torch.Tensor, scale_mm: float):
        """``cfg`` is the ``model`` section of model.yaml; ``template`` is the (V,3)
        normalised mean face from the TRAIN split; ``scale_mm`` the normaliser scale."""
        super().__init__()
        self.cfg = cfg
        self.use_missing_info = cfg["use_missing_info"]
        template = torch.as_tensor(template, dtype=torch.float32)
        a, d, h = cfg["aggregation"], cfg["decoder"], cfg["head"]
        self.encoder = build_encoder(cfg["encoder"], c_feat=1)
        self.aggregate = FeatureAggregation(self.encoder.token_dim, self.encoder.global_dim,
                                            a["d_model"], a["n_freq"], self.use_missing_info)
        self.decoder = TemplateDecoder(template, a["d_model"], d["n_layers"], d["n_heads"], a["n_freq"],
                                       d["vert_embed_dim"], d["self_attention"])
        self.head = GaussianHead(a["d_model"], tuple(h["std_bounds_mm"]), h["init_std_mm"], scale_mm,
                                 cfg["probabilistic"])

    def forward(self, points, mask, knn_dist, missing_fraction) -> dict:
        """points (B,N,3), mask (B,N) bool, knn_dist (B,N,1), missing_fraction (B,).

        Returns ``{"mean": (B,V,3), "logvar": (B,V,3) | None}`` (normalised units).
        """
        feat = knn_dist * (1.0 if self.use_missing_info else 0.0)
        enc = self.encoder(points, mask, feat)
        tokens, ctx = self.aggregate(enc.tokens, enc.token_xyz, enc.glob, missing_fraction)
        mean, logvar = self.head(self.decoder(tokens, ctx), self.decoder.template)
        return dict(mean=mean, logvar=logvar)

    def forward_batch(self, batch: dict) -> dict:
        """Convenience wrapper for batches produced by ``UncertafaceDataset``."""
        return self(batch["points"], batch["mask"], batch["knn_dist"], batch["missing_fraction"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
