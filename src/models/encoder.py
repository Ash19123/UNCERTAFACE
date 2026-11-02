"""Point-cloud encoders that tolerate zero-padded, irregular, incomplete clouds.

Every encoder takes ``xyz (B,N,3)``, ``mask (B,N)`` (True = real point) and a
per-point feature ``feat (B,N,F)`` and returns ``EncoderOut``: local ``tokens``
(B,M,C) at ``token_xyz`` (B,M,3), and a ``glob`` vector (B,G). Padded points
are never selected as centroids or neighbours and never enter a pooled max/mean,
so results do not depend on how much padding is present.

  PointNetEncoder        per-point MLP + masked max/mean pooling (baseline)
  PointNetPPEncoder      two set-abstraction levels (FPS centroids, kNN
                         multi-scale grouping) + a global level (PointNet++-style)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncoderOut:
    tokens: torch.Tensor      # (B, M, C)
    token_xyz: torch.Tensor   # (B, M, 3)
    glob: torch.Tensor        # (B, G)


def mlp(sizes: list[int]) -> nn.Sequential:
    """Stack of Linear -> LayerNorm -> GELU."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers)


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x (B,N,C)`` indexed by ``idx (B,...)`` along N -> ``(B,...,C)``."""
    b = torch.arange(x.shape[0], device=x.device).view(-1, *([1] * (idx.dim() - 1)))
    return x[b, idx]


@torch.no_grad()
def masked_fps(xyz: torch.Tensor, mask: torch.Tensor, m: int) -> torch.Tensor:
    """Farthest point sampling over real points only; returns indices ``(B,m)``.

    Starts at the first real point. If fewer than ``m`` real points exist, the
    already-selected real points repeat (padding is never chosen).
    """
    B, N, _ = xyz.shape
    dist = torch.where(mask, torch.full((B, N), 1e10, device=xyz.device), torch.full((B, N), -1.0, device=xyz.device))
    order = torch.arange(N, device=xyz.device)
    cur = (mask.long() * (N - order)).argmax(1)           # smallest index with mask True
    out = torch.empty(B, m, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        out[:, i] = cur
        d = ((xyz - xyz[ar, cur][:, None]) ** 2).sum(-1)
        dist = torch.where(mask, torch.minimum(dist, d), dist)
        cur = dist.argmax(1)
    return out


class SetAbstraction(nn.Module):
    """FPS centroids + multi-scale kNN grouping + shared MLP + max over neighbours."""

    def __init__(self, n_centroids: int, ks: list[int], c_in: int, c_out: list[int], rel_scale: float):
        super().__init__()
        self.m, self.ks, self.rel_scale = n_centroids, ks, rel_scale
        self.mlps = nn.ModuleList([mlp([3 + c_in, c, c]) for c in c_out])
        self.out_dim = sum(c_out)

    def forward(self, xyz, feat, mask):
        idx = masked_fps(xyz, mask, self.m)
        new_xyz = gather(xyz, idx)                                   # (B,M,3)
        d = torch.cdist(new_xyz, xyz).masked_fill(~mask[:, None, :], float("inf"))
        outs = []
        for k, net in zip(self.ks, self.mlps):
            kk = min(k, xyz.shape[1])
            dk, nn_idx = d.topk(kk, dim=-1, largest=False)           # (B,M,kk)
            nn_idx = torch.where(torch.isinf(dk), nn_idx[..., :1].expand_as(nn_idx), nn_idx)  # drop padding
            rel = (gather(xyz, nn_idx) - new_xyz[:, :, None]) * self.rel_scale
            g = torch.cat([rel, gather(feat, nn_idx)], dim=-1)       # (B,M,kk,3+C)
            outs.append(net(g).max(dim=2).values)
        return new_xyz, torch.cat(outs, dim=-1)


class PointNetEncoder(nn.Module):
    """Baseline: per-point MLP, masked max/mean pooling, tokens at FPS points."""
    token_dim, global_dim = 256, 512

    def __init__(self, c_feat: int = 1, m2: int = 64, **_):
        super().__init__()
        self.m2 = m2
        self.net = mlp([3 + c_feat, 64, 128, 256])

    def forward(self, xyz, mask, feat) -> EncoderOut:
        f = self.net(torch.cat([xyz, feat], dim=-1))
        fmax = f.masked_fill(~mask[..., None], float("-inf")).max(1).values
        fmean = (f * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        idx = masked_fps(xyz, mask, self.m2)
        return EncoderOut(gather(f, idx), gather(xyz, idx), torch.cat([fmax, fmean], dim=-1))


class PointNetPPEncoder(nn.Module):
    """Hierarchical multi-scale encoder (PointNet++-style, padding-aware)."""
    token_dim, global_dim = 256, 512

    def __init__(self, c_feat: int = 1, m1: int = 256, m2: int = 64,
                 k1=(16, 32), k2=(16, 32), **_):
        super().__init__()
        self.sa1 = SetAbstraction(m1, list(k1), c_feat, [64, 64], rel_scale=5.0)
        self.sa2 = SetAbstraction(m2, list(k2), self.sa1.out_dim, [128, 128], rel_scale=2.0)
        self.glob_mlp = mlp([self.sa2.out_dim + 3, 256, 256])

    def forward(self, xyz, mask, feat) -> EncoderOut:
        xyz1, f1 = self.sa1(xyz, feat, mask)
        ones = torch.ones(xyz1.shape[:2], dtype=torch.bool, device=xyz.device)
        xyz2, f2 = self.sa2(xyz1, f1, ones)
        g = self.glob_mlp(torch.cat([xyz2, f2], dim=-1))
        return EncoderOut(f2, xyz2, torch.cat([g.max(1).values, g.mean(1)], dim=-1))


def build_encoder(cfg: dict, c_feat: int = 1) -> nn.Module:
    """Create an encoder from the ``model.encoder`` config section."""
    name = cfg["name"]
    if name == "pointnet":
        return PointNetEncoder(c_feat=c_feat, m2=cfg["m2"])
    if name == "pointnetpp":
        return PointNetPPEncoder(c_feat=c_feat, **{k: cfg[k] for k in ("m1", "m2", "k1", "k2")})
    raise ValueError(f"unknown encoder {name!r}")
