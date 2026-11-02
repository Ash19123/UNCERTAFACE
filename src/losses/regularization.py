"""Mesh-aware regularisers for the fixed-topology face patch.

All three compare the prediction with the TARGET's local geometry instead of
pushing toward absolute smoothness, so fine real structure (nose, lips) is not
smoothed away:
  laplacian_loss   ||L x_pred - L x_target||^2     (L: uniform graph Laplacian)
  edge_loss        (|e_pred| - |e_target|)^2        per mesh edge
  normal_loss      (cos_pred - cos_target)^2        between normals of adjacent faces
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MeshOps(nn.Module):
    """Precomputed topology (edges, uniform Laplacian, adjacent face pairs) of one mesh."""

    def __init__(self, faces: np.ndarray):
        super().__init__()
        faces = np.asarray(faces, dtype=np.int64)
        v = int(faces.max()) + 1
        edge_faces: dict[tuple[int, int], list[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            for e in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault((min(e), max(e)), []).append(fi)
        edges = np.array(sorted(edge_faces), dtype=np.int64)
        pairs = np.array([f for f in edge_faces.values() if len(f) == 2], dtype=np.int64)
        adj = np.zeros((v, v), dtype=np.float32)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj[edges[:, 1], edges[:, 0]] = 1
        deg = adj.sum(1, keepdims=True)
        assert (deg > 0).all(), "mesh has isolated vertices"
        self.register_buffer("faces", torch.from_numpy(faces))
        self.register_buffer("edges", torch.from_numpy(edges))
        self.register_buffer("pairs", torch.from_numpy(pairs))
        self.register_buffer("adj_norm", torch.from_numpy(adj / deg))

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Uniform Laplacian ``x - mean(neighbours)``; ``x`` is (B,V,C)."""
        return x - torch.einsum("uv,bvc->buc", self.adj_norm, x)

    def edge_lengths(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, self.edges[:, 0]] - x[:, self.edges[:, 1]]).norm(dim=-1)

    def adjacent_normal_cos(self, x: torch.Tensor) -> torch.Tensor:
        tri = x[:, self.faces]                                           # (B,F,3,3)
        n = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (n[:, self.pairs[:, 0]] * n[:, self.pairs[:, 1]]).sum(-1)


def laplacian_loss(ops: MeshOps, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((ops.laplacian(pred) - ops.laplacian(target)) ** 2).sum(-1).mean()


def edge_loss(ops: MeshOps, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((ops.edge_lengths(pred) - ops.edge_lengths(target)) ** 2).mean()


def normal_loss(ops: MeshOps, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((ops.adjacent_normal_cos(pred) - ops.adjacent_normal_cos(target)) ** 2).mean()


def logvar_smoothness(ops: MeshOps, logvar: torch.Tensor) -> torch.Tensor:
    """Penalise spatially rough uncertainty: mean squared Laplacian of the log-variance."""
    return (ops.laplacian(logvar) ** 2).mean()
