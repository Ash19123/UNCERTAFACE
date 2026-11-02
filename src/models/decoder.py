"""Template-conditioned surface decoder.

One query per template (face) vertex: Fourier-encoded template position +
learned per-vertex embedding + the global context. Queries cross-attend to the
local skull tokens (and optionally to each other) so each face vertex reads the
skull evidence relevant to it. Because vertex ``v`` is the same anatomical
location in every subject, the learned embedding lets the network keep vertex-
specific behaviour (including vertex-specific uncertainty).
"""
from __future__ import annotations

import torch
from torch import nn

from .feature_aggregation import fourier_features


class DecoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, self_attention: bool):
        super().__init__()
        self.self_attention = self_attention
        self.n_q, self.n_kv, self.n_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        if self_attention:
            self.n_s = nn.LayerNorm(d)
            self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, kv):
        h = self.n_q(q)
        q = q + self.cross(h, self.n_kv(kv), self.n_kv(kv), need_weights=False)[0]
        if self.self_attention:
            h = self.n_s(q)
            q = q + self.self_attn(h, h, h, need_weights=False)[0]
        return q + self.ff(self.n_m(q))


class TemplateDecoder(nn.Module):
    """Maps (tokens, context) to per-template-vertex hidden features ``(B, V, d)``."""

    def __init__(self, template: torch.Tensor, d_model: int, n_layers: int, n_heads: int,
                 n_freq: int, vert_embed_dim: int, self_attention: bool):
        super().__init__()
        self.register_buffer("template", template.float())
        self.n_freq = n_freq
        self.vert_embed = nn.Parameter(torch.randn(len(template), vert_embed_dim) * 0.1)
        self.q_in = nn.Linear(6 * n_freq + vert_embed_dim, d_model)
        self.ctx_in = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, self_attention) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = self.q_in(torch.cat([fourier_features(self.template, self.n_freq), self.vert_embed], dim=-1))
        q = q[None] + self.ctx_in(ctx)[:, None]
        for blk in self.blocks:
            q = blk(q, tokens)
        return self.out_norm(q)
