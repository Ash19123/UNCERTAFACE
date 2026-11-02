import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.preprocessing import Normalizer
from src.models import encoder as enc
from src.models.uncertaface import UncertaFace
from src.uncertainty.estimation import gaussian_to_mm

ROOT = Path(__file__).parents[1]
MCFG = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())["model"]
NORM = Normalizer((0.0, 0.2, -0.1), 76.5)
V = 570


def template(v=V, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(v, 3)).astype(np.float32) * 0.5)


def batch(b=2, n=1024, real=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(b, n, 3, generator=g) * 0.6
    mask = torch.ones(b, n, dtype=torch.bool)
    if real is not None:
        mask[:, real:] = False
        pts[:, real:] = 0
    return pts, mask, torch.rand(b, n, 1, generator=g) * 0.1 * mask[..., None], torch.rand(b, generator=g) * 0.8


def model(**over):
    cfg = copy.deepcopy(MCFG)
    for k, v in over.items():
        if k == "encoder":
            cfg["encoder"]["name"] = v
        else:
            cfg[k] = v
    torch.manual_seed(0)
    return UncertaFace(cfg, template(), NORM.scale)


def test_masked_fps_never_picks_padding_and_is_distinct():
    pts, mask, _, _ = batch(2, 200, real=120)
    idx = enc.masked_fps(pts, mask, 64)
    assert (idx < 120).all()
    assert all(len(set(r.tolist())) == 64 for r in idx)
    idx2 = enc.masked_fps(pts, mask, 64)
    assert torch.equal(idx, idx2)
    few = enc.masked_fps(pts[:, :200], torch.cat([mask[:, :5], torch.zeros(2, 195, dtype=torch.bool)], 1), 20)
    assert (few < 5).all()                                    # fewer real points than centroids: repeats, no padding


@pytest.mark.parametrize("name", ["pointnet", "pointnetpp"])
def test_encoder_shapes_and_padding_invariance(name):
    e = enc.build_encoder(dict(MCFG["encoder"], name=name), c_feat=1).eval()
    pts, mask, f, _ = batch(2, 300, real=300)
    with torch.no_grad():
        a = e(pts[:, :100], mask[:, :100], f[:, :100])
        # same 100 real points followed by 200 rows of GARBAGE padding
        p2, m2, f2 = pts.clone(), mask.clone(), f.clone()
        p2[:, :100], f2[:, :100] = pts[:, :100], f[:, :100]
        m2[:, 100:] = False
        b = e(p2, m2, f2)
    assert a.tokens.shape == (2, 64, 256) and a.token_xyz.shape == (2, 64, 3) and a.glob.shape == (2, 512)
    assert torch.allclose(a.tokens, b.tokens, atol=1e-5) and torch.allclose(a.glob, b.glob, atol=1e-5)


def test_encoder_unknown_name():
    with pytest.raises(ValueError):
        enc.build_encoder(dict(MCFG["encoder"], name="nope"))


@pytest.mark.parametrize("name", ["pointnet", "pointnetpp"])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
def test_forward_shapes_finite_and_positive_variance(name, n):
    m = model(encoder=name).eval()
    with torch.no_grad():
        out = m(*batch(2, n, real=n - 100))
    assert out["mean"].shape == (2, V, 3) and out["logvar"].shape == (2, V, 3)
    assert torch.isfinite(out["mean"]).all() and torch.isfinite(out["logvar"]).all()
    std_mm = torch.exp(0.5 * out["logvar"]) * NORM.scale
    assert std_mm.min() >= 0.1 - 1e-4 and std_mm.max() <= 25.0 + 1e-3        # bounds hold in mm
    assert (torch.exp(out["logvar"]) > 0).all()


def test_initial_uncertainty_near_configured_std_and_mean_near_template():
    m = model().eval()
    with torch.no_grad():
        out = m(*batch(2, 1024))
    mm = gaussian_to_mm(out["mean"], out["logvar"], NORM)
    assert mm["std_mm"].mean().item() == pytest.approx(MCFG["head"]["init_std_mm"], rel=0.2)
    assert (out["mean"] - template()[None]).abs().max() < 0.2


def test_deterministic_mode_has_no_variance_and_fewer_params():
    d, p = model(probabilistic=False), model()
    out = d(*batch())
    assert out["logvar"] is None and out["mean"].shape == (2, V, 3)
    assert d.num_parameters() < p.num_parameters()


def test_gradients_reach_every_parameter():
    m = model().train()
    out = m(*batch(2, 512))
    (out["mean"].pow(2).mean() + out["logvar"].mean()).backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert bad == []
    assert m.num_parameters() < 5_000_000


def test_missing_info_switch_removes_dependence():
    off = model(use_missing_info=False).eval()
    pts, mask, f, mf = batch()
    with torch.no_grad():
        a = off(pts, mask, f, mf)
        b = off(pts, mask, f * 7 + 1, 1 - mf)
        on = model(use_missing_info=True).eval()
        c, d = on(pts, mask, f, mf), on(pts, mask, f * 7 + 1, 1 - mf)
    assert torch.allclose(a["mean"], b["mean"]) and torch.allclose(a["logvar"], b["logvar"])
    assert not torch.allclose(c["mean"], d["mean"])


def test_gaussian_to_mm_conversion():
    mean = torch.zeros(1, 4, 3)
    logvar = torch.log(torch.tensor([[[0.01, 0.04, 0.09]] * 4])).float()
    r = gaussian_to_mm(mean, logvar, NORM)
    assert torch.allclose(r["std_mm"][0, 0], torch.tensor([0.1, 0.2, 0.3]) * 76.5, rtol=1e-5)
    assert torch.allclose(r["var_mm2"], r["std_mm"] ** 2, rtol=1e-5)
    assert torch.allclose(r["pos_var_mm2"], r["var_mm2"].sum(-1))
    assert torch.allclose(r["pos_std_mm"], r["pos_var_mm2"].sqrt())
    assert torch.allclose(r["mean_mm"][0, 0], torch.tensor(NORM.center, dtype=torch.float32))
    assert "var_mm2" not in gaussian_to_mm(mean, None, NORM)


def test_small_batch_can_be_overfit():
    """A few optimiser steps on one fixed batch must reduce the MSE (training smoke test)."""
    m = model(probabilistic=False).train()
    pts, mask, f, mf = batch(2, 512, seed=3)
    # learnable target: scaled template plus a shift that depends on the input cloud
    target = template()[None] * 1.2 + 0.3 * pts.mean(1, keepdim=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (m(pts, mask, f, mf)["mean"] - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all() and losses[-1] < 0.5 * losses[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_forward_backward_batch16():
    m = model().cuda().train()
    pts, mask, f, mf = (t.cuda() for t in batch(16, 1024))
    out = m(pts, mask, f, mf)
    (out["mean"].pow(2).mean() + out["logvar"].mean()).backward()
    assert torch.isfinite(out["mean"]).all()
    assert torch.cuda.max_memory_allocated() < 5.5e9
