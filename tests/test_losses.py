import math

import numpy as np
import pytest
import torch

from src.data import mesh_utils as mu
from src.losses import probabilistic as prob
from src.losses import reconstruction as rec
from src.losses import regularization as reg
from src.losses.composite import CompositeLoss

LCFG = dict(reconstruction="mse", lambda_rec=1.0, lambda_nll=1.0, lambda_surface=0.1, lambda_reg=0.0,
            nll_beta=0.0, unit_mm=10.0, surface_weights=dict(laplacian=1.0, edge=1.0, normal=0.1))
_, FACES = mu.icosphere(2)
VERTS = torch.from_numpy(mu.icosphere(2)[0]).float()[None]


def test_nll_value_and_minimiser():
    mean, tgt = torch.zeros(1, 1, 1), torch.full((1, 1, 1), 2.0)
    assert prob.gaussian_nll(mean, torch.zeros(1, 1, 1), tgt).item() == pytest.approx(0.5 * 4.0)
    lv = torch.linspace(-2, 4, 601).view(-1, 1, 1)
    vals = prob.gaussian_nll(mean.expand_as(lv), lv, tgt.expand_as(lv), reduce=False).flatten()
    assert lv.flatten()[vals.argmin()].item() == pytest.approx(math.log(4.0), abs=0.02)   # sigma^2 = r^2


def test_nll_unit_change_only_adds_constant():
    x, y, lv = torch.randn(2, 5, 3), torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    c = 7.0
    a = prob.gaussian_nll(x, lv, y)
    b = prob.gaussian_nll(c * x, lv + 2 * math.log(c), c * y)
    assert (b - a).item() == pytest.approx(math.log(c), abs=1e-5)


def test_beta_nll_reweights_by_detached_variance():
    mean, tgt = torch.zeros(1, 4, 3), torch.ones(1, 4, 3)
    lv = torch.full((1, 4, 3), 1.0, requires_grad=True)
    plain = prob.gaussian_nll(mean, lv, tgt)
    beta = prob.gaussian_nll(mean, lv, tgt, beta=0.5)
    assert beta.item() == pytest.approx(plain.item() * math.exp(0.5))
    beta.backward()
    assert torch.isfinite(lv.grad).all()


def test_reconstruction_losses_known_values():
    a, b = torch.zeros(1, 2, 3), torch.tensor([[[1.0, 0, 0], [0, 2.0, 0]]])
    assert rec.mse_loss(a, b).item() == pytest.approx((1 + 4) / 6)
    assert rec.mae_loss(a, b).item() == pytest.approx(3 / 6)
    assert rec.chamfer_distance(b, b).item() == 0.0
    perm = b[:, [1, 0]]
    assert rec.chamfer_distance(b, perm).item() == 0.0                       # order-free
    assert rec.chamfer_distance(a, b).item() > 0


def test_mesh_regularisers():
    ops = reg.MeshOps(FACES)
    x = VERTS.clone()
    assert reg.laplacian_loss(ops, x, x).item() == 0 and reg.edge_loss(ops, x, x).item() == 0
    assert reg.normal_loss(ops, x, x).item() == 0
    noisy = (x + 0.02 * torch.randn_like(x)).requires_grad_()
    for f in (reg.laplacian_loss, reg.edge_loss, reg.normal_loss):
        v = f(ops, noisy, x)
        assert v.item() > 0
        v.backward()
        assert torch.isfinite(noisy.grad).all()
    assert torch.allclose(ops.laplacian(torch.ones(1, len(VERTS[0]), 3)), torch.zeros(1, len(VERTS[0]), 3), atol=1e-6)
    assert reg.logvar_smoothness(ops, torch.full((1, len(VERTS[0]), 3), 2.0)).item() < 1e-10
    # uniform scaling changes every edge length -> edge loss positive, normals unchanged
    s = x * 1.1
    assert reg.edge_loss(ops, s, x).item() > 0 and reg.normal_loss(ops, s, x).item() < 1e-8


def make_loss(**over):
    return CompositeLoss(dict(LCFG, **over), reg.MeshOps(FACES), scale_mm=50.0)


def outputs(logvar=True):
    mean = VERTS + 0.01 * torch.randn_like(VERTS)
    return dict(mean=mean.clone().requires_grad_(), logvar=(torch.full_like(VERTS, -4.0).requires_grad_() if logvar else None))


def test_composite_weights_and_units():
    torch.manual_seed(0)
    out = outputs()
    tot, t = make_loss()(out, VERTS)
    k = 50.0 / 10.0
    assert t["rec"].item() == pytest.approx(((out["mean"] - VERTS) ** 2).mean().item() * k * k, rel=1e-4)
    expected = t["rec"] + t["nll"] + 0.1 * (t["lap"] + t["edge"] + 0.1 * t["normal"])
    assert tot.item() == pytest.approx(expected.item(), rel=1e-4)
    tot.backward()
    assert torch.isfinite(out["mean"].grad).all() and torch.isfinite(out["logvar"].grad).all()


def test_composite_switches():
    out = outputs()
    no_nll, t = make_loss(lambda_nll=0.0)(out, VERTS)
    assert no_nll.item() == pytest.approx((t["rec"] + 0.1 * (t["lap"] + t["edge"] + 0.1 * t["normal"])).item(), rel=1e-4)
    ramp0, _ = make_loss()(out, VERTS, nll_factor=0.0)
    assert ramp0.item() == pytest.approx(no_nll.item(), rel=1e-4)
    det, t = make_loss()(outputs(logvar=False), VERTS)
    assert "nll" not in t
    no_surf, t = make_loss(lambda_surface=0.0)(outputs(), VERTS)
    assert "lap" not in t
    _, t = make_loss(lambda_reg=1.0)(outputs(), VERTS)
    assert "reg" in t
    for kind in ("mae", "chamfer"):
        v, _ = make_loss(reconstruction=kind)(outputs(), VERTS)
        assert torch.isfinite(v)
    with pytest.raises(ValueError):
        make_loss(reconstruction="nope")
