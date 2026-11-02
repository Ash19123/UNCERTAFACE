import math

import numpy as np
import pytest
from scipy.stats import norm

from src.data import mesh_utils as mu
from src.evaluation import reconstruction_metrics as rm
from src.evaluation import sparsification as sp
from src.evaluation import uncertainty_metrics as um

V, F = mu.icosphere(2)
V = V * 50.0


def test_vertex_and_pointset_metrics_known_values():
    gt = V.copy()
    assert rm.vertex_rmse(gt, gt) == 0 and rm.chamfer(gt, gt) == 0 and rm.hausdorff(gt, gt) == 0
    shifted = gt + np.array([3.0, 0, 0])
    assert rm.vertex_rmse(shifted, gt) == pytest.approx(3.0)
    assert rm.vertex_mae(shifted, gt) == pytest.approx(1.0)                 # mean over 3 coords: (3+0+0)/3
    perm = np.random.default_rng(0).permutation(len(gt))
    assert rm.chamfer(gt[perm], gt) == 0 and rm.hausdorff(gt[perm], gt) == 0   # correspondence-free
    assert rm.vertex_rmse(gt[perm], gt) > 10                                 # correspondence-based
    assert rm.hausdorff(shifted, gt) >= rm.chamfer(shifted, gt)


def test_point_triangle_distance_exact_cases():
    a, b, c = np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]])
    cases = {(0.2, 0.2, 1.0): 1.0, (2, 0, 0): 1.0, (-1, -1, 0): math.sqrt(2), (0.5, 0.5, 0): 0.0,
             (0.5, -1, 0): 1.0, (0, 2, 0): 1.0, (0.3, 0.3, -2): 2.0, (1, 1, 0): math.sqrt(0.5)}
    for p, expect in cases.items():
        d = math.sqrt(rm._point_triangle_sq_dist(np.array([p], float), a, b, c)[0])
        assert d == pytest.approx(expect, abs=1e-9), p


def test_point_to_surface_matches_bruteforce_and_is_below_vertex_distance():
    rng = np.random.default_rng(1)
    pts = V[rng.choice(len(V), 60)] * rng.uniform(0.8, 1.2, size=(60, 1))
    d = rm.point_to_surface(pts, V, F)
    tri = V[F]
    brute = np.array([np.sqrt(min(rm._point_triangle_sq_dist(np.repeat(p[None], len(F), 0), tri[:, 0], tri[:, 1], tri[:, 2]))) for p in pts])
    assert np.allclose(d, brute, atol=1e-9)
    nn = np.array([np.linalg.norm(V - p, axis=1).min() for p in pts])
    assert (d <= nn + 1e-9).all()                                            # surface distance <= nearest-vertex distance
    assert rm.point_to_surface(V, V, F).max() < 1e-9


def test_surface_error_and_batch():
    gt = np.stack([V, V * 1.02])
    pred = gt + 0.5
    out = rm.evaluate_batch(pred, gt, F)
    assert all(len(v) == 2 for v in out.values())
    assert np.allclose(out["vertex_rmse"], math.sqrt(3 * 0.25))
    assert (out["surface_error"] <= out["vertex_rmse"] + 1e-9).all()
    s = rm.summarise(out)
    assert set(s["chamfer"]) == {"mean", "std"}


def test_coverage_matches_nominal_when_calibrated_and_not_when_overconfident():
    rng = np.random.default_rng(0)
    std = rng.uniform(0.5, 3.0, size=(40, 200, 3))
    gt = rng.normal(size=std.shape) * std
    mean = np.zeros_like(std)
    t = um.coverage_table(mean, std, gt)
    for k in ("50", "80", "90", "95"):
        assert abs(t[k]["gap"]) < 0.01
    assert t["ece_coverage"] < 0.01
    over = um.coverage_table(mean, std * 0.5, gt)
    assert all(over[k]["gap"] < -0.1 for k in ("80", "90", "95"))
    assert um.coverage(mean, std, gt, 0.9, scale=0.5) < 0.9
    assert um.z_for(0.95) == pytest.approx(1.959964, abs=1e-5)


def test_crps_closed_form_and_monte_carlo():
    assert um.crps_gaussian(np.array(0.0), np.array(1.0), np.array(0.0)) == pytest.approx(2 * norm.pdf(0) - 1 / math.sqrt(math.pi))
    rng = np.random.default_rng(3)
    mu_, s, y = 1.0, 2.0, 2.5
    x, x2 = rng.normal(mu_, s, 400000), rng.normal(mu_, s, 400000)
    mc = np.abs(x - y).mean() - 0.5 * np.abs(x - x2).mean()
    assert um.crps_gaussian(np.array(mu_), np.array(s), np.array(y)) == pytest.approx(mc, abs=0.01)
    assert um.crps_gaussian(np.array(0.0), np.array(1.0), np.array(0.0)) < um.crps_gaussian(np.array(0.0), np.array(1.0), np.array(2.0))


def test_summary_keys_and_scale():
    rng = np.random.default_rng(0)
    std = np.full((10, 50, 3), 2.0)
    gt = rng.normal(size=std.shape) * 2.0
    s = um.summary(np.zeros_like(std), std, gt)
    assert s["z_std"] == pytest.approx(1.0, abs=0.05) and s["mean_std_mm"] == 2.0
    s2 = um.summary(np.zeros_like(std), std, gt, scale=2.0)
    assert s2["mean_std_mm"] == 4.0 and s2["coverage"]["90"]["observed"] > s["coverage"]["90"]["observed"]


def test_sparsification_perfect_random_and_inverted():
    rng = np.random.default_rng(0)
    err = rng.gamma(2.0, 1.0, 20000)
    perfect = sp.sparsification_curves(err, err)
    assert perfect["sparsification_error_mm"] == pytest.approx(0, abs=1e-9) and perfect["normalised_se"] == pytest.approx(0, abs=1e-9)
    rand = sp.sparsification_curves(err, rng.random(len(err)))
    assert rand["normalised_se"] == pytest.approx(1.0, abs=0.1)
    inv = sp.sparsification_curves(err, -err)
    assert inv["normalised_se"] > 1.5
    assert np.all(np.diff(perfect["oracle"]) <= 1e-9)                       # removing worst first lowers risk
    assert perfect["aurc_mm"] < perfect["aurc_random_mm"]
    rc = sp.risk_coverage(err, err)
    assert rc["1"] == pytest.approx(err.mean()) and rc["0.1"] < rc["0.5"] < rc["1"]


def test_best_of_n_is_labelled_oracle_and_ordered():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(30, 40, 3))
    mean = gt + rng.normal(scale=1.0, size=gt.shape)
    std = np.full_like(gt, 1.0)
    r = sp.best_of_n(mean, std, gt, n_samples=20)
    assert r["best_of_n_rmse_mm"] < r["mean_sampled_rmse_mm"]
    assert r["single_prediction_rmse_mm"] < r["mean_sampled_rmse_mm"]       # sampling adds noise to the mean
    assert r["n_samples"] == 20
