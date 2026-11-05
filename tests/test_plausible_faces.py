"""Tests for plausible-face sampling and the certain-region table (src/uncertainty/plausible_faces.py)."""
import numpy as np

from src.uncertainty import plausible_faces as pf


def _grid(step=4.0):
    xs = np.arange(-60, 60.1, step)
    zs = np.arange(30, -130.1, -step)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), Z.ravel()], 1)


def test_every_point_gets_exactly_one_known_zone():
    xz = _grid()
    z = pf.zone_ids(xz[:, 0], xz[:, 1])
    assert z.shape == (len(xz),) and z.min() >= 0 and z.max() < len(pf.ZONES)
    assert set(np.unique(z)) == set(range(len(pf.ZONES)))      # all six zones occur on a full face grid


def test_zone_layout_is_top_to_bottom():
    assert pf.zone_ids(np.array([0.0]), np.array([20.0]))[0] == pf.ZONES.index("forehead")
    assert pf.zone_ids(np.array([0.0]), np.array([-60.0]))[0] == pf.ZONES.index("nose")
    assert pf.zone_ids(np.array([31.0]), np.array([-40.0]))[0] == pf.ZONES.index("eyes")
    assert pf.zone_ids(np.array([0.0]), np.array([-85.0]))[0] == pf.ZONES.index("mouth")
    assert pf.zone_ids(np.array([0.0]), np.array([-120.0]))[0] == pf.ZONES.index("chin and jaw")


def test_smoothing_rows_have_unit_norm_so_variance_is_kept():
    xz = _grid(6.0)
    W = pf.smoothing_matrix(xz, 12.0)
    norms = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
    assert np.allclose(norms, 1.0)


def test_samples_match_calibrated_std_and_mean_and_are_smooth():
    xz = _grid(6.0)
    V = len(xz)
    mean = np.zeros((V, 3))
    std = np.full((V, 3), 5.0)
    q90 = 1.2
    W = pf.smoothing_matrix(xz, 12.0)
    S = pf.sample_faces(mean, std, W, q90, k=400, seed=1)
    assert S.shape == (400, V, 3)
    target = 5.0 * q90 / pf.Z90_NOMINAL
    assert abs(S.mean()) < 0.1
    assert abs(S.std(0).mean() - target) < 0.15 * target
    # smooth: neighbouring grid points differ far less than independent draws would
    idx = np.arange(V - 1)
    nb = np.abs(S[:, idx] - S[:, idx + 1]).mean()
    ind = np.abs(np.random.default_rng(0).standard_normal((400, V - 1, 3)) * target * np.sqrt(2)).mean() * 0.8
    assert nb < ind


def test_samples_are_reproducible_and_seed_dependent():
    xz = _grid(8.0)
    W = pf.smoothing_matrix(xz, 12.0)
    m, s = np.zeros((len(xz), 3)), np.ones((len(xz), 3))
    a = pf.sample_faces(m, s, W, 1.0, 3, seed=5)
    assert np.array_equal(a, pf.sample_faces(m, s, W, 1.0, 3, seed=5))
    assert not np.array_equal(a, pf.sample_faces(m, s, W, 1.0, 3, seed=6))


def test_zero_uncertainty_gives_identical_faces():
    xz = _grid(8.0)
    W = pf.smoothing_matrix(xz, 12.0)
    mean = np.random.default_rng(0).normal(size=(len(xz), 3))
    S = pf.sample_faces(mean, np.zeros_like(mean), W, 1.4, 3, seed=0)
    assert np.allclose(S, mean[None])


def test_zone_table_is_sorted_most_certain_first():
    xz = _grid()
    z = pf.zone_ids(xz[:, 0], xz[:, 1])
    hw = np.where(z == pf.ZONES.index("chin and jaw"), 12.0, 6.0) + 0.0 * z
    rows = pf.zone_table(z, hw, spread_mm=hw / 2)
    assert rows[0]["halfwidth_mm"] <= rows[-1]["halfwidth_mm"]
    assert rows[-1]["zone"] == "chin and jaw" and "sample_spread_mm" in rows[0]
