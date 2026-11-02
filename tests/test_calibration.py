import numpy as np
import pytest

from src.uncertainty import calibration as cal
from src.uncertainty import ensemble as ens

LEVELS = (0.5, 0.8, 0.9, 0.95)


def draw(rng, n, v=60, true_scale=(1.0, 1.0, 1.0), pred_scale=1.0, groups=None, group_scale=None):
    base = rng.uniform(0.5, 3.0, size=(n, v, 3))
    ts = np.broadcast_to(np.array(true_scale), (n, v, 3)).copy()
    if groups is not None:
        ts = ts * np.array(group_scale)[groups][None, :, None]
    gt = rng.normal(size=base.shape) * base * ts
    return np.zeros_like(base), base * pred_scale, gt                # model sigma = base * pred_scale


def test_conformal_quantile_finite_sample_rule():
    s = np.arange(1, 10)                                              # n = 9
    assert cal.conformal_quantile(s, 0.8) == 8                        # k = ceil(10*0.8) = 8
    assert cal.conformal_quantile(s, 0.95) == float("inf")            # k = ceil(9.5) = 10 > 9
    assert cal.conformal_quantile(np.array([5.0]), 0.5) == 5.0         # k = ceil(2*0.5) = 1 <= n
    assert cal.conformal_quantile(np.array([5.0]), 0.8) == float("inf")  # k = ceil(1.6) = 2 > n


def test_global_calibration_repairs_overconfidence_on_independent_test():
    rng = np.random.default_rng(0)
    m, s, y = draw(rng, 400, pred_scale=0.5)                          # model claims half the true spread
    c = cal.ConformalCalibrator("global", levels=LEVELS).fit(m, s, y)
    mt, st, yt = draw(rng, 400, pred_scale=0.5)
    raw = cal.evaluate_intervals(mt, st, yt, None, LEVELS)
    post = cal.evaluate_intervals(mt, st, yt, c, LEVELS)
    for k, p in zip(("50", "80", "90", "95"), LEVELS):
        assert raw[k]["pointwise"] < p - 0.1                          # badly under-covered before
        assert post[k]["pointwise"] == pytest.approx(p, abs=0.012)    # near nominal after
    assert post["90"]["mean_half_width_mm"] > raw["90"]["mean_half_width_mm"]
    assert post["95"]["simultaneous"] < post["95"]["pointwise"]       # pointwise != simultaneous


def test_per_coordinate_fixes_axis_specific_miscalibration():
    rng = np.random.default_rng(1)
    ts = (0.5, 1.0, 2.0)
    m, s, y = draw(rng, 300, true_scale=ts)
    g = cal.ConformalCalibrator("global", levels=LEVELS).fit(m, s, y)
    pc = cal.ConformalCalibrator("per_coordinate", levels=LEVELS).fit(m, s, y)
    mt, st, yt = draw(rng, 300, true_scale=ts)
    rg = cal.evaluate_intervals(mt, st, yt, g, LEVELS)["90"]["per_axis"]
    rp = cal.evaluate_intervals(mt, st, yt, pc, LEVELS)["90"]["per_axis"]
    assert max(abs(a - 0.9) for a in rg) > 0.05                       # global is right only on average
    assert all(abs(a - 0.9) < 0.02 for a in rp)


def test_per_group_fixes_region_specific_miscalibration():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(3), 20)
    gs = (0.5, 1.0, 2.0)
    m, s, y = draw(rng, 300, groups=groups, group_scale=gs)
    c = cal.ConformalCalibrator("per_group", groups=groups, levels=LEVELS).fit(m, s, y)
    mt, st, yt = draw(rng, 300, groups=groups, group_scale=gs)
    pg = cal.evaluate_intervals(mt, st, yt, c, LEVELS, groups)["90"]["per_group"]
    assert all(abs(a - 0.9) < 0.02 for a in pg)
    with pytest.raises(ValueError):
        cal.ConformalCalibrator("per_group")


def test_subject_max_gives_simultaneous_coverage_and_inf_when_too_few_bases():
    rng = np.random.default_rng(3)
    nb = 300
    m, s, y = draw(rng, nb, v=30, pred_scale=0.7)
    ids = np.arange(nb)                                               # one observation per base here
    c = cal.ConformalCalibrator("subject_max", levels=LEVELS).fit(m, s, y, base_id=ids)
    mt, st, yt = draw(rng, 3000, v=30, pred_scale=0.7)
    ev = cal.evaluate_intervals(mt, st, yt, c, LEVELS)
    for k, p in zip(("50", "80", "90", "95"), LEVELS):
        assert ev[k]["simultaneous"] >= p - 0.03                      # valid simultaneous coverage
    assert ev["95"]["pointwise"] > 0.99                               # and far more conservative pointwise
    few = cal.ConformalCalibrator("subject_max", levels=LEVELS).fit(m[:5], s[:5], y[:5], base_id=ids[:5])
    assert np.isinf(few.q[0.95]) and np.isfinite(few.q[0.5])
    with pytest.raises(ValueError):
        cal.ConformalCalibrator("subject_max").fit(m, s, y)


def test_save_load_roundtrip_and_scales(tmp_path):
    rng = np.random.default_rng(4)
    m, s, y = draw(rng, 50)
    groups = np.repeat(np.arange(3), 20)
    for scope in ("global", "per_coordinate", "per_group", "subject_max"):
        c = cal.ConformalCalibrator(scope, groups=groups if scope == "per_group" else None, levels=LEVELS)
        c.fit(m, s, y, base_id=np.arange(50))
        c.save(tmp_path / "c.json")
        c2 = cal.ConformalCalibrator.load(tmp_path / "c.json")
        lo, hi = c.interval(m, s, 0.9)
        lo2, hi2 = c2.interval(m, s, 0.9)
        assert np.allclose(lo, lo2) and np.allclose(hi, hi2) and lo.shape == m.shape


def test_calibrator_is_a_pure_function_of_calibration_arrays():
    rng = np.random.default_rng(5)
    m, s, y = draw(rng, 100)
    a = cal.ConformalCalibrator("global", levels=LEVELS).fit(m, s, y)
    b = cal.ConformalCalibrator("global", levels=LEVELS).fit(m.copy(), s.copy(), y.copy())
    assert all(a.q[p] == b.q[p] for p in LEVELS)                      # nothing else can influence it


def test_vertex_groups_deterministic():
    t = np.random.default_rng(0).normal(size=(200, 3))
    a, b = cal.make_vertex_groups(t, 5, 1), cal.make_vertex_groups(t, 5, 1)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 5


def test_ensemble_decomposition():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 10, 3))
    m1 = dict(mean=base + 1.0, std=np.full_like(base, 2.0))
    m2 = dict(mean=base - 1.0, std=np.full_like(base, 2.0))
    r = ens.combine_members([m1, m2])
    assert np.allclose(r["mean"], base) and np.allclose(r["var_aleatoric"], 4.0)
    assert np.allclose(r["var_epistemic"], 1.0)                       # ((+1)^2 + (-1)^2)/2
    assert np.allclose(r["var_total"], 5.0) and np.allclose(r["std_total"], np.sqrt(5.0))
    assert r["epistemic_available"] and r["n_members"] == 2
    same = ens.combine_members([m1, dict(m1)])
    assert np.allclose(same["var_epistemic"], 0)
    one = ens.combine_members([m1])
    assert not one["epistemic_available"] and np.allclose(one["var_total"], one["var_aleatoric"])
