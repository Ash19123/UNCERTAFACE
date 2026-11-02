from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import generator as gen
from src.data import mesh_utils as mu
from src.data import pointcloud as pc

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "dataset.yaml").read_text())
OBS, BINS = CFG["observation"], CFG["missingness_bins"]
SUB = gen.generate_base(0, CFG["dataset"]["seed"], CFG["generator"])


def sim(sev="moderate", pat="patchy", n=1024, seed=0, **kw):
    return pc.simulate_observation(SUB.skull_verts, SUB.skull_faces, SUB.region, sev, pat, n,
                                   np.random.default_rng(seed), OBS, BINS, **kw)


@pytest.mark.parametrize("pattern", pc.PATTERNS)
@pytest.mark.parametrize("severity", list(BINS))
def test_measured_fraction_in_bin_and_verified_independently(pattern, severity):
    o = sim(severity, pattern, seed=3)
    lo, hi = BINS[severity]
    assert lo <= o.meta["measured_missing_fraction"] <= hi
    # third, much larger independent estimate
    big = mu.sample_surface(SUB.skull_verts, SUB.skull_faces, 200000, np.random.default_rng(99))
    frac = 1.0 - o.visible_fn(big).mean()
    assert abs(frac - o.meta["measured_missing_fraction"]) < 4 * o.meta["measured_fraction_mc_se"] + 0.005
    assert o.visible_fn(o.points_clean).all()                      # every point is in the visible region
    assert o.points.shape == (1024, 3) and o.points_clean.shape == (1024, 3)
    vv = o.vertex_visible
    assert vv.shape == (len(SUB.skull_verts),) and 0 < vv.sum() < len(vv)


def test_missing_fraction_tracks_vertex_mask():
    o = sim("severe", "cap", seed=1)
    vf = mu.missing_area_fraction(SUB.skull_verts, SUB.skull_faces, o.vertex_visible)
    assert abs(vf - o.meta["measured_missing_fraction"]) < 0.03


def test_deterministic_and_seed_sensitive():
    a, b, c = sim(seed=5), sim(seed=5), sim(seed=6)
    assert np.array_equal(a.points, b.points) and a.meta == b.meta
    assert not np.array_equal(a.points, c.points)


def test_noise_statistics_and_outliers():
    o = sim("mild", n=4096, noise_sigma=1.0, outlier_fraction=0.0)
    r = (o.points - o.points_clean)
    assert r.std() == pytest.approx(1.0, rel=0.05) and abs(r.mean()) < 0.05
    o0 = sim("mild", n=512)
    assert np.array_equal(o0.points, o0.points_clean)
    o2 = sim("mild", n=1000, outlier_fraction=0.02)
    assert (np.linalg.norm(o2.points - o2.points_clean, axis=1) > 0).sum() == 20


def test_density_bias_shifts_sampling():
    lo = sim("mild", n=4096, seed=2, density_kappa=0.0)
    hi = sim("mild", n=4096, seed=2, density_kappa=3.0)
    ax = np.array(hi.meta["density_axis"])
    # bias is relative to each call's own random axis; compare concentration along that axis
    ax_lo = np.array(lo.meta["density_axis"])
    c_hi = (pc._dirs(hi.points_clean) @ ax).mean()
    c_lo = (pc._dirs(lo.points_clean) @ ax_lo).mean()
    assert c_hi > c_lo + 0.1


def test_fps_sampler_is_more_uniform():
    from scipy.spatial import cKDTree
    a = sim("mild", n=1024, sampler="area", seed=4).points_clean
    f = sim("mild", n=1024, sampler="fps", seed=4).points_clean
    nn = lambda p: cKDTree(p).query(p, k=2)[0][:, 1].min()
    assert nn(f) > nn(a)


def test_extreme_with_4096_points_and_random_observation():
    o = sim("extreme", "region", n=4096, seed=8)
    assert o.points.shape == (4096, 3)
    ro = pc.random_observation(SUB.skull_verts, SUB.skull_faces, SUB.region, np.random.default_rng(1),
                               OBS, BINS, CFG["sampling"]["point_counts"])
    assert ro.meta["n_points"] in CFG["sampling"]["point_counts"]
    import json
    json.dumps(ro.meta)  # meta is serialisable


def test_bad_pattern_raises():
    with pytest.raises(ValueError):
        sim(pat="nope")
