import numpy as np
import pytest

from src.data import mesh_utils as mu


@pytest.mark.parametrize("s", [0, 1, 2, 3])
def test_icosphere_counts_and_topology(s):
    v, f = mu.icosphere(s)
    assert len(v) == 10 * 4 ** s + 2 and len(f) == 20 * 4 ** s
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    rep = mu.validate_mesh(v, f)
    assert rep.is_watertight and rep.euler_characteristic == 2
    assert rep.n_degenerate_faces == 0 and rep.n_nonmanifold_edges == 0


def test_area_converges_to_sphere_and_normals_outward():
    v, f = mu.icosphere(4)
    assert mu.surface_area(v, f) == pytest.approx(4 * np.pi, rel=5e-3)
    assert (mu.vertex_normals(v, f) * v).sum(1).min() > 0.99
    centroids = v[f].mean(1)
    assert (mu.face_normals(v, f) * centroids).sum(1).min() > 0  # outward winding
    assert mu.vertex_areas(v, f).sum() == pytest.approx(mu.surface_area(v, f))


def test_clean_mesh_removes_defects_and_reports_index():
    v, f = mu.icosphere(0)  # 12 verts, 20 faces
    v = np.vstack([v, [[np.nan, 0, 0]], [[5, 5, 5]]])  # NaN vert, unreferenced vert
    f = np.vstack([f, [0, 0, 1], f[0], [3, 4, 99]])  # degenerate, duplicate, out of range
    rep = mu.validate_mesh(v, f)
    assert rep.n_nonfinite_verts == 1 and rep.n_out_of_range_faces == 1
    assert rep.n_degenerate_faces >= 1 and rep.n_duplicate_faces == 1
    cv, cf, idx = mu.clean_mesh(v, f)
    assert len(cv) == 12 and len(cf) == 20 and len(idx) == 12
    assert np.allclose(cv, v[idx])
    assert mu.validate_mesh(cv, cf).is_watertight


def test_open_mesh_not_watertight():
    v, f = mu.icosphere(1)
    assert not mu.validate_mesh(v, f[1:]).is_watertight


def test_sample_surface_on_surface_and_area_uniform():
    v, f = mu.icosphere(3)
    rng = np.random.default_rng(0)
    p = mu.sample_surface(v, f, 20000, rng)
    # points lie within the faceted sphere (radius <= 1, close to 1)
    r = np.linalg.norm(p, axis=1)
    assert r.max() <= 1.0 + 1e-9 and r.min() > 0.97
    # area-uniform: hemisphere fractions ~0.5
    assert abs((p[:, 2] > 0).mean() - 0.5) < 0.02
    # determinism
    p2 = mu.sample_surface(v, f, 100, np.random.default_rng(5))
    p3 = mu.sample_surface(v, f, 100, np.random.default_rng(5))
    assert np.array_equal(p2, p3)
    pts, fi = mu.sample_surface(v, f, 50, rng, return_face_idx=True)
    assert pts.shape == (50, 3) and fi.max() < len(f)


def test_sample_vertices_shapes():
    v, _ = mu.icosphere(2)
    rng = np.random.default_rng(0)
    assert mu.sample_vertices(v, 10, rng).shape == (10, 3)
    assert mu.sample_vertices(v, len(v) + 5, rng).shape == (len(v) + 5, 3)


def test_fps_spread_and_errors():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(500, 3))
    idx = mu.farthest_point_sampling(pts, 32)
    assert len(set(idx.tolist())) == 32
    def min_pair(p):
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 1e9
        return d.min()
    rand = pts[rng.choice(500, 32, replace=False)]
    assert min_pair(pts[idx]) > min_pair(rand)
    with pytest.raises(ValueError):
        mu.farthest_point_sampling(pts, 501)


def test_missing_area_fraction_matches_hemisphere():
    v, f = mu.icosphere(4)
    visible = v[:, 2] <= 0  # equator-ring vertices count as visible -> ~1% below 0.5
    assert mu.missing_area_fraction(v, f, visible) == pytest.approx(0.5, abs=0.02)
    assert mu.missing_area_fraction(v, f, np.ones(len(v), bool)) == 0.0
    assert mu.missing_area_fraction(v, f, np.zeros(len(v), bool)) == pytest.approx(1.0)


def test_mesh_roundtrip_preserves_vertex_order(tmp_path):
    v, f = mu.icosphere(2)
    p = str(tmp_path / "m.ply")
    mu.save_mesh(p, v, f)
    v2, f2 = mu.load_mesh(p)
    assert np.allclose(v, v2, atol=1e-6) and np.array_equal(f, f2)
