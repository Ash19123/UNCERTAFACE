from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import generator as gen
from src.data import mesh_utils as mu

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "dataset.yaml").read_text())
GCFG = CFG["generator"]
SEED = CFG["dataset"]["seed"]
SUBS = [gen.generate_base(i, SEED, GCFG) for i in range(12)]


def test_deterministic_and_order_independent():
    a = gen.generate_base(3, SEED, GCFG)
    b = gen.generate_base(3, SEED, GCFG)
    assert np.array_equal(a.skull_verts, b.skull_verts) and np.array_equal(a.face_verts, b.face_verts)
    assert a.params == b.params
    assert not np.array_equal(a.skull_verts, SUBS[4].skull_verts)


def test_params_within_documented_bounds():
    for s in SUBS:
        assert gen.param_bounds_violations(s.params, GCFG) == []


def test_skull_valid_closed_outward_mesh():
    for s in SUBS:
        rep = mu.validate_mesh(s.skull_verts, s.skull_faces)
        assert rep.is_watertight and rep.euler_characteristic == 2 and rep.n_degenerate_faces == 0
        assert np.isfinite(s.skull_verts).all()
        # star-shaped: no inverted triangles
        assert (mu.face_normals(s.skull_verts, s.skull_faces) * s.skull_verts[s.skull_faces].mean(1)).sum(1).min() > 0
        sizes = np.ptp(s.skull_verts, axis=0)  # full extents in mm: plausible head scale
        assert 100 < sizes[0] < 190 and 140 < sizes[1] < 230 and 100 < sizes[2] < 180


def test_face_correspondence_and_thickness():
    idx0 = SUBS[0].face_vert_index
    for s in SUBS:
        assert np.array_equal(s.face_vert_index, idx0)          # same correspondence everywhere
        assert (s.region[s.face_vert_index] == gen.FACE_REGION).all()
        vn = mu.vertex_normals(s.skull_verts, s.skull_faces)[s.face_vert_index]
        off = s.face_verts - s.skull_verts[s.face_vert_index]
        assert np.allclose(off, vn * s.thickness[:, None])      # face = skull + t * normal
        assert s.thickness.min() >= GCFG["min_thickness_mm"] - 1e-9
        assert s.thickness.max() < 40.0
        assert 300 < len(s.face_verts) < 1200


def test_face_not_wider_than_skull():
    for s in SUBS:
        assert np.ptp(s.face_verts[:, 0]) < np.ptp(s.skull_verts[:, 0])
        assert np.abs(s.face_verts[:, 0]).max() < np.abs(s.skull_verts[:, 0]).max()


def test_attempt_changes_params_and_zero_is_default():
    a0 = gen.generate_base(2, SEED, GCFG, attempt=0)
    assert np.array_equal(a0.skull_verts, gen.generate_base(2, SEED, GCFG).skull_verts)
    assert not np.array_equal(a0.skull_verts, gen.generate_base(2, SEED, GCFG, attempt=1).skull_verts)


def test_face_patch_is_single_disk():
    s = SUBS[0]
    rep = mu.validate_mesh(s.face_verts, s.face_faces)
    assert rep.n_unreferenced_verts == 0 and rep.n_degenerate_faces == 0
    assert rep.n_nonmanifold_edges == 0 and rep.n_boundary_edges > 0
    assert rep.euler_characteristic == 1


def test_diversity_not_effectively_identical():
    skulls = np.stack([s.skull_verts for s in SUBS])
    rms = np.sqrt(((skulls[:, None] - skulls[None]) ** 2).sum(-1).mean(-1))
    off = rms[~np.eye(len(SUBS), dtype=bool)]
    assert off.min() > 1.0  # mm, per-vertex RMS between any two bases


def test_face_has_irreducible_variation_given_fixed_skull():
    p = dict(SUBS[0].params)
    a = gen.build_subject(0, p, GCFG)
    p2 = dict(p, nose_projection=GCFG["face"]["nose_projection"][1] if p["nose_projection"] < 6 else 0.0,
              unobserved_field_seed=p["unobserved_field_seed"] + 1)
    b = gen.build_subject(0, p2, GCFG)
    assert np.array_equal(a.skull_verts, b.skull_verts)         # identical bone
    assert np.abs(a.face_verts - b.face_verts).max() > 1.0      # different face


def test_symmetry_when_asymmetry_zero():
    p = dict(SUBS[0].params, asymmetry=0.0)
    s = gen.build_subject(0, p, GCFG)
    d = mu.icosphere(4)[0]
    from scipy.spatial import cKDTree
    # mirrored skull surface should coincide with itself up to tessellation error
    mirrored = s.skull_verts * np.array([-1, 1, 1])
    dist, _ = cKDTree(s.skull_verts).query(mirrored)
    assert dist.max() < 0.05
    p1 = dict(p, asymmetry=1.0, field_amp=GCFG["skull"]["field_amp"][1])
    s1 = gen.build_subject(0, p1, GCFG)
    d1, _ = cKDTree(s1.skull_verts).query(s1.skull_verts * np.array([-1, 1, 1]))
    assert d1.max() > dist.max()


def test_regions_cover_all_labels():
    lab = SUBS[0].region
    assert set(np.unique(lab)) == set(range(len(gen.REGION_NAMES)))
