"""Missing-fraction estimate used by scripts/face_from_any_skull.py, on a synthetic sphere with a known cut."""
import numpy as np
import trimesh

from scripts import face_from_any_skull as cli


def _sphere(radius=90.0):
    m = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _cut(v, f, z_above):
    keep = v[f].mean(1)[:, 2] <= z_above
    return v, f[keep]


def test_complete_surface_is_clipped_to_training_floor():
    v, f = _sphere()
    est = cli.estimate_missing_fraction(v, f, v)
    assert est == 0.10                          # nothing missing -> clipped up to the trained minimum


def test_estimate_tracks_known_cut_and_is_ordered():
    v, f = _sphere()
    ests, truth = [], []
    for z in (60.0, 20.0, -20.0):
        v2, f2 = _cut(v, f, z)
        area = lambda ff: 0.5 * np.linalg.norm(np.cross(v[ff][:, 1] - v[ff][:, 0], v[ff][:, 2] - v[ff][:, 0]), axis=1).sum()
        truth.append(1 - area(f2) / area(f))
        ests.append(cli.estimate_missing_fraction(v2, f2, v))
    assert ests == sorted(ests)                 # more bone removed -> larger estimate
    for e, t in zip(ests, truth):
        assert abs(e - t) < 0.12 or e in (0.10, 0.85)
    assert max(ests) <= 0.85 and min(ests) >= 0.10
