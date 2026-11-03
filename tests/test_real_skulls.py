import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import augmentation as aug
from src.data import build_dataset as bd
from src.data import dataset as ds
from src.data import face_from_skull as ffs
from src.data import generator as gen
from src.data import mesh_utils as mu
from src.data import preprocessing as pp
from src.data import real_skulls as rs
from src.data.geometry_sources import FACE_KEYS, Mug500Source, make_source

ROOT = Path(__file__).parents[1]
DCFG = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())
MCFG = yaml.safe_load((ROOT / "configs" / "dataset_mug500.yaml").read_text())
AUG = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text())["augmentation"]


def write_stl(path, verts, faces):
    tri = verts[faces].astype("<f4")
    rec = np.zeros(len(tri), dtype=rs._STL_REC)
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())


def fake_skull(seed=0):
    s = gen.generate_base(seed, 5, DCFG["generator"])
    return s.skull_verts.astype(float), s.skull_faces


# ---------------------------------------------------------------- STL / decimation / components
def test_stl_roundtrip_and_rejects_corrupt(tmp_path):
    v, f = mu.icosphere(2)
    write_stl(tmp_path / "a.stl", v * 40, f)
    tri = rs.read_stl(tmp_path / "a.stl")
    assert tri.shape == (len(f), 3, 3) and np.allclose(tri, (v * 40)[f], atol=1e-4)
    (tmp_path / "bad.stl").write_bytes((tmp_path / "a.stl").read_bytes()[:-30])
    with pytest.raises(ValueError):
        rs.read_stl(tmp_path / "bad.stl")


def test_decimate_hits_target_and_keeps_shape():
    v, f = mu.icosphere(5)                                      # 10242 verts
    tri = (v * 60)[f]
    dv, df, cell = rs.decimate(tri, 2000)
    assert 1200 < len(dv) < 2600 and df.max() < len(dv) and len(np.unique(df)) == len(dv)
    assert abs(np.linalg.norm(dv, axis=1).mean() - 60) < 2.5     # still on the sphere
    assert mu.surface_area(dv, df) == pytest.approx(4 * np.pi * 60 ** 2, rel=0.12)


def test_keep_main_parts_removes_specks():
    v, f = mu.icosphere(3)
    sv, sf = mu.icosphere(0)
    v2 = np.vstack([v * 50, sv * 2 + 200])
    f2 = np.vstack([f, sf + len(v)])
    kv, kf, n = rs.keep_main_parts(v2, f2, 0.02)
    assert n == 1 and len(kv) == len(v) and kf.max() == len(kv) - 1


# ---------------------------------------------------------------- alignment
def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3)) * 40
    a = np.deg2rad(37)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    dst = 1.07 * src @ R.T + np.array([3.0, -5, 8])
    s, R2, t = rs.umeyama(src, dst)
    assert s == pytest.approx(1.07, rel=1e-6) and np.allclose(R2, R, atol=1e-6) and np.allclose(t, [3, -5, 8], atol=1e-5)


def test_prepare_case_aligns_flipped_scaled_scan(tmp_path):
    ref_v, ref_f = fake_skull(0)
    scan_v = (ref_v * 0.94) @ rs._rot_z(180).T + np.array([4.0, -7, 12])          # flipped, smaller, shifted CT-style scan
    write_stl(tmp_path / "A0002_clear.stl", scan_v, ref_f)
    r = rs.prepare_case(tmp_path / "A0002_clear.stl", ref_v, target_verts=100000)
    assert r["scale"] == pytest.approx(1 / 0.94, rel=0.03)                          # maps scan to reference size
    aligned_ref_frame = r["scale"] * r["verts"].astype(float)
    d = np.linalg.norm(aligned_ref_frame[:, None] - ref_v[None], axis=-1).min(1)
    assert np.sqrt((d ** 2).mean()) < 2.0
    # aligned verts stay in the skull's OWN millimetres: size is the real (smaller) one
    assert np.ptp(r["verts"][:, 0]) == pytest.approx(np.ptp(ref_v[:, 0]) * 0.94, rel=0.03)


def test_prepare_case_rejects_bad_alignment(tmp_path):
    ref_v, ref_f = fake_skull(0)
    junk = np.random.default_rng(1).normal(size=(len(ref_v), 3)) * np.array([5, 150, 5])
    write_stl(tmp_path / "bad_clear.stl", junk, ref_f)
    with pytest.raises(ValueError):
        rs.prepare_case(tmp_path / "bad_clear.stl", ref_v, target_verts=100000, max_rms=3.0)


def test_region_labels_cover_regions():
    v = np.random.default_rng(0).uniform([-70, -90, -130], [70, 90, 70], size=(20000, 3))
    lab = rs.region_labels(v)
    assert set(np.unique(lab)) == set(range(7))
    assert (lab[v[:, 2] < -80] == 6).all()


# ---------------------------------------------------------------- face from skull
def test_envelope_bridges_a_hole_and_face_geometry():
    # front plate at y=80 with a big hole (orbit-like); the closed envelope must span it
    x, z = np.meshgrid(np.arange(-60, 60, 1.0), np.arange(-130, 40, 1.0))
    ok = ~(((x - 30) ** 2 + (z + 35) ** 2) < 16 ** 2)
    plate = np.stack([x[ok], np.full(ok.sum(), 80.0), z[ok]], 1)
    back = np.stack([x.ravel(), np.full(x.size, 20.0), z.ravel()], 1)
    H, xr, zr = ffs.front_envelope(np.vstack([plate, back]))
    i = int((-35 - zr[0]) / ffs.RES); j = int((30 - xr[0]) / ffs.RES)
    assert H[i, j] > 70                                           # hole bridged (raw would fall back to y=20)
    p = {k: v for k, v in gen.sample_params(np.random.default_rng(0), DCFG["generator"]).items() if k in FACE_KEYS}
    out = ffs.build_face(np.vstack([plate, back]), p)
    assert out["face"].shape == out["base"].shape and 1900 < len(out["face"]) < 2300
    off = out["face"] - out["base"]
    assert np.allclose(np.linalg.norm(off, axis=1), out["thickness"], atol=1e-6)
    assert (off[:, 1] > 0).all() and out["thickness"].min() >= 1.0 and out["thickness"].max() < 40
    assert mu.validate_mesh(out["base"], out["faces"]).n_degenerate_faces == 0


def test_thickness_latents_are_bone_independent_and_irreducible():
    sk = fake_skull(1)[0]
    p = {k: v for k, v in gen.sample_params(np.random.default_rng(1), DCFG["generator"]).items() if k in FACE_KEYS}
    a = ffs.build_face(sk, p)
    q = dict(p, nose_projection=11.0, unobserved_field_seed=p["unobserved_field_seed"] + 1)
    b = ffs.build_face(sk, q)
    assert np.array_equal(a["base"], b["base"])                   # same bone -> same base surface
    assert np.abs(a["face"] - b["face"]).max() > 1.0              # different latents -> different face


def test_face_grid_is_left_right_symmetric_for_mirroring():
    X, Z = np.meshgrid(ffs.FACE_X, ffs.FACE_Z, indexing="xy")
    m = ffs.oval_mask(X, Z)
    g = np.stack([X[m], Z[m]], 1)
    from scipy.spatial import cKDTree
    d, idx = cKDTree(g).query(g * [-1, 1])
    assert d.max() < 1e-9 and np.array_equal(idx[idx], np.arange(len(g)))


# ---------------------------------------------------------------- augmentation: warp + mirror
def test_warp_and_mirror_are_joint_and_keep_correspondence():
    g = np.stack(np.meshgrid(np.arange(-20, 21, 10.0), np.arange(-40, 41, 10.0)), -1).reshape(-1, 2)
    face = np.column_stack([g[:, 0], np.full(len(g), 60.0), g[:, 1]])
    from scipy.spatial import cKDTree
    perm = cKDTree(g).query(g * [-1, 1])[1]
    pts = face.copy()                                              # input points ARE the face vertices here
    cfg = dict(AUG, rotation_deg=0, translation_mm=0, scale=0, jitter_mm=0, dropout=[0, 0], density_keep=[1, 1],
               occlusion_prob=0.0, warp_mm=3.0, mirror_prob=1.0)
    o_p, o_f, info = aug.Augmenter(cfg, perm)(pts, face, 0.3, np.random.default_rng(0))
    assert info["mirrored"] and info["warp_mm"] > 0
    assert np.allclose(o_f[perm], o_p)                              # target vertex i sits where its mirrored point went
    # warp is smooth and bounded
    nowarp = dict(cfg, warp_mm=0.0, mirror_prob=0.0)
    _, f0, _ = aug.Augmenter(nowarp, perm)(pts, face, 0.3, np.random.default_rng(0))
    assert np.abs(f0 - face).max() < 1e-9
    cfgw = dict(nowarp, warp_mm=3.0)
    _, fw, _ = aug.Augmenter(cfgw, perm)(pts, face, 0.3, np.random.default_rng(1))
    assert 0 < np.abs(fw - face).max() <= 3.0 * 4 + 1e-6
    assert aug.Augmenter(dict(cfg, mirror_prob=1.0), None)(pts, face, 0.3, np.random.default_rng(0))[2]["mirrored"] is False


# ---------------------------------------------------------------- source + full build
@pytest.fixture(scope="module")
def fake_real(tmp_path_factory):
    d = tmp_path_factory.mktemp("mug")
    proc = d / "processed"
    proc.mkdir()
    for i in range(6):
        v, f = fake_skull(i)
        np.savez_compressed(proc / f"A{i + 1:04d}.npz", verts=v.astype(np.float32), faces=f.astype(np.int32),
                            scale=1.0 + 0.01 * i, align_rms_mm=2.0)
    (d / "SOURCE.json").write_text(json.dumps(dict(dataset="fake", licence="n/a")))
    cfg = json.loads(json.dumps(MCFG))
    cfg["mug500"]["processed_dir"] = str(proc)
    return cfg, d


def test_mug500_source_geometry_and_flags(fake_real):
    cfg, _ = fake_real
    src = make_source(cfg)
    assert isinstance(src, Mug500Source) and src.available() == 6
    g, g2 = src.get(2), src.get(2)
    assert g.real_skull and "SYNTHETIC" in g.provenance and "real" in g.provenance.lower()
    assert np.array_equal(g.face_verts, g2.face_verts)             # deterministic
    assert set(g.params) == set(FACE_KEYS)
    assert g.face_grid_xz.shape == (len(g.face_verts), 2)
    assert np.allclose(np.linalg.norm(g.face_verts - g.face_base, axis=1), g.thickness, atol=1e-5)
    assert not np.array_equal(src.get(0).face_verts, src.get(1).face_verts)


def test_end_to_end_real_style_build_train_and_mirror(fake_real, tmp_path):
    cfg, d = fake_real
    rep = bd.build_dataset(cfg, tmp_path / "data", obs_per_base=4, log=lambda *_: None)
    assert rep["real_skull"] and rep["face_synthetic"] and rep["source"] == "mug500"
    assert rep["all_checks_passed"], rep["checks"]
    assert rep["counts"]["independent_base_geometries"] == 6 and "min_pairwise_skull_chamfer_mm" in rep["duplicates"]
    rec = json.loads((tmp_path / "data/metadata/samples.jsonl").read_text().splitlines()[0])
    assert rec["real_skull"] and rec["face_synthetic"] and rec["skull_source"] == "mug500"
    norm = pp.fit_normalizer_from_train(tmp_path / "data")
    dset = ds.UncertafaceDataset(tmp_path / "data", "train", norm, aug_cfg=dict(AUG, mirror_prob=1.0, warp_mm=3.0), augment=True)
    assert dset.augmenter.mirror_perm is not None                  # grid-based faces support mirroring
    it = dset[0]
    assert it["face"].shape[0] == len(np.load(tmp_path / "data/processed/facial_targets/B0000.npz")["face_verts"])
    proc_ds = ds.UncertafaceDataset(tmp_path / "data", "train", norm)
    assert proc_ds.augmenter is None


def test_procedural_dataset_has_no_mirror_perm(tmp_path):
    bd.build_dataset(DCFG, tmp_path, n_base=10, obs_per_base=1, log=lambda *_: None)
    norm = pp.fit_normalizer_from_train(tmp_path)
    d = ds.UncertafaceDataset(tmp_path, "train", norm, aug_cfg=AUG, augment=True)
    assert d.augmenter.mirror_perm is None


def test_missing_processed_dir_gives_clear_error(tmp_path):
    cfg = json.loads(json.dumps(MCFG))
    cfg["mug500"]["processed_dir"] = str(tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        make_source(cfg)
