from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data import augmentation as aug
from src.data import build_dataset as bd
from src.data import dataset as ds
from src.data import preprocessing as pp

ROOT = Path(__file__).parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())
AUG = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text())["augmentation"]
OFF = dict(AUG, jitter_mm=0.0, dropout=[0, 0], density_keep=[1, 1], occlusion_prob=0.0)


def pts(n=500, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3)) * 50


def test_rotation_matrix_is_proper_and_has_requested_angle():
    R = aug.rotation_matrix(np.array([1.0, 2, 3]), 0.3)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    assert np.arccos((np.trace(R) - 1) / 2) == pytest.approx(0.3)


def test_joint_transform_preserves_correspondence_and_relations():
    p = pts()
    face = p[:80].copy()                                  # target vertices ARE some input points
    out_p, out_f, info = aug.Augmenter(OFF)(p, face, 0.3, np.random.default_rng(1))
    assert np.allclose(out_p[:80], out_f)                 # same transform on input and target
    s = info["scale"]
    d0 = np.linalg.norm(p[0] - p[1:50], axis=1)
    d1 = np.linalg.norm(out_p[0] - out_p[1:50], axis=1)
    assert np.allclose(d1, s * d0)                        # similarity: distances scale uniformly
    assert info["rotation_deg"] <= AUG["rotation_deg"] and abs(s - 1) <= AUG["scale"]
    assert max(abs(v) for v in info["translation_mm"]) <= AUG["translation_mm"]


def test_input_only_degradations_leave_target_alone():
    p, face = pts(), pts(60, 3)
    cfg = dict(OFF, rotation_deg=0.0, translation_mm=0.0, scale=0.0, jitter_mm=1.0, dropout=[0.3, 0.3])
    o_p, o_f, info = aug.Augmenter(cfg)(p, face, 0.2, np.random.default_rng(2))
    assert np.allclose(o_f, face)                          # target untouched
    assert len(o_p) < len(p) and info["dropout_fraction"] == pytest.approx(0.3)
    assert abs(len(o_p) / len(p) - 0.7) < 0.08
    assert info["missing_fraction"] == pytest.approx(0.2)  # thinning does not change missing area


def test_occlusion_removes_contiguous_cap_and_updates_missing_fraction():
    p = pts(2000)
    cfg = dict(OFF, rotation_deg=0.0, translation_mm=0.0, scale=0.0,
               occlusion_prob=1.0, occlusion_fraction=[0.2, 0.2])
    o_p, _, info = aug.Augmenter(cfg)(p, pts(10, 5), 0.5, np.random.default_rng(3))
    assert info["occlusion_removed_fraction"] == pytest.approx(0.2, abs=0.01)
    assert len(o_p) == pytest.approx(1600, abs=5)
    assert info["missing_fraction"] == pytest.approx(1 - 0.5 * 0.8, abs=0.01)
    gone = p[~np.isin(p[:, 0], o_p[:, 0])]
    assert gone.std(0).max() < p.std(0).min()              # removed points are spatially concentrated


def test_augmenter_deterministic_and_never_empties_cloud():
    p, f = pts(), pts(30, 7)
    a = aug.Augmenter(AUG)(p, f, 0.3, np.random.default_rng(4))
    b = aug.Augmenter(AUG)(p, f, 0.3, np.random.default_rng(4))
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]
    harsh = dict(AUG, dropout=[0.999, 0.999], density_keep=[0.001, 0.001])
    assert len(aug.Augmenter(harsh)(p, f, 0.3, np.random.default_rng(5))[0]) >= 16


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    r = tmp_path_factory.mktemp("ds")
    bd.build_dataset(CFG, r, n_base=10, obs_per_base=2, log=lambda *_: None)
    return r, pp.fit_normalizer_from_train(r)


def test_dataset_item_contents_and_units(small):
    root, norm = small
    d = ds.UncertafaceDataset(root, "train", norm, n_points=1024)
    assert len(d) == 14
    it = d[0]
    assert it["points"].shape == (1024, 3) and it["mask"].dtype == torch.bool and it["knn_dist"].shape == (1024, 1)
    assert it["face"].shape == it["face_mm"].shape and it["face"].dtype == torch.float32
    assert np.allclose(norm.inverse(it["face"].numpy()), it["face_mm"].numpy(), atol=1e-3)
    m = it["mask"]
    assert (it["points"][~m] == 0).all() and (it["knn_dist"][~m] == 0).all() and (it["knn_dist"][m] > 0).all()
    assert 0.1 <= float(it["missing_fraction"]) <= 0.85


def test_dataset_padding_when_cloud_smaller(small):
    root, norm = small
    d = ds.UncertafaceDataset(root, "train", norm, n_points=4096)
    sizes = {int(d[i]["mask"].sum()) for i in range(len(d))}
    assert max(sizes) <= 4096 and min(sizes) in (1024, 2048, 4096)
    assert any(int(d[i]["mask"].sum()) < 4096 for i in range(len(d)))     # some padded


def test_eval_deterministic_and_train_augmentation_epoch_behaviour(small):
    root, norm = small
    ev = ds.UncertafaceDataset(root, "validation", norm, aug_cfg=AUG, augment=False)
    assert torch.equal(ev[0]["points"], ev[0]["points"])
    tr = ds.UncertafaceDataset(root, "train", norm, aug_cfg=AUG, augment=True, seed=3)
    a1, a2 = tr[0], tr[0]
    assert torch.equal(a1["points"], a2["points"])                          # same epoch -> same
    tr.set_epoch(1)
    a3 = tr[0]
    assert not torch.equal(a1["points"], a3["points"])                      # new epoch -> new aug
    assert not torch.equal(a1["face_mm"], a3["face_mm"])                    # joint transform moved the target


def test_missing_info_ablation_switch(small):
    root, norm = small
    on = ds.UncertafaceDataset(root, "test", norm, use_missing_info=True)[0]
    off = ds.UncertafaceDataset(root, "test", norm, use_missing_info=False)[0]
    assert torch.equal(on["points"], off["points"]) and float(off["missing_fraction"]) == 0
    assert off["knn_dist"].abs().sum() == 0 and on["knn_dist"].abs().sum() > 0


def test_loader_batches_and_template(small):
    root, norm = small
    d = ds.UncertafaceDataset(root, "train", norm, aug_cfg=AUG, augment=True)
    b = next(iter(ds.make_loader(d, 4, shuffle=True, seed=1)))
    assert b["points"].shape == (4, 1024, 3) and b["face"].shape[0] == 4 and len(b["sample_id"]) == 4
    train_bases = pp.read_split(root, "train")["base_ids"]
    t = ds.fit_template(root, norm, train_bases)
    faces = np.stack([np.load(root / f"processed/facial_targets/B{i:04d}.npz")["face_verts"] for i in train_bases])
    assert np.allclose(norm.inverse(t), faces.mean(0), atol=1e-3)
    ids1 = [x for x in next(iter(ds.make_loader(d, 4, True, seed=1)))["sample_id"]]
    ids2 = [x for x in next(iter(ds.make_loader(d, 4, True, seed=1)))["sample_id"]]
    assert ids1 == ids2


def test_empty_split_raises(small):
    root, norm = small
    with pytest.raises(ValueError):
        ds.UncertafaceDataset(root, "nonexistent", norm)
