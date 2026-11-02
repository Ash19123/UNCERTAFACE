import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import build_dataset as bd

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "dataset.yaml").read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rep = bd.build_dataset(CFG, root, n_base=10, obs_per_base=2, log=lambda *_: None)
    return root, rep


def test_layout_and_counts(built):
    root, rep = built
    for p in ["raw/generated/cranial_meshes/B0000.ply", "raw/generated/facial_meshes/B0000.ply",
              "raw/generated/paired_samples/B0000.json", "processed/skull_pointclouds/B0000_O00.npz",
              "processed/facial_targets/B0000.npz", "processed/correspondence/correspondence.npz",
              "metadata/samples.jsonl", "splits/train.json", "splits/validation.json",
              "splits/calibration.json", "splits/test.json", "dataset_report.json",
              "visualizations/missingness_density.png"]:
        assert (root / p).exists(), p
    c = rep["counts"]
    assert c["independent_base_geometries"] == 10 and c["observations"] == 20
    assert c["base_per_split"] == {"train": 7, "validation": 1, "calibration": 1, "test": 1}
    assert c["observations_per_split"] == {"train": 14, "validation": 2, "calibration": 2, "test": 2}
    assert json.loads((root / "dataset_report.json").read_text())["synthetic"] is True


def test_all_checks_pass_and_flags(built):
    _, rep = built
    assert rep["all_checks_passed"], rep["checks"]
    meta = json.loads((built[0] / "raw/generated/paired_samples/B0003.json").read_text())
    assert meta["synthetic"] is True and "SYNTHETIC" in meta["provenance"] and meta["units"] == "mm"


def test_sample_record_fields(built):
    root, _ = built
    rec = json.loads((root / "metadata/samples.jsonl").read_text().splitlines()[0])
    for k in ["sample_id", "base_id", "synthetic", "units", "scale_mm_per_unit", "coordinate_transform",
              "observation", "generator_params", "provenance", "pointcloud", "face_target", "correspondence"]:
        assert k in rec
    d = np.load(root / rec["pointcloud"])
    assert d["points"].shape == (rec["observation"]["n_points"], 3) and d["vertex_visible"].dtype == bool


def test_splits_are_by_base_and_disjoint(built):
    root, _ = built
    sp = {s: json.loads((root / f"splits/{s}.json").read_text()) for s in bd.SPLIT_NAMES}
    allb = [b for s in sp.values() for b in s["base_ids"]]
    assert sorted(allb) == list(range(10)) and len(set(allb)) == 10
    for s in sp.values():
        assert {int(i[1:5]) for i in s["sample_ids"]} == set(s["base_ids"])


def test_severities_cycled(built):
    _, rep = built
    assert all(v["count"] > 0 and v["all_in_bin"] for v in rep["missingness"].values())


def test_leakage_detector_catches_leaks():
    samples = [dict(sample_id="B0000_O00", base_id=0, split="train"), dict(sample_id="B0001_O00", base_id=1, split="test")]
    ok = bd.check_leakage({"train": [0], "validation": [], "calibration": [], "test": [1]}, samples)
    assert ok["passed"]
    dup = bd.check_leakage({"train": [0, 1], "validation": [], "calibration": [], "test": [1]}, samples)
    assert not dup["passed"] and dup["base_id_overlap"]
    wrong = bd.check_leakage({"train": [0, 1], "validation": [], "calibration": [], "test": []}, samples)
    assert not wrong["passed"] and wrong["samples_in_wrong_split"] == ["B0001_O00"]


def test_make_splits_proportions_and_determinism():
    s = bd.make_splits(300, CFG["splits"], 1)
    assert [len(s[k]) for k in bd.SPLIT_NAMES] == [210, 30, 30, 30]
    assert s == bd.make_splits(300, CFG["splits"], 1) and s != bd.make_splits(300, CFG["splits"], 2)
    with pytest.raises(ValueError):
        bd.make_splits(3, CFG["splits"], 1)


def test_rebuild_is_reproducible(built, tmp_path):
    root, _ = built
    bd.build_dataset(CFG, tmp_path, n_base=10, obs_per_base=2, log=lambda *_: None)
    for rel in ["processed/skull_pointclouds/B0005_O01.npz", "processed/facial_targets/B0005.npz"]:
        a, b = np.load(root / rel), np.load(tmp_path / rel)
        for k in a.files:
            assert np.array_equal(a[k], b[k]), (rel, k)
    assert (root / "splits/train.json").read_text() == (tmp_path / "splits/train.json").read_text()
