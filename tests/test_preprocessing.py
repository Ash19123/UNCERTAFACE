import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import build_dataset as bd
from src.data import mesh_utils as mu
from src.data import preprocessing as pp

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "dataset.yaml").read_text())


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("pp")
    bd.build_dataset(CFG, r, n_base=10, obs_per_base=2, log=lambda *_: None)
    return r


def test_fit_uses_only_requested_bases(root):
    train = pp.read_split(root, "train")["base_ids"]
    n = pp.fit_normalizer(root, train)
    verts = np.concatenate([mu.load_mesh(str(root / f"raw/generated/cranial_meshes/{bd.base_name(b)}.ply"))[0] for b in train])
    assert np.allclose(n.center, verts.mean(0))
    assert n.scale == pytest.approx(np.sqrt(((verts - verts.mean(0)) ** 2).sum(1).mean()))
    assert n.scale != pp.fit_normalizer(root, range(10)).scale          # held-out data would change it
    assert pp.fit_normalizer_from_train(root) == n
    assert 40 < n.scale < 120                                            # mm, head-sized


def test_roundtrip_and_isotropy():
    n = pp.Normalizer((1.0, -2.0, 3.0), 50.0)
    x = np.random.default_rng(0).normal(size=(100, 3)) * 40
    assert np.allclose(n.inverse(n.forward(x)), x)
    d_mm = np.linalg.norm(x[:50] - x[50:], axis=1)
    d_n = np.linalg.norm(n.forward(x[:50]) - n.forward(x[50:]), axis=1)
    assert np.allclose(d_mm, d_n * 50.0)                                 # pure similarity: no shear


def test_uncertainty_converts_to_mm_exactly():
    n = pp.Normalizer((5.0, 5.0, 5.0), 62.5)
    rng = np.random.default_rng(1)
    sigma_n = np.array([0.01, 0.02, 0.03])
    draws_n = rng.normal(0.3, sigma_n, size=(400000, 3))
    mm = n.inverse(draws_n)                                              # push samples through the inverse
    assert np.allclose(mm.std(0), n.std_to_mm(sigma_n), rtol=0.01)       # empirical std in mm
    assert np.allclose(mm.var(0), n.var_to_mm2(sigma_n ** 2), rtol=0.02)
    assert np.allclose(np.exp(n.logvar_to_mm(np.log(sigma_n ** 2))), n.var_to_mm2(sigma_n ** 2))
    assert np.allclose(n.logvar_from_mm(n.logvar_to_mm(-3.0)), -3.0)
    # a 1.6 mm bound corresponds to a normalised half-width of 1.6/62.5
    assert n.std_to_mm(1.6 / 62.5) == pytest.approx(1.6)


def test_normalizer_validation_and_serialisation(tmp_path):
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pp.Normalizer((0, 0, 0), bad)
    n = pp.Normalizer((1.5, 2.5, 3.5), 70.0, "x")
    n.save(tmp_path / "n.json")
    assert pp.Normalizer.load(tmp_path / "n.json") == n
    assert json.loads((tmp_path / "n.json").read_text())["units_of_scale"] == "mm"


def test_load_sample(root):
    rec = json.loads((root / "metadata/samples.jsonl").read_text().splitlines()[0])
    n = pp.fit_normalizer_from_train(root)
    s = pp.load_sample(root, rec, n)
    assert s["points_mm"].dtype == np.float32 and s["points"].shape == s["points_mm"].shape
    assert np.allclose(n.inverse(s["points"]), s["points_mm"], atol=1e-3)
    assert np.allclose(n.inverse(s["face"]), s["face_mm"], atol=1e-3)
    assert s["face"].ndim == 2 and s["face"].shape[1] == 3 and "face_vert_index" not in s
    assert abs(s["points"]).max() < 3                                    # roughly unit-scale after fitting
    assert 0.1 <= s["missing_fraction"] <= 0.85
    assert "points" not in pp.load_sample(root, rec)


def test_to_fixed_size():
    p = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    big, m = pp.to_fixed_size(p, 60, np.random.default_rng(1))
    assert big.shape == (60, 3) and m.all()
    assert set(map(tuple, big)) <= set(map(tuple, p))
    assert np.array_equal(big, pp.to_fixed_size(p, 60, np.random.default_rng(1))[0])
    pad, m = pp.to_fixed_size(p, 150)
    assert pad.shape == (150, 3) and m.sum() == 100 and (pad[100:] == 0).all() and np.array_equal(pad[:100], p)
    same, m = pp.to_fixed_size(p, 100)
    assert np.array_equal(same, p) and m.all()


def test_knn_matches_bruteforce_and_respects_mask():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(200, 3))
    nn = pp.knn_indices(p, 8)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    assert np.array_equal(np.sort(nn, 1), np.sort(np.argsort(d, 1)[:, :8], 1))
    pad, mask = pp.to_fixed_size(p, 260)
    nn2 = pp.knn_indices(pad, 8, mask)
    assert nn2[:200].max() < 200                                         # never selects padding
    assert (nn2[200:] == np.arange(200, 260)[:, None]).all()
    with pytest.raises(ValueError):
        pp.knn_indices(p[:5], 8)
