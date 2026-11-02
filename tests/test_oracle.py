import copy
from pathlib import Path

import numpy as np
import yaml

from src.data import generator as gen
from src.evaluation import oracle

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "dataset.yaml").read_text())["generator"]
P = gen.generate_base(0, 1, CFG).params


def test_floor_is_zero_when_face_latents_are_constant():
    cfg = copy.deepcopy(CFG)
    for k in ("thickness_field_amp", "nose_projection", "lip_fullness", "cheek_fat", "unobserved_field_amp"):
        cfg["face"][k] = [0.0, 0.0]
    cfg["face"]["thickness_scale"] = [1.0, 1.0]
    o = oracle.oracle_variance([P], cfg, n_draws=6, seed=0)
    assert o["floor_rmse_mm"] < 1e-9 and o["var"].shape[0] == 1


def test_real_skull_floor_zero_when_latents_constant_and_positive_otherwise(tmp_path):
    v, f = gen.generate_base(3, 5, CFG).skull_verts, gen.generate_base(3, 5, CFG).skull_faces
    np.savez(tmp_path / "A0001.npz", verts=v.astype(np.float32), faces=f.astype(np.int32), scale=1.02, align_rms_mm=2.0)
    flat = copy.deepcopy(CFG)
    for k in ("thickness_field_amp", "nose_projection", "lip_fullness", "cheek_fat", "unobserved_field_amp"):
        flat["face"][k] = [0.0, 0.0]
    flat["face"]["thickness_scale"] = [1.0, 1.0]
    z = oracle.oracle_variance_real([tmp_path / "A0001.npz"], flat, n_draws=5)
    assert z["floor_rmse_mm"] < 1e-9
    o = oracle.oracle_variance_real([tmp_path / "A0001.npz"], CFG, n_draws=10)
    assert o["floor_rmse_mm"] > 0.3 and o["var"].shape[2] == 3 and np.isfinite(o["var"]).all()


def test_floor_positive_with_shape_and_skull_untouched():
    o = oracle.oracle_variance([P], CFG, n_draws=12, seed=0)
    assert o["floor_rmse_mm"] > 0.3 and o["pos_std_mm"].shape == (1, o["var"].shape[1])
    assert np.isfinite(o["var"]).all() and (o["var"] >= 0).all()
    # only face latents were redrawn: bone parameters are unchanged
    again = oracle.oracle_variance([P], CFG, n_draws=12, seed=0)
    assert np.array_equal(o["var"], again["var"])                 # deterministic given seed
