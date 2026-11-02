import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import build_dataset as bd
from src.evaluation import ablation, run_evaluation as rev, stress_tests as st
from src.training import trainer as tr

ROOT = Path(__file__).parents[1]
DCFG = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("ev")
    bd.build_dataset(DCFG, d / "data", n_base=10, obs_per_base=4, log=lambda *_: None)
    ov = [f"run.data_root={json.dumps(str(d / 'data'))}", f"run.out_dir={json.dumps(str(d / 'runs'))}", "run.name=t",
          "run.device=cpu", "run.tensorboard=false", "optim.epochs=2", "optim.batch_size=8", "optim.warmup_epochs=1",
          "model.encoder.m1=32", "model.encoder.m2=16", "model.encoder.k1=[8,16]", "model.encoder.k2=[8,16]",
          "model.aggregation.d_model=32", "model.decoder.n_layers=1", "model.decoder.self_attention=false"]
    tr.Trainer(tr.load_config(ROOT / "configs/training.yaml", ROOT / "configs/model.yaml", ov)).fit(log=lambda *_: None)
    return d


def test_full_evaluation_pipeline_outputs(trained, tmp_path):
    r = rev.evaluate_run(trained / "runs/t", tmp_path / "out", eval_split="validation", sweeps=True, n_sweep_bases=2,
                         dataset_cfg_path=str(ROOT / "configs/dataset.yaml"))
    out = tmp_path / "out"
    for f in ["results.json", "reliability.png", "sparsification.png", "spatial_maps.png", "training_curves.png",
              "stress_grouped.png", "stress_sweeps.png", "calibrator_global.json"]:
        assert (out / f).exists(), f
    assert r["synthetic"] and r["units"] == "mm" and r["probabilistic"]
    for k in ("vertex_rmse", "chamfer", "hausdorff", "point_to_surface", "surface_error"):
        assert r["reconstruction"][k]["mean"] > 0
    assert r["reconstruction"]["surface_error"]["mean"] <= r["reconstruction"]["vertex_rmse"]["mean"] + 1e-6
    for scope in ("global", "per_coordinate", "per_group", "subject_max", "raw"):
        assert "90" in (r["calibration"][scope]["test"] if scope != "raw" else r["calibration"]["raw"])
    assert r["best_of_n"]["note"].startswith("oracle-selected")
    assert set(r["stress_controlled"]) == {"noise", "density", "sampler", "deformation", "missingness"}
    assert len(r["stress_controlled"]["missingness"]) == 20
    json.loads((out / "results.json").read_text())                    # valid JSON (no NaN tokens)


def test_group_table_and_controlled_sweep_determinism(trained):
    from src.evaluation.predict import collect
    from src.evaluation import run_evaluation as r
    from src.visualization.results import load_run
    model, norm, cfg, root = load_run(trained / "runs/t")
    p = collect(model, norm, r._dataset(root, "validation", norm, cfg))
    rows = st.group_table(p, "severity")
    assert sum(x["n"] for x in rows) == len(p["gt"]) and all(x["rmse_mm"] > 0 for x in rows)
    binned = st.group_table(p, "noise_sigma_mm", [0, 0.5, 1.0, 1.51])
    assert sum(x["n"] for x in binned) == len(p["gt"])
    base = sorted(set(p["base_id"].tolist()))[:2]
    a = st.controlled_sweep(model, norm, cfg, root, base, "noise", [0.0, 2.0], DCFG, device="cpu")
    b = st.controlled_sweep(model, norm, cfg, root, base, "noise", [0.0, 2.0], DCFG, device="cpu")
    assert [x["rmse_mm"] for x in a] == [x["rmse_mm"] for x in b]
    with pytest.raises(ValueError):
        st.controlled_sweep(model, norm, cfg, root, base, "bogus", [1], DCFG)
    dz = st.controlled_sweep(model, norm, cfg, root, base, "deformation", [0.0], DCFG)
    assert dz[0]["n"] == len(base)                                   # one prediction per base per setting


def test_ablation_variants_are_valid_overrides():
    base = tr.load_config(ROOT / "configs/training.yaml", ROOT / "configs/model.yaml", [])
    for name, ov in ablation.VARIANTS.items():
        cfg = tr.apply_overrides(base, ov)                            # every key must exist
        assert cfg is not base
    assert len(ablation.VARIANTS) == 9 and {"full_seed1", "full_seed2"} <= set(ablation.VARIANTS)


def test_ensemble_evaluation_separates_uncertainty_kinds(trained, tmp_path):
    from src.evaluation import run_ensemble as re
    d = trained
    ov = [f"run.data_root={json.dumps(str(d / 'data'))}", f"run.out_dir={json.dumps(str(d / 'runs'))}", "run.name=t2",
          "run.seed=1", "run.device=cpu", "run.tensorboard=false", "optim.epochs=2", "optim.batch_size=8",
          "optim.warmup_epochs=1", "model.encoder.m1=32", "model.encoder.m2=16", "model.encoder.k1=[8,16]",
          "model.encoder.k2=[8,16]", "model.aggregation.d_model=32", "model.decoder.n_layers=1",
          "model.decoder.self_attention=false"]
    tr.Trainer(tr.load_config(ROOT / "configs/training.yaml", ROOT / "configs/model.yaml", ov)).fit(log=lambda *_: None)
    r = re.evaluate_ensemble([d / "runs/t", d / "runs/t2"], "e", "validation", out_root=str(tmp_path))
    assert r["n_members"] == 2 and set(r["uncertainty"]) == {"aleatoric", "epistemic", "combined"}
    u = r["uncertainty"]
    assert u["epistemic"]["mean_std_mm"] > 0                         # different seeds disagree
    assert u["combined"]["mean_std_mm"] >= max(u["aleatoric"]["mean_std_mm"], u["epistemic"]["mean_std_mm"]) - 1e-9
    assert 0 < r["epistemic_share_of_total_variance"] < 1
    assert (tmp_path / "e/results.json").exists() and "90" in r["combined_calibrated_global"]
    single = r["ensemble_reconstruction"]["vertex_rmse"]["mean"]
    assert single > 0 and len(r["single_member_rmse_mm"]) == 2
    with pytest.raises(ValueError):                                  # deterministic member rejected
        det = [f"run.data_root={json.dumps(str(d / 'data'))}", f"run.out_dir={json.dumps(str(d / 'runs'))}", "run.name=td",
               "model.probabilistic=false", "run.device=cpu", "run.tensorboard=false", "optim.epochs=1", "optim.batch_size=8",
               "optim.warmup_epochs=1", "model.encoder.m1=32", "model.encoder.m2=16", "model.encoder.k1=[8,16]",
               "model.encoder.k2=[8,16]", "model.aggregation.d_model=32", "model.decoder.n_layers=1",
               "model.decoder.self_attention=false"]
        tr.Trainer(tr.load_config(ROOT / "configs/training.yaml", ROOT / "configs/model.yaml", det)).fit(log=lambda *_: None)
        re.evaluate_ensemble([d / "runs/t", d / "runs/td"], "bad", "validation", out_root=str(tmp_path))


def test_uncertainty_vs_missingness_handles_deterministic():
    p = dict(mean=np.zeros((5, 4, 3)), gt=np.random.default_rng(0).normal(size=(5, 4, 3)), std=None,
             missing_fraction=np.linspace(0.1, 0.8, 5))
    out = rev.uncertainty_vs_missingness(p)
    assert "spearman_missing_vs_error" in out and "pooled_spearman_err_vs_std" not in out
