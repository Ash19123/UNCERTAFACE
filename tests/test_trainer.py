import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from src.data import build_dataset as bd
from src.losses.composite import CompositeLoss
from src.training import checkpoint as ck
from src.training import trainer as tr

ROOT = Path(__file__).parents[1]
DCFG = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    r = tmp_path_factory.mktemp("trdata")
    bd.build_dataset(DCFG, r, n_base=10, obs_per_base=2, log=lambda *_: None)
    return r


def cfg_for(data_root, out, name, *extra):
    base = tr.load_config(ROOT / "configs/training.yaml", ROOT / "configs/model.yaml", [
        f"run.data_root={json.dumps(str(data_root))}", f"run.out_dir={json.dumps(str(out))}", f"run.name={name}",
        "run.device=cpu", "run.tensorboard=false", "optim.epochs=2", "optim.batch_size=8", "optim.warmup_epochs=1",
        "model.encoder.m1=32", "model.encoder.m2=16", "model.encoder.k1=[8,16]", "model.encoder.k2=[8,16]",
        "model.aggregation.d_model=32", "model.decoder.n_layers=1", "model.decoder.self_attention=false",
        "loss.nll_warmup_epochs=1", "early_stopping.patience=50", *extra])
    return base


def test_overrides():
    c = tr.apply_overrides({"a": {"b": 1, "c": [1]}}, ["a.b=5", "a.c=[2,3]"])
    assert c == {"a": {"b": 5, "c": [2, 3]}}
    with pytest.raises(KeyError):
        tr.apply_overrides({"a": {"b": 1}}, ["a.x=1"])


def test_smoke_train_outputs_and_units(tmp_path, data_root):
    exp = tr.Trainer(cfg_for(data_root, tmp_path, "smoke")).fit(log=lambda *_: None)
    d = tmp_path / "smoke"
    for f in ["config.json", "experiment.json", "history.json", "metrics.csv", "latest.pt", "best.pt"]:
        assert (d / f).exists(), f
    hist = json.loads((d / "history.json").read_text())
    assert len(hist) == 2 and exp["status"] == "completed"
    h = hist[-1]
    for k in ["train_loss", "val_loss", "val_rmse_mm", "val_nll_mm", "val_pos_std_mm", "val_z2", "grad_norm_max", "lr"]:
        assert math.isfinite(h[k]), k
    assert 0.1 < h["val_pos_std_mm"] < 43.4                  # sqrt(3)*25 mm upper bound
    assert 0 < h["val_rmse_mm"] < 100                        # millimetres, not normalised units
    assert exp["hardware"]["device"] == "cpu" and exp["dataset"]["synthetic"] is True
    assert exp["best_validation_metrics"]["val_rmse_mm"] > 0 and exp["seed"] == 0


def test_resume_continues_and_completed_run_is_noop(tmp_path, data_root):
    c1 = cfg_for(data_root, tmp_path, "res", "optim.epochs=1")
    tr.Trainer(c1).fit(log=lambda *_: None)
    c2 = cfg_for(data_root, tmp_path, "res", "optim.epochs=3")
    t2 = tr.Trainer(c2)
    msgs = []
    t2.fit(log=msgs.append)
    assert any("resumed from epoch 1" in m for m in msgs)
    hist = json.loads((tmp_path / "res/history.json").read_text())
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    t3 = tr.Trainer(c2)
    msgs = []
    t3.fit(log=msgs.append)
    assert any("nothing to do" in m for m in msgs) and len(t3.history) == 3


def test_early_stopping(tmp_path, data_root):
    c = cfg_for(data_root, tmp_path, "es", "optim.epochs=10", "optim.lr=0.0", "early_stopping.patience=1",
                "optim.weight_decay=0.0")
    exp = tr.Trainer(c).fit(log=lambda *_: None)
    assert exp["status"] == "early_stopped" and exp["epochs_run"] == 2 and exp["best_epoch"] == 0


def test_nan_loss_is_detected_and_reported(tmp_path, data_root, monkeypatch):
    def nan_forward(self, out, face, nll_factor=1.0):
        return out["mean"].sum() * float("nan"), {}
    monkeypatch.setattr(CompositeLoss, "forward", nan_forward)
    c = cfg_for(data_root, tmp_path, "nan", "monitors.max_nonfinite_steps=2")
    exp = tr.Trainer(c).fit(log=lambda *_: None)
    assert exp["status"] == "diverged"
    assert any("non-finite" in w for e in exp["all_warnings"] for w in e["warnings"])


def test_health_checks(tmp_path, data_root):
    t = tr.Trainer(cfg_for(data_root, tmp_path, "hc"))
    row = dict(train_steps=5, train_skipped_steps=0, grad_norm_over_warn=0, val_loss=1.0, train_loss=1.0)
    assert t.check_health(row) == []
    assert any("collapse" in w for w in t.check_health(dict(row, val_frac_std_at_lower=0.9)))
    assert any("inflation" in w for w in t.check_health(dict(row, val_frac_std_at_upper=0.9)))
    assert any("exploding" in w for w in t.check_health(dict(row, grad_norm_over_warn=3)))
    assert any("non-finite" in w for w in t.check_health(dict(row, val_loss=float("nan"))))
    t.history = [dict(val_loss=v, train_loss=2 - v / 10) for v in (1, 2, 3, 4, 5)]
    assert any("overfitting" in w for w in t.check_health(dict(row, val_loss=6.0, train_loss=0.5)))


def test_lr_schedule_shape(tmp_path, data_root):
    t = tr.Trainer(cfg_for(data_root, tmp_path, "lr", "optim.epochs=10", "optim.warmup_epochs=2"))
    w = t.steps_per_epoch * 2
    f = [t._lr_factor(i) for i in range(t.steps_per_epoch * 10)]
    assert all(a < b for a, b in zip(f[:w - 1], f[1:w])) and f[w] == pytest.approx(1.0, abs=1e-3)
    assert f[-1] == pytest.approx(t.cfg["optim"]["min_lr_frac"], abs=0.02)
    assert t.nll_factor(0) == 0.0 and t.nll_factor(1) == 1.0


def test_deterministic_variant_and_reproducibility(tmp_path, data_root):
    c = cfg_for(data_root, tmp_path, "det", "model.probabilistic=false", "optim.epochs=1")
    t = tr.Trainer(c)
    t.fit(log=lambda *_: None)
    assert "val_pos_std_mm" not in t.history[0] and "train_nll" not in t.history[0]
    a = tr.Trainer(cfg_for(data_root, tmp_path, "r1", "optim.epochs=1")); a.fit(log=lambda *_: None)
    b = tr.Trainer(cfg_for(data_root, tmp_path, "r2", "optim.epochs=1")); b.fit(log=lambda *_: None)
    assert a.history[0]["val_loss"] == pytest.approx(b.history[0]["val_loss"], rel=1e-5)
    assert a.history[0]["train_loss"] == pytest.approx(b.history[0]["train_loss"], rel=1e-5)


def test_warm_start_from_deterministic_checkpoint(tmp_path, data_root):
    det = tr.Trainer(cfg_for(data_root, tmp_path, "w_det", "model.probabilistic=false", "optim.epochs=1"))
    det.fit(log=lambda *_: None)
    c = cfg_for(data_root, tmp_path, "w_prob", f"run.init_from={json.dumps(str(tmp_path / 'w_det/best.pt'))}")
    t = tr.Trainer(c)
    ref = ck.load_checkpoint(tmp_path / "w_det/best.pt")["model"]
    for k in ("decoder.template", "head.mean.weight", "decoder.vert_embed"):
        assert torch.equal(t.model.state_dict()[k].cpu(), ref[k].cpu()), k
    assert t.init_info["fresh_params"] and all(p.startswith("head.raw") for p in t.init_info["fresh_params"])
    t.fit(log=lambda *_: None)
    assert json.loads((tmp_path / "w_prob/experiment.json").read_text())["init_from"]["fresh_params"]
    bad = tmp_path / "bad.pt"
    ck.save_checkpoint(bad, {"model": {"nonsense": torch.zeros(1)}})
    with pytest.raises(ValueError):
        tr.Trainer(cfg_for(data_root, tmp_path, "w_bad", f"run.init_from={json.dumps(str(bad))}"))


def test_checkpoint_roundtrip_is_atomic(tmp_path):
    p = tmp_path / "x.pt"
    ck.save_checkpoint(p, {"a": torch.arange(3), "rng": ck.rng_state()})
    s = ck.load_checkpoint(p)
    assert torch.equal(s["a"], torch.arange(3)) and not (tmp_path / "x.pt.tmp").exists()
    ck.set_rng_state(s["rng"])
    first = torch.rand(4)
    ck.set_rng_state(s["rng"])
    assert torch.equal(first, torch.rand(4))                 # restored state replays the same stream
