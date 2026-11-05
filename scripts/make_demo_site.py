"""Build the multi-skull demo site: one JSON per skull plus an index, from a trained run.

  python scripts/make_demo_site.py --run experiments/runs/face_v3_pn --root UNCERTAFACE_DATA_REAL --out demo/site

Writes ``<out>/data/index.json`` (list of skulls with their split, headline numbers read from reports/) and
``<out>/data/skull_<case>.json`` (decimated real skull, true and predicted synthetic face, per-vertex predicted
uncertainty and actual error for the four missingness severities). ``<out>/index.html`` is ``demo/template.html``.

Every skull of the dataset can be chosen, and each is labelled by how the model used it:
  test        never used for training, tuning or calibration
  validation  not trained on, used to choose the checkpoint
  calibration not trained on, used to calibrate the intervals
  train       SEEN in training (error on these is optimistic)

The skull meshes are decimated derivatives of CC BY 4.0 data (MUG500+, Li et al. 2021); the page carries the
attribution. Faces are synthetic. Face meshes are interpolated onto a finer grid for DISPLAY only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data import dataset as ds
from src.data import mesh_utils as mu
from src.data import real_skulls as rs
from src.evaluation.predict import predict_cloud
from src.visualization.results import load_run

SEVERITIES = ["mild", "moderate", "severe", "extreme"]
SPLITS = ["test", "validation", "calibration", "train"]


def r1(a, d=1):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def make_upsampler(q: np.ndarray, scale: float, step: float = 2.25):
    """Display-only: C1 cubic interpolation of per-vertex fields of the 3 mm face grid onto a finer mesh. Interior points are
    a regular grid, the outline is a ring of points exactly on a smooth oval, and the triangulation is Delaunay (the oval is
    convex), so the edge is smooth instead of stair-stepped. Returns ``(up, faces)``."""
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.spatial import Delaunay
    probe = CloughTocher2DInterpolator(q, np.zeros(len(q)))
    cz, ax, az = -48.0, 70.0, 86.0
    ring_k, in_k = 0.93, 0.895
    xs = np.arange(q[:, 0].min(), q[:, 0].max() + 1e-6, step)
    zs = np.arange(q[:, 1].max(), q[:, 1].min() - 1e-6, -step)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    inner = ((X * scale / ax) ** 2 + ((Z * scale - cz) / az) ** 2 <= in_k ** 2) & np.isfinite(probe(X, Z))
    perim = 2 * np.pi * np.sqrt((ax ** 2 + az ** 2) / 2) * ring_k
    t = np.linspace(0, 2 * np.pi, int(perim / (step * scale)), endpoint=False)
    ring = np.stack([ring_k * ax * np.cos(t) / scale, (cz + ring_k * az * np.sin(t)) / scale], 1)
    ring = ring[np.isfinite(probe(ring[:, 0], ring[:, 1]))]
    q2 = np.concatenate([np.stack([X[inner], Z[inner]], 1), ring])
    faces = Delaunay(q2).simplices

    def up(vals):
        return CloughTocher2DInterpolator(q, np.asarray(vals, float))(q2)
    return up, faces


def build_skull(model, norm, cfg, root, recs: dict, base_id: int, split: str, device, cell_mm: float, step: float):
    rms = lambda e: float(np.sqrt((e ** 2).mean()))
    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{base_id:04d}.ply"))
    dv, df = rs.cluster_decimate(sk_v[sk_f], cell_mm)
    keep = np.unique(df)
    remap = -np.ones(len(dv), int)
    remap[keep] = np.arange(len(keep))
    dv, df = dv[keep], remap[df]
    nn = cKDTree(sk_v).query(dv)[1]
    first = recs[SEVERITIES[0]]
    corr = np.load(root / first["correspondence"])
    tgt = np.load(root / first["face_target"])
    gt, base = tgt["face_verts"], tgt["face_base"]
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    meta = json.loads((root / f"raw/generated/paired_samples/B{base_id:04d}.json").read_text())
    up, f2 = make_upsampler(base[:, [0, 2]], float(meta["extra"]["scale"]), step)
    rng = np.random.default_rng(0)
    obs = {}
    for s, r in recs.items():
        pcz = np.load(root / r["pointcloud"])
        miss = r["observation"]["measured_missing_fraction"]
        o = predict_cloud(model, norm, pcz["points"], miss, cfg["data"]["knn_k"], device, cfg["data"]["n_points"])
        pred = o["mean_mm"][0].cpu().numpy()
        unc = o["pos_std_mm"][0].cpu().numpy()
        err = np.linalg.norm(pred - gt, axis=1)
        pts = pcz["points"]
        pts = pts[rng.choice(len(pts), min(900, len(pts)), replace=False)]
        obs[s] = dict(missing_fraction=miss, pattern=r["observation"]["pattern"], n_points=r["observation"]["n_points"],
                      points=r1(pts), vis=pcz["vertex_visible"][nn].astype(int).tolist(), pred=r1(up(pred)),
                      unc=r1(up(unc)), err=r1(up(err)), rmse=rms(err), baseline_rmse=rms(np.linalg.norm(tpl - gt, axis=1)),
                      mean_unc=float(unc.mean()), mean_err=float(err.mean()))
    return dict(id=meta["extra"]["case"], case=meta["extra"]["case"], split=split, base_id=base_id, skull_v=r1(dv),
                skull_f=df.astype(int).ravel().tolist(), face_faces=f2.astype(int).ravel().tolist(), base=r1(up(base)),
                gt=r1(up(gt)), observations=obs, n_model_vertices=int(len(gt)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/face_v3_pn")
    ap.add_argument("--root", default="UNCERTAFACE_DATA_REAL")
    ap.add_argument("--out", default="demo/site")
    ap.add_argument("--page", default="demo/template.html")
    ap.add_argument("--per-split", type=int, default=6, help="max skulls per split (test is always complete)")
    ap.add_argument("--cell-mm", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=2.25)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, norm, cfg, root = load_run(a.run, a.device)
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    listing = []
    for split in SPLITS:
        d = ds.UncertafaceDataset(root, split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"], cache=False)
        by: dict[int, dict] = {}
        for r in d.records:
            by.setdefault(r["base_id"], {}).setdefault(r["observation"]["severity"], r)
        ids = [b for b, m in sorted(by.items()) if all(s in m for s in SEVERITIES)]
        ids = ids if split == "test" else ids[: a.per_split]
        for b in ids:
            sk = build_skull(model, norm, cfg, root, by[b], b, split, a.device, a.cell_mm, a.step)
            (out / "data" / f"skull_{sk['id']}.json").write_text(json.dumps(sk, separators=(",", ":")))
            listing.append(dict(id=sk["id"], split=split, file=f"data/skull_{sk['id']}.json",
                                rmse={s: sk["observations"][s]["rmse"] for s in SEVERITIES}))
            print(split, sk["id"], round((out / "data" / f"skull_{sk['id']}.json").stat().st_size / 1e6, 2), "MB", flush=True)
    rd = Path("reports")
    res = load_json(rd / "results" / Path(a.run).name / "results.json")
    bl, fl = load_json(rd / "baselines.json"), load_json(rd / "oracle_floor.json")
    rep, exp = load_json(root / "dataset_report.json"), load_json(Path(a.run) / "experiment.json")
    metrics = dict(
        n_test_obs=res["n_test"], model_rmse=res["reconstruction"]["vertex_rmse"]["mean"],
        model_chamfer=res["reconstruction"]["chamfer"]["mean"],
        baseline_rmse=bl["mean_template"]["test"]["vertex_rmse"]["mean"] if bl else None,
        floor_rmse=fl["floor_rmse_mm"] if fl else None,
        by_severity={k: v["vertex_rmse"]["mean"] for k, v in res["reconstruction_by_severity"].items()},
        coverage={lvl: dict(nominal=res["calibration"]["raw"][lvl]["nominal"], raw=res["calibration"]["raw"][lvl]["pointwise"],
                            calibrated=res["calibration"]["global"]["test"][lvl]["pointwise"],
                            raw_simultaneous=res["calibration"]["raw"][lvl]["simultaneous"]) for lvl in ("50", "80", "90", "95")},
        spearman_missing_vs_pred_std=res["missingness_relationship"].get("spearman_missing_vs_pred_std"),
        pooled_spearman_err_vs_std=res["missingness_relationship"].get("pooled_spearman_err_vs_std"),
        sparsification_norm_se=res["sparsification"]["normalised_se"] if "sparsification" in res else None)
    index = dict(skulls=listing, severities=SEVERITIES, metrics=metrics, counts=rep["counts"] if rep else None,
                 run=dict(epochs=exp["epochs_run"], best_epoch=exp["best_epoch"], params=exp["model_params"],
                          seconds=exp["duration_s"]) if exp else None,
                 source=rep.get("source_summary", {}).get("skull_source") if rep else None)
    (out / "data" / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    Path(out / "index.html").write_text(Path(a.page).read_text(encoding="utf-8"), encoding="utf-8")
    print("wrote", out, len(listing), "skulls")


if __name__ == "__main__":
    main()
