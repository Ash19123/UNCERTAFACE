"""Build the data for the interactive demo page from the trained real-skull model and saved results.

  python scripts/make_demo_page.py --run experiments/runs/real_main --root UNCERTAFACE_DATA_REAL

Writes ``demo/demo_data.json`` (one HELD-OUT test skull, its four observations at increasing missingness, the model's
predictions and uncertainty, and evaluation numbers read from reports/). ``demo/build_page.py``-style assembly is done by
``--page demo/template.html --out demo/uncertaface_demo.html`` (replaces ``__DATA__``).

The skull mesh is decimated (~3.6 mm vertex clustering) for the browser. It is a derivative of CC BY 4.0 data
(MUG500+, Li et al. 2021); the page must carry the attribution. Faces are synthetic.
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


def r1(a):
    return np.round(np.asarray(a, dtype=float), 1).tolist()


def make_upsampler(q: np.ndarray, scale: float, step: float = 1.5):
    """Display-only: C1 cubic interpolation of any per-vertex field of the 3 mm face grid onto a fine regular grid
    (``step`` mm) clipped to a smooth oval, so the outline is not stair-stepped. Returns (up, faces)."""
    from scipy.interpolate import CloughTocher2DInterpolator
    from src.data import face_from_skull as ffs
    xs = np.arange(q[:, 0].min(), q[:, 0].max() + 1e-6, step)
    zs = np.arange(q[:, 1].max(), q[:, 1].min() - 1e-6, -step)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    mask = ffs.oval_mask(X * scale / 1.0, Z * scale / 1.0) & np.isfinite(CloughTocher2DInterpolator(q, np.zeros(len(q)))(X, Z))
    q2 = np.stack([X[mask], Z[mask]], 1)
    faces = ffs.grid_triangles(mask)
    # snap outline vertices onto the analytic oval so the edge is smooth instead of stair-stepped
    pad = np.pad(mask, 1)
    edge = (mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]))[mask]
    cx, cz, ax, az = 0.0, -50.0, 58.0 * 0.985, 82.0 * 0.985
    rx, rz = q2[edge, 0] * scale, q2[edge, 1] * scale - cz
    r = np.sqrt((rx / ax) ** 2 + (rz / az) ** 2)
    moved = q2.copy()
    moved[edge, 0], moved[edge, 1] = rx / r / scale, (rz / r + cz) / scale
    ok = np.isfinite(CloughTocher2DInterpolator(q, np.zeros(len(q)))(moved[:, 0], moved[:, 1]))
    q2 = np.where(ok[:, None], moved, q2)

    def up(vals):
        return CloughTocher2DInterpolator(q, np.asarray(vals, float))(q2)
    return up, faces


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def build(run_dir, root, out_json, device="cpu", cell_mm=2.4, n_hidden=1800) -> dict:
    model, norm, cfg, root = load_run(run_dir, device)
    d = ds.UncertafaceDataset(root, "test", norm, cfg["data"]["n_points"], cfg["data"]["knn_k"])
    by_base: dict[int, dict] = {}
    for i, r in enumerate(d.records):
        by_base.setdefault(r["base_id"], {})[r["observation"]["severity"]] = i
    base_id = next(b for b, m in sorted(by_base.items()) if all(s in m for s in SEVERITIES))
    recs = {s: d.records[by_base[base_id][s]] for s in SEVERITIES}

    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{base_id:04d}.ply"))
    dv, df = rs.cluster_decimate(sk_v[sk_f], cell_mm)
    keep = np.unique(df)
    remap = -np.ones(len(dv), int)
    remap[keep] = np.arange(len(keep))
    dv, df = dv[keep], remap[df]
    tree = cKDTree(sk_v)
    nn = tree.query(dv)[1]

    first = recs[SEVERITIES[0]]
    corr = np.load(root / first["correspondence"])
    tgt = np.load(root / first["face_target"])
    gt, base = tgt["face_verts"], tgt["face_base"]
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    meta = json.loads((root / f"raw/generated/paired_samples/B{base_id:04d}.json").read_text())
    rms = lambda e: float(np.sqrt((e ** 2).mean()))

    obs = {}
    rng = np.random.default_rng(0)
    for s, r in recs.items():
        pcz = np.load(root / r["pointcloud"])
        vis = pcz["vertex_visible"]
        miss = r["observation"]["measured_missing_fraction"]
        o = predict_cloud(model, norm, pcz["points"], miss, cfg["data"]["knn_k"], device, cfg["data"]["n_points"])
        pred = o["mean_mm"][0].cpu().numpy()
        unc = o["pos_std_mm"][0].cpu().numpy() if "pos_std_mm" in o else None
        err = np.linalg.norm(pred - gt, axis=1)
        pts = pcz["points"]
        pts = pts[rng.choice(len(pts), min(1024, len(pts)), replace=False)]
        obs[s] = dict(sample_id=r["sample_id"], missing_fraction=miss, pattern=r["observation"]["pattern"],
                      n_points=r["observation"]["n_points"], noise_sigma_mm=r["observation"]["noise_sigma_mm"],
                      points=r1(pts), vis=vis[nn].astype(int).tolist(), pred=r1(pred), unc=None if unc is None else r1(unc),
                      err=r1(err), rmse=rms(err), baseline_rmse=rms(np.linalg.norm(tpl - gt, axis=1)),
                      mean_unc=None if unc is None else float(unc.mean()))

    up, f2 = make_upsampler(base[:, [0, 2]], float(meta["extra"]["scale"]))
    for o in obs.values():
        o["pred"] = r1(up(o["pred"]))
        o["err"] = r1(up(o["err"]))
        if o["unc"] is not None:
            o["unc"] = r1(up(o["unc"]))
    gt_up, base_up, tpl_up = up(gt), up(base), up(tpl)
    rd = Path("reports")
    res = load_json(rd / "results" / Path(run_dir).name / "results.json")
    bl = load_json(rd / "baselines.json")
    fl = load_json(rd / "oracle_floor.json")
    rep = load_json(root / "dataset_report.json")
    exp = load_json(Path(run_dir) / "experiment.json")
    metrics = dict(
        n_test_obs=res["n_test"] if res else None,
        model_rmse=res["reconstruction"]["vertex_rmse"]["mean"] if res else None,
        model_chamfer=res["reconstruction"]["chamfer"]["mean"] if res else None,
        model_surface=res["reconstruction"]["surface_error"]["mean"] if res else None,
        baseline_rmse=bl["mean_template"]["test"]["vertex_rmse"]["mean"] if bl else None,
        floor_rmse=fl["floor_rmse_mm"] if fl else None,
        by_severity={k: v["vertex_rmse"]["mean"] for k, v in res["reconstruction_by_severity"].items()} if res else None,
        coverage=None if not res or not res.get("probabilistic") else {
            lvl: dict(nominal=res["calibration"]["raw"][lvl]["nominal"], raw=res["calibration"]["raw"][lvl]["pointwise"],
                      calibrated=res["calibration"]["global"]["test"][lvl]["pointwise"],
                      raw_simultaneous=res["calibration"]["raw"][lvl]["simultaneous"]) for lvl in ("50", "80", "90", "95")},
        spearman_missing_vs_pred_std=res["missingness_relationship"].get("spearman_missing_vs_pred_std") if res else None,
        sparsification_norm_se=res["sparsification"]["normalised_se"] if res and "sparsification" in res else None)
    data = dict(
        case=meta.get("extra", {}).get("case"), base_id=base_id, skull_v=r1(dv), skull_f=df.astype(int).ravel().tolist(),
        hidden_idx_hint=n_hidden, face_faces=f2.astype(int).ravel().tolist(), base=r1(base_up), gt=r1(gt_up),
        template=r1(tpl_up), n_model_vertices=int(len(gt)), observations=obs, severities=SEVERITIES, metrics=metrics,
        counts=rep["counts"] if rep else None, run=dict(epochs=exp["epochs_run"], best_epoch=exp["best_epoch"],
                                                          params=exp["model_params"], seconds=exp["duration_s"]) if exp else None,
        source=rep.get("source_summary", {}).get("skull_source") if rep else None)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data, separators=(",", ":")))
    return dict(path=str(out_json), kb=round(Path(out_json).stat().st_size / 1024), case=data["case"], metrics=metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/real_main")
    ap.add_argument("--root", default="UNCERTAFACE_DATA_REAL")
    ap.add_argument("--out-json", default="demo/demo_data.json")
    ap.add_argument("--page")
    ap.add_argument("--out", default="demo/uncertaface_demo.html")
    a = ap.parse_args()
    info = build(a.run, a.root, a.out_json)
    print(json.dumps(info, indent=1, default=str))
    if a.page:
        html = Path(a.page).read_text(encoding="utf-8").replace("__DATA__", Path(a.out_json).read_text())
        Path(a.out).write_text(html, encoding="utf-8")
        print("wrote", a.out, round(len(html) / 1024), "KB")


if __name__ == "__main__":
    main()
