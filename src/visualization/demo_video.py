"""Demo video / poster of the full pipeline on one held-out REAL skull (MUG500+, CC BY 4.0).

Scenes (all from real model output; nothing is mocked):
  1  the complete real skull (CT-derived mesh) rotating
  2  the partial observation the model actually receives (black = observed points, orange = hidden bone)
  3  the predicted face growing over the skull, coloured by the model's predicted uncertainty (mm)
  4  the predicted face coloured by its error against the ground-truth synthetic face (mm), plus the numbers

Every frame carries the disclosure: the FACES are synthetic (procedural soft-tissue layer) - research prototype,
not forensic. The skull is real anonymised CT data (Li et al. 2021, CC BY 4.0).

  python -m src.visualization.demo_video --run experiments/runs/demo_real --out demo --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import dataset as ds
from ..data import mesh_utils as mu
from ..data import real_skulls as rs
from ..evaluation.predict import predict_cloud
from .results import load_run
from .skull_view import draw_solid

CREDIT = "Skull: real anonymised CT data, MUG500+ (Li et al. 2021, CC BY 4.0)  |  Faces: SYNTHETIC  |  Research prototype, not forensic"


def _smooth_step(u: float) -> float:
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


def load_scene_data(run_dir, split="test", sample=None, severity="severe", device="cpu") -> dict:
    model, norm, cfg, root = load_run(run_dir, device)
    d = ds.UncertafaceDataset(root, split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"])
    idx = sample if sample is not None else next(i for i, r in enumerate(d.records) if r["observation"]["severity"] == severity)
    rec = d.records[idx]
    b = rec["base_id"]
    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{b:04d}.ply"))
    tri = sk_v[sk_f]
    dv, df = rs.cluster_decimate(tri, 3.4)                        # lighter mesh for rendering only
    keep = np.unique(df)
    remap = -np.ones(len(dv), int)
    remap[keep] = np.arange(len(keep))
    dv, df = dv[keep], remap[df]
    tgt = np.load(root / rec["face_target"])
    pc = np.load(root / rec["pointcloud"])
    miss = rec["observation"]["measured_missing_fraction"]
    out = predict_cloud(model, norm, pc["points"], miss, cfg["data"]["knn_k"], device, cfg["data"]["n_points"])
    pred = out["mean_mm"][0].cpu().numpy()
    unc = out["pos_std_mm"][0].cpu().numpy() if "pos_std_mm" in out else None
    gt = tgt["face_verts"]
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    meta = json.loads((root / f"raw/generated/paired_samples/B{b:04d}.json").read_text())
    return dict(sk_v=sk_v, dec_v=dv, dec_f=df, faces=np.load(root / rec["correspondence"])["face_faces"], base=tgt["face_base"],
                gt=gt, pred=pred, unc=unc, err=np.linalg.norm(pred - gt, axis=1), template=tpl,
                points=pc["points"], hidden=sk_v[~pc["vertex_visible"]], missing=miss, sample_id=rec["sample_id"],
                case=meta.get("extra", {}).get("case", f"B{b:04d}"), pattern=rec["observation"]["pattern"],
                n_points=rec["observation"]["n_points"])


def _face_poly(ax, verts, faces, vals, vmax, alpha=0.92, cmap="viridis"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    col = plt.get_cmap(cmap)(np.clip(vals[faces].mean(1) / vmax, 0, 1))
    col[:, 3] = alpha
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=col, edgecolors=(0.2, 0.2, 0.2, 0.18), linewidths=0.25))


def _frame(fig, ax, axT, cax, scene: int, u: float, D: dict, metrics: dict | None):
    import matplotlib.pyplot as plt
    ax.clear(); axT.clear(); cax.clear()
    ax.set_axis_off(); axT.set_axis_off(); cax.set_axis_off()
    lim = 105
    cx = D["sk_v"].mean(0)
    def setlim():
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim - 30, lim - 30); ax.set_box_aspect((1, 1, 1))
    vmax = float(np.ceil(max(D["unc"].max() if D["unc"] is not None else 0, D["err"].max())))
    titles = {1: ("1  A real skull", "CT-derived human skull mesh (anonymised, 70k vertices)"),
              2: ("2  What the model sees", f"Only part of the bone is observed: {D['missing'] * 100:.0f}% is hidden\n{D['n_points']} points, '{D['pattern']}' occlusion"),
              3: ("3  It predicts a face", "The face grows from the bone surface.\nColour = the model's own uncertainty (mm)"),
              4: ("4  Checked against the truth", "Colour = actual error vs the (synthetic)\nground-truth face (mm)")}
    if scene == 1:
        draw_solid(ax, D["dec_v"], D["dec_f"], np.zeros(len(D["dec_v"]), bool), 8, 35 + 130 * _smooth_step(u)); setlim()
    elif scene == 2:
        h = D["hidden"][:: max(1, len(D["hidden"]) // 7000)]
        ax.scatter(*h.T, s=2.0, c="#e8a15a", alpha=0.45, depthshade=False)
        if u > 0.25:
            p = D["points"]
            k = int(len(p) * _smooth_step((u - 0.25) / 0.4))
            ax.scatter(*p[:k].T, s=2.6, c="#1b1b1b", depthshade=False)
        ax.view_init(12, 100 + 70 * u); setlim()
    elif scene in (3, 4):
        az = 100 + 45 * _smooth_step(u) if scene == 3 else 145 - 40 * _smooth_step(u)
        draw_solid(ax, D["dec_v"], D["dec_f"], np.zeros(len(D["dec_v"]), bool), 8, az)
        if scene == 3:
            lam = _smooth_step(u * 1.25)
            pos = D["base"] + lam * (D["pred"] - D["base"])
            _face_poly(ax, pos, D["faces"], D["unc"] if D["unc"] is not None else D["err"], vmax)
        else:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.add_collection3d(Poly3DCollection(D["gt"][D["faces"]], facecolors=(0.75, 0.75, 0.75, 0.25), edgecolors="none"))
            _face_poly(ax, D["pred"], D["faces"], D["err"], vmax)
        setlim()
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="viridis")
        cax.set_axis_on(); cax.axis("off")
        cb = fig.colorbar(sm, ax=cax, orientation="horizontal", fraction=1.0, pad=0.0, aspect=22)
        cb.set_label("millimetres (mm)", fontsize=10)
    ttl, sub = titles[scene]
    axT.text(0, 0.96, ttl, fontsize=20, fontweight="bold", va="top")
    axT.text(0, 0.86, sub, fontsize=12.5, va="top", color="#333333", linespacing=1.5)
    if scene == 4:
        rm = lambda e: float(np.sqrt((e ** 2).mean()))
        tpl = rm(np.linalg.norm(D["template"] - D["gt"], axis=1))
        lines = [f"This skull ({D['case']}):",
                 f"  model error      {rm(D['err']):.2f} mm RMSE",
                 f"  no-skull baseline {tpl:.2f} mm RMSE",
                 f"  predicted uncertainty {D['unc'].mean():.1f} mm (mean)" if D["unc"] is not None else ""]
        if metrics:
            lines += ["", f"All held-out skulls (n={metrics['n_skulls']}):",
                      f"  model {metrics['model_rmse']:.2f} mm  vs baseline {metrics['template_rmse']:.2f} mm",
                      f"  floor (best possible here) {metrics['floor']:.2f} mm" if metrics.get("floor") else ""]
        axT.text(0, 0.62, "\n".join(l for l in lines), fontsize=12, va="top", family="monospace", linespacing=1.55)
    axT.text(0, 0.02, "Early research prototype.\nThe FACE is synthetic (a procedural\nsoft-tissue layer), so this shows the\npipeline, not forensic accuracy.", fontsize=10.5, va="bottom", color="#7a1f1f", linespacing=1.4)
    axT.set_xlim(0, 1); axT.set_ylim(0, 1)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render_video(D: dict, out_dir, metrics=None, fps: int = 12, secs=(3.5, 3.5, 4.0, 3.5), scale: float = 1.0, preview: bool = False) -> dict:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = int(1280 * scale), int(720 * scale)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.07, 0.64, 0.93], projection="3d")
    axT = fig.add_axes([0.66, 0.1, 0.33, 0.88])
    cax = fig.add_axes([0.70, 0.045, 0.26, 0.04])
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=9, color="#555555")
    frames_per = [int(s * fps) for s in secs]
    if preview:
        frames_per = [1, 1, 1, 1]
    wr = None if preview else imageio.get_writer(str(out_dir / "uncertaface_demo.mp4"), fps=fps, codec="libx264", quality=8,
                                                  macro_block_size=1, pixelformat="yuv420p")
    gif_frames, previews = [], []
    for scene, n in enumerate(frames_per, start=1):
        for i in range(n):
            u = (i + (0.7 if preview else 0.5)) / n if preview else i / max(1, n - 1)
            img = _frame(fig, ax, axT, cax, scene, u, D, metrics)
            if preview:
                previews.append(img)
            else:
                wr.append_data(img)
                if i % 2 == 0:
                    gif_frames.append(img[::2, ::2])
    plt.close(fig)
    res = {}
    if preview:
        top = np.concatenate(previews[:2], axis=1)
        bot = np.concatenate(previews[2:], axis=1)
        imageio.imwrite(str(out_dir / "preview_scenes.png"), np.concatenate([top, bot], axis=0)[::2, ::2])
    else:
        wr.close()
        imageio.mimsave(str(out_dir / "uncertaface_demo.gif"), gif_frames, duration=1000 / (fps / 2), loop=0)
        res = dict(mp4=str(out_dir / "uncertaface_demo.mp4"), gif=str(out_dir / "uncertaface_demo.gif"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--severity", default="severe")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--metrics-json", help="JSON with n_skulls, model_rmse, template_rmse, floor")
    a = ap.parse_args()
    D = load_scene_data(a.run, a.split, a.sample, a.severity)
    metrics = json.loads(Path(a.metrics_json).read_text()) if a.metrics_json else None
    print(render_video(D, a.out, metrics, preview=a.preview), D["sample_id"], D["case"])


if __name__ == "__main__":
    main()
