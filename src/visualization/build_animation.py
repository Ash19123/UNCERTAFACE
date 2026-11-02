"""Interactive "face being built" animation from REAL model predictions.

One validation skull is revealed progressively (a nested visibility region, so
each step shows strictly more bone than the one before). At every step the
trained model predicts the face from only the revealed points. Frames:
  0  template   - what the model assumes with NO skull evidence (the train-mean face)
  1..n          - model prediction from the revealed skull (missing fraction shrinking)
The ghost surface is the ground-truth synthetic face. Colour = per-vertex error
against that ground truth in mm. This shows how the PREDICTION changes with
evidence; it is not a depiction of the network's internal computation.

Usage:
  python -m src.visualization.build_animation --run experiments/runs/deterministic --out experiments/figures/face_build.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import dataset as ds
from ..data import mesh_utils as mu
from ..data import pointcloud as pc
from ..data import preprocessing as pp
from ..uncertainty.estimation import gaussian_to_mm
from .results import load_run

REVEAL_AXES = {"front": (0.0, 1.0, 0.0), "top": (0.0, 0.0, 1.0), "left": (-1.0, 0.0, 0.0)}


def nested_observations(verts, faces, axis, missing_fractions, n_points, rng, noise_mm=0.3):
    """Observed clouds for decreasing missing fractions that grow from one nested region.

    Visibility is ``d . axis >= tau`` on the direction ``d`` of each surface point;
    ``tau`` is the quantile of a dense area-uniform sample giving the requested
    missing area fraction, so regions are nested and fractions are measured, not nominal.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    dense = mu.sample_surface(verts, faces, 40000, rng)
    s = (dense / np.linalg.norm(dense, axis=1, keepdims=True)) @ a
    out = []
    for m in missing_fractions:
        tau = np.quantile(s, m)                                  # fraction m of area has s < tau (hidden)
        vis = lambda p, _t=tau: (p / np.linalg.norm(p, axis=1, keepdims=True)) @ a >= _t
        pts = pc._sample_visible(verts, faces, vis, n_points, rng, 0.0, a, "area")
        out.append(dict(missing=float((s < tau).mean()), points=pts + rng.normal(scale=noise_mm, size=pts.shape)))
    return out


@torch.no_grad()
def predict_points(model, norm, points_mm, missing_fraction, n_points, knn_k=8, device="cpu"):
    """Model prediction (mm) from a raw point cloud, via the same path as the dataset."""
    p, mask = pp.to_fixed_size(norm.forward(points_mm).astype(np.float32), n_points, np.random.default_rng(0))
    nn = pp.knn_indices(p, min(knn_k, int(mask.sum())), mask)
    kd = np.linalg.norm(p[nn] - p[:, None], axis=-1).mean(1, keepdims=True).astype(np.float32)
    kd[~mask] = 0
    t = lambda x: torch.from_numpy(x)[None].to(device)
    out = model(t(p), t(mask), t(kd), torch.tensor([missing_fraction], dtype=torch.float32, device=device))
    return gaussian_to_mm(out["mean"].float(), None if out["logvar"] is None else out["logvar"].float(), norm)


def build(run_dir, out_html, split="validation", sample=0, reveal="front", device="cpu",
          fractions=(0.85, 0.7, 0.55, 0.4, 0.25, 0.1)) -> dict:
    import plotly.graph_objects as go

    model, norm, cfg, root = load_run(run_dir, device)
    d = ds.UncertafaceDataset(root, split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"])
    rec = d.records[sample]
    bid = rec["base_id"]
    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{bid:04d}.ply"))
    corr = np.load(root / rec["correspondence"])
    ff = corr["face_faces"]
    gt = np.load(root / rec["face_target"])["face_verts"]
    rng = np.random.default_rng(7)
    obs = nested_observations(sk_v, sk_f, REVEAL_AXES[reveal], fractions, cfg["data"]["n_points"], rng)

    steps = [dict(name="template (no skull)", missing=1.0, points=np.zeros((0, 3)),
                  mean=norm.inverse(model.decoder.template.cpu().numpy()), unc=None)]
    for o in obs:
        r = predict_points(model, norm, o["points"], o["missing"], cfg["data"]["n_points"], cfg["data"]["knn_k"], device)
        steps.append(dict(name=f"{o['missing']*100:.0f}% of skull missing", missing=o["missing"], points=o["points"],
                          mean=r["mean_mm"][0].cpu().numpy(),
                          unc=r["pos_std_mm"][0].cpu().numpy() if "pos_std_mm" in r else None))
    for s in steps:
        s["err"] = np.linalg.norm(s["mean"] - gt, axis=1)
        s["rmse"] = float(np.sqrt((s["err"] ** 2).mean()))

    cmax = 8.0
    def face_trace(s):
        hover = [f"error {e:.1f} mm" + (f"<br>uncertainty {u:.1f} mm" if s["unc"] is not None else "")
                 for e, u in zip(s["err"], s["unc"] if s["unc"] is not None else s["err"] * 0)]
        return go.Mesh3d(x=s["mean"][:, 0], y=s["mean"][:, 1], z=s["mean"][:, 2], i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
                         intensity=s["err"], cmin=0, cmax=cmax, colorscale="Viridis", flatshading=False,
                         colorbar=dict(title="error vs ground truth (mm)", len=0.6), text=hover, hoverinfo="text",
                         lighting=dict(ambient=0.55, diffuse=0.8), name="predicted face")
    def pts_trace(s):
        p = s["points"]
        return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="observed skull points",
                            marker=dict(size=1.6, color="rgba(60,60,60,0.85)"), hoverinfo="skip")
    def title(s):
        return (f"SYNTHETIC data | {s['name']} | face RMSE vs ground truth: {s['rmse']:.2f} mm "
                f"(mean-template baseline: {steps[0]['rmse']:.2f} mm)")

    ghost = go.Mesh3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], i=ff[:, 0], j=ff[:, 1], k=ff[:, 2], color="lightgray",
                      opacity=0.18, name="ground-truth face (ghost)", hoverinfo="skip")
    frames = [go.Frame(data=[face_trace(s), pts_trace(s)], traces=[1, 2], name=str(i), layout=dict(title_text=title(s)))
              for i, s in enumerate(steps)]
    fig = go.Figure(data=[ghost, face_trace(steps[0]), pts_trace(steps[0])], frames=frames)
    allp = np.concatenate([gt] + [s["points"] for s in steps if len(s["points"])] + [sk_v])
    lo, hi = allp.min(0), allp.max(0)
    fig.update_layout(
        title=dict(text=title(steps[0]), x=0.02), height=720, margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(aspectmode="data", xaxis=dict(visible=False, range=[lo[0], hi[0]]),
                   yaxis=dict(visible=False, range=[lo[1], hi[1]]), zaxis=dict(visible=False, range=[lo[2], hi[2]]),
                   camera=dict(eye=dict(x=0.9, y=1.9, z=0.5), up=dict(x=0, y=0, z=1))),
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.02, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=1100, redraw=True),
                                                                   transition=dict(duration=300), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(x=0.18, len=0.78, currentvalue=dict(prefix="Evidence step: "),
                      steps=[dict(label=("none" if i == 0 else f"{s['missing']*100:.0f}% missing"), method="animate",
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))]) for i, s in enumerate(steps)])])
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True, auto_play=False)
    summary = dict(run=str(run_dir), sample_id=rec["sample_id"], reveal=reveal,
                   steps=[dict(name=s["name"], missing_fraction=s["missing"], rmse_mm=s["rmse"],
                               n_points=int(len(s["points"])),
                               mean_uncertainty_mm=None if s["unc"] is None else float(s["unc"].mean())) for s in steps])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="experiments/figures/face_build.html")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--reveal", default="front", choices=list(REVEAL_AXES))
    a = ap.parse_args()
    print(json.dumps(build(a.run, a.out, a.split, a.sample, a.reveal), indent=2))


if __name__ == "__main__":
    main()
