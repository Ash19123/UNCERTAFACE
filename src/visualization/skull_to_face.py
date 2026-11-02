"""Interactive "face grows from the skull" animation (plus a static side-profile strip).

Shows ONE validation sample:
  * the complete skull mesh (grey = bone the model was given, orange = bone hidden from the model),
  * the model's predicted face, drawn as a soft-tissue layer whose vertices move from the skull
    surface (lambda = 0) outward to the predicted position (lambda = 1):
        face(lambda) = skull_vertex + lambda * (predicted_vertex - skull_vertex),
  * the ground-truth synthetic face as a faint ghost,
  * colour = predicted positional uncertainty in mm (hover also shows the actual error).

The prediction itself is the model's genuine output computed from the PARTIAL observed cloud. The growth
over lambda is a visualisation device (linear interpolation along the vertex-wise skull->face offset): it
illustrates the tissue-thickness layer, not the network's internal computation. Skull-face vertex
correspondence is exact only because the skull/face pair is synthetic.

  python -m src.visualization.skull_to_face --run experiments/runs/probabilistic_scratch --severity severe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import dataset as ds
from ..data import mesh_utils as mu
from ..evaluation.predict import predict_cloud
from .results import load_run


def growth_frames(skull_face_verts: np.ndarray, pred: np.ndarray, n_steps: int = 11):
    """Vertex positions for ``lambda`` in [0, 1]; frame 0 == skull surface, last == prediction."""
    lam = np.linspace(0.0, 1.0, n_steps)
    return lam, [skull_face_verts + l * (pred - skull_face_verts) for l in lam]


def _pick(d: ds.UncertafaceDataset, sample: int | None, severity: str | None) -> int:
    if sample is not None:
        return sample
    for i, r in enumerate(d.records):
        if severity is None or r["observation"]["severity"] == severity:
            return i
    raise ValueError(f"no sample with severity {severity!r}")


def build(run_dir, out_html, split="validation", sample=None, severity="severe", device="cpu", n_steps=11,
          profile_png=None) -> dict:
    import plotly.graph_objects as go

    model, norm, cfg, root = load_run(run_dir, device)
    d = ds.UncertafaceDataset(root, split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"])
    i = _pick(d, sample, severity)
    rec = d.records[i]
    b = rec["base_id"]
    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{b:04d}.ply"))
    corr = np.load(root / rec["correspondence"])
    fidx, ff = corr["face_vert_index"], corr["face_faces"]
    gt = np.load(root / rec["face_target"])["face_verts"]
    pc = np.load(root / rec["pointcloud"])
    visible = pc["vertex_visible"]
    miss = rec["observation"]["measured_missing_fraction"]

    out = predict_cloud(model, norm, pc["points"], miss, cfg["data"]["knn_k"], device, cfg["data"]["n_points"])
    pred = out["mean_mm"][0].cpu().numpy()
    unc = out["pos_std_mm"][0].cpu().numpy() if "pos_std_mm" in out else None
    err = np.linalg.norm(pred - gt, axis=1)
    base = sk_v[fidx]
    lam, frames_pos = growth_frames(base, pred, n_steps)

    skull_col = np.where(visible, 0.0, 1.0)
    skull = go.Mesh3d(x=sk_v[:, 0], y=sk_v[:, 1], z=sk_v[:, 2], i=sk_f[:, 0], j=sk_f[:, 1], k=sk_f[:, 2],
                      intensity=skull_col, cmin=0, cmax=1, colorscale=[[0, "#8a8f98"], [1, "#e8a15a"]], showscale=False,
                      opacity=0.55, name="skull (grey: given to model, orange: hidden)", hoverinfo="skip",
                      lighting=dict(ambient=0.6, diffuse=0.7))
    ghost = go.Mesh3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], i=ff[:, 0], j=ff[:, 1], k=ff[:, 2], color="lightgray", opacity=0.12,
                      name="ground-truth face (ghost)", hoverinfo="skip")
    cmax = float(np.ceil(unc.max())) if unc is not None else float(np.ceil(err.max()))
    shown = unc if unc is not None else err
    hover = [f"uncertainty {u:.1f} mm<br>actual error {e:.1f} mm" if unc is not None else f"error {e:.1f} mm"
             for u, e in zip(shown, err)]

    def face(pos):
        return go.Mesh3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], i=ff[:, 0], j=ff[:, 1], k=ff[:, 2], intensity=shown, cmin=0,
                         cmax=cmax, colorscale="Viridis", text=hover, hoverinfo="text", name="predicted face",
                         colorbar=dict(title=("predicted uncertainty (mm)" if unc is not None else "error (mm)"), len=0.55),
                         lighting=dict(ambient=0.55, diffuse=0.8))

    thick = [float(np.linalg.norm(p - base, axis=1).mean()) for p in frames_pos]

    def title(k):
        return (f"SYNTHETIC | {rec['sample_id']} | {miss * 100:.0f}% of skull hidden from the model | soft-tissue layer "
                f"{lam[k] * 100:.0f}% grown (mean distance from bone {thick[k]:.1f} mm) | final face error vs truth "
                f"{np.sqrt((err ** 2).mean()):.2f} mm RMSE")

    frames = [go.Frame(data=[face(p)], traces=[2], name=str(k), layout=dict(title_text=title(k))) for k, p in enumerate(frames_pos)]
    fig = go.Figure(data=[skull, ghost, face(frames_pos[0])], frames=frames)
    allp = np.concatenate([sk_v, pred, gt])
    lo, hi = allp.min(0), allp.max(0)
    cams = dict(front=dict(eye=dict(x=0.1, y=2.3, z=0.35)), side=dict(eye=dict(x=2.3, y=0.5, z=0.25)),
                oblique=dict(eye=dict(x=1.4, y=1.8, z=0.6)))
    fig.update_layout(
        title=dict(text=title(0), x=0.02, font=dict(size=13)), height=740, margin=dict(l=0, r=0, t=70, b=0),
        scene=dict(aspectmode="data", camera=dict(cams["side"], up=dict(x=0, y=0, z=1)),
                   xaxis=dict(visible=False, range=[lo[0], hi[0]]), yaxis=dict(visible=False, range=[lo[1], hi[1]]),
                   zaxis=dict(visible=False, range=[lo[2], hi[2]])),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.02, y=0.02, buttons=[
                dict(label="Grow", method="animate", args=[None, dict(frame=dict(duration=450, redraw=True), transition=dict(duration=150), fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])]),
            dict(type="buttons", direction="left", x=0.55, y=0.02, buttons=[
                dict(label=n, method="relayout", args=[{"scene.camera": dict(c, up=dict(x=0, y=0, z=1))}]) for n, c in cams.items()])],
        sliders=[dict(x=0.15, len=0.8, currentvalue=dict(prefix="Face grown: "),
                      steps=[dict(label=f"{l * 100:.0f}%", method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                                                    transition=dict(duration=0))]) for k, l in enumerate(lam)])],
        annotations=[dict(text="grey bone: observed by the model &nbsp;|&nbsp; orange bone: hidden from the model &nbsp;|&nbsp; ghost: ground-truth face",
                          x=0.02, y=0.96, xref="paper", yref="paper", showarrow=False, font=dict(size=11))])
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, auto_play=False)

    if profile_png:
        _profile_strip(sk_v, base, gt, frames_pos, lam, visible, shown, profile_png, rec["sample_id"], miss, cmax, unc is not None)
    summary = dict(run=str(run_dir), sample_id=rec["sample_id"], split=split, missing_fraction=miss,
                   final_vertex_rmse_mm=float(np.sqrt((err ** 2).mean())),
                   mean_predicted_pos_std_mm=None if unc is None else float(unc.mean()),
                   mean_distance_from_bone_mm_by_step=[dict(lam=float(l), mm=t) for l, t in zip(lam, thick)],
                   note="growth over lambda is linear interpolation for visualisation; the final frame is the model prediction")
    out_html.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def _profile_strip(sk_v, base, gt, frames_pos, lam, visible, shown, path, sid, miss, cmax, has_unc, strip_mm=4.0) -> None:
    """Side-profile (y-z) strip through the midline at several growth stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [0, len(lam) // 3, 2 * len(lam) // 3, len(lam) - 1]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharex=True, sharey=True)
    mid_sk = np.abs(sk_v[:, 0]) < strip_mm
    mid_fc = np.abs(base[:, 0]) < strip_mm
    for ax, k in zip(axes, pick):
        ax.scatter(sk_v[mid_sk, 1], sk_v[mid_sk, 2], s=8, c=np.where(visible[mid_sk], "#8a8f98", "#e8a15a"))
        ax.scatter(gt[np.abs(gt[:, 0]) < strip_mm, 1], gt[np.abs(gt[:, 0]) < strip_mm, 2], s=6, c="lightgray", marker="x")
        p = frames_pos[k]
        sc = ax.scatter(p[mid_fc, 1], p[mid_fc, 2], s=16, c=shown[mid_fc], cmap="viridis", vmin=0, vmax=cmax)
        ax.set(title=f"face grown {lam[k] * 100:.0f}%", xlabel="y, anterior (mm)", aspect="equal")
    axes[0].set_ylabel("z, superior (mm)")
    fig.colorbar(sc, ax=axes, shrink=0.8, label="predicted uncertainty (mm)" if has_unc else "error (mm)")
    fig.suptitle(f"SYNTHETIC midline profile | {sid} | {miss * 100:.0f}% skull hidden | grey/orange: bone given/hidden, x: ground-truth face, "
                 f"coloured: predicted face", fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="reports/skull_to_face.html")
    ap.add_argument("--profile", default="reports/skull_to_face_profile.png")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--severity", default="severe")
    a = ap.parse_args()
    s = build(a.run, a.out, a.split, a.sample, a.severity, profile_png=a.profile)
    print(json.dumps({k: s[k] for k in ("sample_id", "missing_fraction", "final_vertex_rmse_mm", "mean_predicted_pos_std_mm")}, indent=2))


if __name__ == "__main__":
    main()
