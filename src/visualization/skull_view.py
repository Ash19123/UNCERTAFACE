"""Solid, shaded renderings of the synthetic skull (PNG, no browser needed).

Row 1: the COMPLETE skull from four viewpoints (frontal bone area where the face is built is tinted blue).
Row 2: what the model actually receives for the SAME skull at the four missingness levels:
       observed points (dark) and the hidden bone (faint orange dots).

Back-faces are culled per view and faces are shaded by a headlight plus ambient term, so surfaces read as
solid. SYNTHETIC geometry (a deformed ellipsoid with bumps), not real anatomy.

  python -m src.visualization.skull_view --base 9 --out reports/skull_view.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import mesh_utils as mu

VIEWS = {"front": (5, 90), "left side": (5, 180), "top": (85, 90), "oblique": (22, 125)}


def _view_vec(elev: float, azim: float) -> np.ndarray:
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_solid(ax, verts, faces, face_vert_mask, elev, azim, base=(0.80, 0.80, 0.78), tint=(0.55, 0.72, 0.95)) -> None:
    """Draw a shaded, back-face-culled mesh on a 3D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = _view_vec(elev, azim)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    front = n @ v > 0.0
    light = np.clip(n @ (0.8 * v + 0.35 * np.array([-0.3, 0.2, 0.9])), 0, 1)
    shade = 0.28 + 0.72 * light
    col = np.where(face_vert_mask[faces].all(axis=1)[:, None], np.array(tint), np.array(base))
    rgb = np.clip(col * shade[:, None], 0, 1)
    order = np.argsort(tri.mean(1) @ v)                         # far to near
    sel = order[front[order]]
    pc = Poly3DCollection(tri[sel], facecolors=rgb[sel], edgecolors=rgb[sel], linewidths=0.2)
    ax.add_collection3d(pc)
    lim = np.abs(verts).max() * 0.62
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev, azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render(root, base_id: int, out_png, split_obs=("mild", "moderate", "severe", "extreme")) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{base_id:04d}.ply"))
    corr = np.load(root / "processed/correspondence/correspondence.npz")
    is_face = np.zeros(len(sk_v), bool)
    is_face[corr["face_vert_index"]] = True
    recs = [json.loads(l) for l in (root / "metadata/samples.jsonl").read_text().splitlines()]
    recs = {r["observation"]["severity"]: r for r in recs if r["base_id"] == base_id}

    fig = plt.figure(figsize=(17, 8.6))
    for c, (name, (el, az)) in enumerate(VIEWS.items()):
        ax = fig.add_subplot(2, 4, c + 1, projection="3d")
        draw_solid(ax, sk_v, sk_f, is_face, el, az)
        ax.set_title(f"complete skull - {name}", fontsize=10)
    shown = {}
    for c, sev in enumerate(split_obs):
        if sev not in recs:
            continue
        r = recs[sev]
        d = np.load(root / r["pointcloud"])
        ax = fig.add_subplot(2, 4, 5 + c, projection="3d")
        hidden = ~d["vertex_visible"]
        ax.scatter(*sk_v[hidden].T, s=2, c="#e8a15a", alpha=0.35, depthshade=False)
        ax.scatter(*d["points"].T, s=2.2, c="#222222", depthshade=False)
        lim = np.abs(sk_v).max() * 0.62
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.view_init(*VIEWS["oblique"]); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        m = r["observation"]["measured_missing_fraction"]
        ax.set_title(f"model input: {sev}, {m * 100:.0f}% hidden\n({r['observation']['n_points']} points, {r['observation']['pattern']})", fontsize=10)
        shown[sev] = dict(sample_id=r["sample_id"], missing_fraction=m, n_points=r["observation"]["n_points"], pattern=r["observation"]["pattern"])
    fig.suptitle(f"SYNTHETIC skull B{base_id:04d}  |  top row: the complete skull (blue-tinted area = where the face is built)  |  "
                 f"bottom row: what the model is given (black) vs hidden (orange)", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.0, hspace=0.05)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=105)
    plt.close(fig)
    return dict(base_id=base_id, extents_mm=np.ptp(sk_v, axis=0).round(1).tolist(), n_vertices=len(sk_v), observations=shown)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="UNCERTAFACE_DATA")
    ap.add_argument("--base", type=int, default=9)
    ap.add_argument("--out", default="reports/skull_view.png")
    a = ap.parse_args()
    print(json.dumps(render(a.root, a.base, a.out), indent=2))


if __name__ == "__main__":
    main()
