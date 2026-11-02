"""Gallery of several REAL skulls (MUG500+, CC BY 4.0) each with its SYNTHETIC soft-tissue face.

Rows = different real skulls; columns = front, side, oblique. Skull = real CT-derived geometry; the coloured
surface is the synthetic face (colour = assumed soft-tissue thickness in mm).

  python -m src.visualization.real_skull_gallery --root UNCERTAFACE_DATA_REAL --bases 0 1 2 3 --out reports/real_skull_gallery.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import mesh_utils as mu
from .skull_view import draw_solid


def render(root, bases, out_png, face: bool = True) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    root = Path(root)
    corr = np.load(root / "processed/correspondence/correspondence.npz")
    ff = corr["face_faces"]
    views = [(4, 90, "front"), (4, 180, "side"), (18, 55, "oblique")]
    fig = plt.figure(figsize=(5.4 * len(views), 5.0 * len(bases)))
    info = {}
    for r, b in enumerate(bases):
        sk_v, sk_f = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/B{b:04d}.ply"))
        t = np.load(root / f"processed/facial_targets/B{b:04d}.npz")
        meta = json.loads((root / f"raw/generated/paired_samples/B{b:04d}.json").read_text())
        case = meta.get("extra", {}).get("case", f"B{b:04d}")
        info[b] = dict(case=case, extents_mm=np.ptp(sk_v, 0).round(1).tolist(), n_vertices=len(sk_v))
        for c, (el, az, name) in enumerate(views):
            ax = fig.add_subplot(len(bases), len(views), r * len(views) + c + 1, projection="3d")
            draw_solid(ax, sk_v, sk_f, np.zeros(len(sk_v), bool), el, az)
            if face:
                col = plt.cm.YlOrRd(np.clip(t["thickness"][ff].mean(1) / 20, 0, 1))
                col[:, 3] = 0.8
                ax.add_collection3d(Poly3DCollection(t["face_verts"][ff], facecolors=col, edgecolors=(0.3, 0.3, 0.3, 0.2), linewidths=0.25))
            lim = 100
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim - 35, lim - 35)
            ax.set_title(f"{case} - {name}", fontsize=10)
    fig.suptitle("REAL skulls (MUG500+, CC BY 4.0) + SYNTHETIC soft-tissue faces (colour = assumed thickness, 0-20 mm)", fontsize=12)
    fig.subplots_adjust(left=0, right=1, top=0.96, bottom=0, wspace=0, hspace=0.02)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="UNCERTAFACE_DATA_REAL")
    ap.add_argument("--bases", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="reports/real_skull_gallery.png")
    ap.add_argument("--no-face", action="store_true")
    a = ap.parse_args()
    print(json.dumps(render(a.root, a.bases, a.out, not a.no_face), indent=1))


if __name__ == "__main__":
    main()
