"""Reconstruct a face, with per-vertex uncertainty, from ANY skull mesh (.stl / .ply / .obj).

  python scripts/face_from_any_skull.py --skull my_skull.stl --run experiments/runs/face_v3_pn --out out/my_skull

Steps: clean + align the mesh to the reference skull frame (``real_skulls.prepare_case``; fails loudly if the
alignment residual or scale is implausible), sample a point cloud from its surface, run the trained model, apply the
split-conformal scale from the run's calibration split. Writes ``face.ply`` (vertex colour = predicted uncertainty),
``skull_aligned.ply`` and ``result.json``.

Limits (also written to result.json): the model was trained on healthy adult CT skulls (MUG500+) paired with a
SYNTHETIC soft-tissue face, so the output is that synthetic rule applied to your skull, NOT a real face and NOT a
forensic result. Skulls unlike the training set (children, damaged or non-human skulls) may align badly or be
extrapolated; the uncertainty then understates the real error. The reference skull is derived from MUG500+ and stays local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import dataset as ds
from src.data import mesh_utils as mu
from src.data import real_skulls as rs
from src.evaluation.predict import predict_cloud
from src.uncertainty import plausible_faces as pf
from src.uncertainty.calibration import ConformalCalibrator
from src.visualization.results import load_run


def load_triangles(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".stl":
        try:
            return rs.read_stl(path)
        except ValueError:
            pass
    v, f = mu.load_mesh(str(path))
    return v[f]


def grey_colours(vals: np.ndarray, vmax: float) -> np.ndarray:
    g = (255 * (1 - np.clip(vals / vmax, 0, 1) * 0.85)).astype(np.uint8)       # light = sure, dark = unsure
    return np.stack([g, g, g, np.full_like(g, 255)], 1)


def estimate_missing_fraction(skull_v: np.ndarray, skull_f: np.ndarray, ref_v: np.ndarray, tol_mm: float = 8.0, seed: int = 0) -> float:
    """Rough area fraction of the (aligned) skull that is missing: share of reference-skull points with no input surface
    within ``tol_mm``. Individual skulls differ from the reference by a few mm, so this is an estimate (it was checked on
    skulls with known cuts, see docs/BROKEN_SKULL.md), clipped to the training range 0.10-0.85."""
    from scipy.spatial import cKDTree
    pts = mu.sample_surface(skull_v, skull_f, 30000, np.random.default_rng(seed))
    d = cKDTree(pts).query(ref_v)[0]
    return float(np.clip((d > tol_mm).mean(), 0.10, 0.85))


def reconstruct(skull, run_dir, reference="external_data/mug500/processed/reference.npz", missing_fraction="auto",
                n_points=2048, device="cpu", seed=0, n_samples=4, corr_mm=12.0) -> dict:
    model, norm, cfg, root = load_run(run_dir, device)
    ref = np.load(reference)["verts"].astype(np.float64)
    case = rs.prepare_case(load_triangles(Path(skull)) if not isinstance(skull, np.ndarray) else skull, ref)
    v, f = case["verts"].astype(np.float64), case["faces"].astype(np.int64)
    if missing_fraction == "auto":
        missing_fraction = estimate_missing_fraction(v, f, ref)
    missing_fraction = float(missing_fraction)
    pts = mu.sample_surface(v, f, n_points, np.random.default_rng(seed))
    o = predict_cloud(model, norm, pts, missing_fraction, cfg["data"]["knn_k"], device, cfg["data"]["n_points"])
    mean = o["mean_mm"][0].cpu().numpy()
    unc = o["pos_std_mm"][0].cpu().numpy()
    std = o["std_mm"][0].cpu().numpy()
    cal = ConformalCalibrator.load(Path("reports/results") / Path(run_dir).name / "calibrator_global.json")
    k90 = float(cal.scale(0.9, mean[None].shape))
    d = ds.UncertafaceDataset(root, "test", norm, cfg["data"]["n_points"], cfg["data"]["knn_k"])
    faces = np.load(root / d.records[0]["correspondence"])["face_faces"].astype(np.int64)
    tpl = norm.inverse(model.decoder.template.cpu().numpy())
    zone = pf.zone_ids(tpl[:, 0], tpl[:, 2])
    samples = pf.sample_faces(mean, std, pf.smoothing_matrix(tpl[:, [0, 2]], corr_mm), k90, n_samples, seed)
    halfwidth = k90 * unc / np.sqrt(3.0)          # per-coordinate half-width (RMS over axes), mm
    spread = samples.std(0).sum(-1) if n_samples > 1 else None
    return dict(skull_v=v, skull_f=f, face=mean, faces=faces, unc=unc, std_axis=std, calibrated90_halfwidth_axis=k90 * std,
                scale=case["scale"], align_rms_mm=case["align_rms_mm"], missing_fraction=missing_fraction, samples=samples,
                zones=pf.zone_table(zone, halfwidth, None if spread is None else np.linalg.norm(samples.std(0), axis=1)),
                halfwidth_positional=halfwidth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skull", required=True)
    ap.add_argument("--run", default="experiments/runs/face_v3_pn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="external_data/mug500/processed/reference.npz")
    ap.add_argument("--missing-fraction", default="auto",
                    help="'auto' (estimate from the skull) or a number: fraction of the skull missing (training range 0.10-0.85)")
    ap.add_argument("--samples", type=int, default=4, help="number of plausible faces to write (smooth samples, see docs)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    mf = a.missing_fraction if a.missing_fraction == "auto" else float(a.missing_fraction)
    r = reconstruct(a.skull, a.run, a.reference, mf, device=a.device, n_samples=a.samples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import trimesh
    vmax = float(np.quantile(r["unc"], 0.98))
    trimesh.Trimesh(r["face"], r["faces"], vertex_colors=grey_colours(r["unc"], vmax), process=False).export(out / "face.ply")
    mu.save_mesh(str(out / "skull_aligned.ply"), r["skull_v"], r["skull_f"])
    for i, smp in enumerate(r["samples"]):
        trimesh.Trimesh(smp, r["faces"], process=False).export(out / f"face_sample_{i + 1}.ply")
    (out / "result.json").write_text(json.dumps(dict(
        skull=str(a.skull), run=a.run, units="mm", n_face_vertices=int(len(r["face"])), align_rms_mm=r["align_rms_mm"],
        mean_uncertainty_mm=float(r["unc"].mean()), max_uncertainty_mm=float(r["unc"].max()),
        mean_calibrated90_halfwidth_mm=float(np.linalg.norm(r["calibrated90_halfwidth_axis"], axis=1).mean()),
        missing_fraction_used=r["missing_fraction"], missing_fraction_mode=str(a.missing_fraction),
        plausible_faces=dict(n=a.samples, note="smooth samples from the calibrated predictive distribution; spatial correlation 12 mm is an assumption; they show the size of the uncertainty, are NOT better guesses than face.ply"),
        regions_most_to_least_certain=r["zones"],
        limits="synthetic face rule on healthy adult CT skulls; not a real face; not forensic; extrapolates on unlike skulls"),
        indent=2))
    print(json.dumps(json.loads((out / "result.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
