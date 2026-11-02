"""Write the UNCERTAFACE dataset to disk, split it, and validate it.

Usage:
  python -m src.data.build_dataset --config configs/dataset.yaml --out UNCERTAFACE_DATA            # procedural toy world
  python -m src.data.build_dataset --config configs/dataset_mug500.yaml --out UNCERTAFACE_DATA_REAL  # real skulls

Geometry source (``dataset.source``): ``procedural`` (fully synthetic) or ``mug500`` (REAL CT-derived skulls from
MUG500+, CC BY 4.0, with a SYNTHETIC soft-tissue face; see ``geometry_sources.py``). Every record carries
``synthetic``, ``real_skull``, ``face_synthetic`` flags and a provenance string.

Layout (``root``):
  raw/generated/cranial_meshes/B0007.ply        complete skull (aligned frame, mm)
  raw/generated/facial_meshes/B0007.ply         ground-truth face (vertex-corresponding across subjects)
  raw/generated/paired_samples/B0007.json       parameters + provenance of the base geometry
  processed/skull_pointclouds/B0007_O01.npz     observed points, clean points, vertex visibility
  processed/facial_targets/B0007.npz            face vertices (mm), thickness, face base, per-vertex skull region
  processed/correspondence/correspondence.npz   shared face triangulation (+ grid coordinates if grid-based)
  metadata/samples.jsonl                        one record per observation
  splits/{train,validation,calibration,test}.json
  visualizations/*.png
  dataset_report.json

Splits are made over BASE geometries before observations are attached, so no base geometry appears in more than
one split. Real skulls are never redistributed from this repository (the dataset folder is git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from . import generator as gen
from . import mesh_utils as mu
from . import pointcloud as pc
from .geometry_sources import (MUG500_PROVENANCE, PROCEDURAL_PROVENANCE, generate_distinct_base, make_source,  # noqa: F401
                               skull_rms)

SPLIT_NAMES = ("train", "validation", "calibration", "test")
PROVENANCE = PROCEDURAL_PROVENANCE          # kept for backward compatibility


def base_name(i: int) -> str:
    return f"B{i:04d}"


def sample_name(i: int, k: int) -> str:
    return f"{base_name(i)}_O{k:02d}"


def make_splits(n_base: int, fractions: dict, seed: int) -> dict[str, list[int]]:
    """Seeded split of base ids. Every split gets >= 1 base; remainder goes to train."""
    order = np.random.default_rng(np.random.SeedSequence([seed, 7777])).permutation(n_base).tolist()
    counts = {s: max(1, int(round(fractions[s] * n_base))) for s in SPLIT_NAMES if s != "train"}
    counts["train"] = n_base - sum(counts.values())
    if counts["train"] < 1:
        raise ValueError("too few base geometries for the requested split")
    out, pos = {}, 0
    for s in SPLIT_NAMES:
        out[s] = sorted(order[pos:pos + counts[s]])
        pos += counts[s]
    return out


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def build_dataset(cfg: dict, root: str | Path, n_base: int | None = None, obs_per_base: int | None = None,
                  log=print, source: str | None = None) -> dict:
    """Generate and write the dataset; returns the validation report."""
    root = Path(root)
    ds = cfg["dataset"]
    src = make_source(cfg, source)
    avail = src.available()
    n_base = int(n_base or (avail if avail is not None else ds["n_base"]))
    if avail is not None:
        n_base = min(n_base, avail)
    n_obs = int(obs_per_base or ds["obs_per_base"])
    seed = int(ds["seed"])
    bins = cfg["missingness_bins"]
    sev_names = list(bins)

    splits = make_splits(n_base, cfg["splits"], seed)
    split_of = {b: s for s, ids in splits.items() for b in ids}
    records, ref_faces = [], None
    for sub_dir in ("cranial_meshes", "facial_meshes", "paired_samples"):
        (root / "raw/generated" / sub_dir).mkdir(parents=True, exist_ok=True)
    for sub_dir in ("facial_targets", "skull_pointclouds", "correspondence"):
        (root / "processed" / sub_dir).mkdir(parents=True, exist_ok=True)

    for b in range(n_base):
        g = src.get(b)
        name = base_name(b)
        if ref_faces is None:
            ref_faces = g.face_faces
            extra = {} if g.face_grid_xz is None else dict(face_grid_xz=g.face_grid_xz)
            np.savez_compressed(root / "processed/correspondence/correspondence.npz", face_faces=g.face_faces, **extra)
        assert np.array_equal(ref_faces, g.face_faces), "face triangulation changed between subjects"
        mu.save_mesh(str(root / "raw/generated/cranial_meshes" / f"{name}.ply"), g.skull_verts, g.skull_faces)
        mu.save_mesh(str(root / "raw/generated/facial_meshes" / f"{name}.ply"), g.face_verts, g.face_faces)
        _write_json(root / "raw/generated/paired_samples" / f"{name}.json", dict(
            base_id=b, split=split_of[b], synthetic=True, real_skull=g.real_skull, face_synthetic=True, units="mm",
            provenance=g.provenance, skull_source=src.name, generator_params=g.params, extra=g.extra,
            skull_mesh=f"raw/generated/cranial_meshes/{name}.ply", face_mesh=f"raw/generated/facial_meshes/{name}.ply", seed=seed))
        np.savez_compressed(root / "processed/facial_targets" / f"{name}.npz", face_verts=g.face_verts,
                            thickness=g.thickness, face_base=g.face_base, skull_region=g.region.astype(np.int8))
        for k in range(n_obs):
            rng = np.random.default_rng(np.random.SeedSequence([seed, b, k, 1]))
            sev = sev_names[(b + k) % len(sev_names)]
            o = pc.random_observation(g.skull_verts, g.skull_faces, g.region, rng, cfg["observation"], bins,
                                      cfg["sampling"]["point_counts"], severity=sev)
            sid = sample_name(b, k)
            np.savez_compressed(root / "processed/skull_pointclouds" / f"{sid}.npz", points=o.points,
                                points_clean=o.points_clean, vertex_visible=o.vertex_visible)
            records.append(dict(
                sample_id=sid, base_id=b, split=split_of[b], synthetic=True, real_skull=g.real_skull, face_synthetic=True,
                skull_source=src.name, units="mm", scale_mm_per_unit=1.0,
                coordinate_transform=dict(type="identity", frame="aligned_skull_mm"),
                pointcloud=f"processed/skull_pointclouds/{sid}.npz", face_target=f"processed/facial_targets/{name}.npz",
                correspondence="processed/correspondence/correspondence.npz",
                observation=o.meta, generator_params=g.params, provenance=g.provenance))
        if (b + 1) % 10 == 0 or b + 1 == n_base:
            log(f"built {b + 1}/{n_base} base geometries")

    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata/samples.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in records))
    for s in SPLIT_NAMES:
        _write_json(root / "splits" / f"{s}.json", dict(
            split=s, base_ids=splits[s], sample_ids=[r["sample_id"] for r in records if r["split"] == s]))
    report = validate_dataset(root, cfg)
    report["source"] = src.name
    report["source_summary"] = src.summary()
    report["config_sha256"] = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    report["dataset_version"] = ds["version"]
    report["config"] = cfg
    _write_json(root / "dataset_report.json", report)
    try:
        make_plots(root, records)
    except Exception as e:  # plots are auxiliary; never hide the failure
        log(f"WARNING: plotting failed: {e!r}")
    return report


# --------------------------------------------------------------------------
# Validation (reads only what is on disk)
# --------------------------------------------------------------------------
def check_leakage(splits: dict[str, list[int]], samples: list[dict]) -> dict:
    """Leakage checks over base ids and samples. ``splits`` maps split -> base ids."""
    sets = {s: set(v) for s, v in splits.items()}
    pair_overlap = {f"{a}&{b}": sorted(sets[a] & sets[b]) for i, a in enumerate(SPLIT_NAMES)
                    for b in SPLIT_NAMES[i + 1:] if sets[a] & sets[b]}
    owner = {b: s for s, ids in splits.items() for b in ids}
    wrong = [r["sample_id"] for r in samples if owner.get(r["base_id"]) != r["split"]]
    by_split_bases = {s: {r["base_id"] for r in samples if r["split"] == s} for s in SPLIT_NAMES}
    cross = [f"{a}&{b}" for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]
             if by_split_bases[a] & by_split_bases[b]]
    return dict(base_id_overlap=pair_overlap, samples_in_wrong_split=wrong,
                splits_sharing_samples_bases=cross, passed=not pair_overlap and not wrong and not cross)


def min_pairwise_chamfer(skulls: list[np.ndarray], n_pts: int = 1500, seed: int = 0) -> float:
    """Smallest symmetric Chamfer distance (mm) between any two skulls (vertex subsets, any topology)."""
    rng = np.random.default_rng(seed)
    pts = [s[rng.choice(len(s), min(n_pts, len(s)), replace=False)] for s in skulls]
    trees = [cKDTree(p) for p in pts]
    best = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = min(best, 0.5 * (trees[j].query(pts[i])[0].mean() + trees[i].query(pts[j])[0].mean()))
    return float(best)


def validate_dataset(root: str | Path, cfg: dict) -> dict:
    """Recompute counts, statistics and integrity checks from the files on disk."""
    root = Path(root)
    samples = [json.loads(l) for l in (root / "metadata/samples.jsonl").read_text().splitlines()]
    splits = {s: json.loads((root / "splits" / f"{s}.json").read_text())["base_ids"] for s in SPLIT_NAMES}
    bins, gcfg = cfg["missingness_bins"], cfg["generator"]
    base_ids = sorted({r["base_id"] for r in samples})
    source = samples[0]["skull_source"]

    skulls = [mu.load_mesh(str(root / f"raw/generated/cranial_meshes/{n}.ply"))[0] for n in map(base_name, base_ids)]
    faces = np.stack([mu.load_mesh(str(root / f"raw/generated/facial_meshes/{n}.ply"))[0] for n in map(base_name, base_ids)])
    ext_s, ext_f = np.stack([np.ptp(s, axis=0) for s in skulls]), np.ptp(faces, axis=1)
    corr = np.load(root / "processed/correspondence/correspondence.npz")
    corr_ok = bool(all(np.allclose(faces[i], np.load(root / f"processed/facial_targets/{base_name(b)}.npz")["face_verts"], atol=1e-3)
                       for i, b in enumerate(base_ids)) and corr["face_faces"].max() + 1 <= faces.shape[1])

    hashes = {hashlib.sha256(np.round(s, 3).tobytes()).hexdigest() for s in skulls}
    dup = dict(source=source)
    if source == "procedural":                                   # shared topology: per-vertex RMS and parameter distance
        flat = np.stack([s.reshape(-1) for s in skulls])
        sq = (flat ** 2).sum(1)
        rms = np.sqrt(np.maximum((sq[:, None] + sq[None] - 2 * flat @ flat.T) / skulls[0].shape[0], 0))
        off = ~np.eye(len(skulls), dtype=bool)
        pv = np.stack([gen.param_vector(json.loads((root / f"raw/generated/paired_samples/{base_name(b)}.json").read_text())["generator_params"], gcfg)
                       for b in base_ids])
        lo = np.array([r[0] for g in ("skull", "face") for r in gcfg[g].values()])
        hi = np.array([r[1] for g in ("skull", "face") for r in gcfg[g].values()])
        dist_p = np.linalg.norm(((pv - lo) / (hi - lo))[:, None] - ((pv - lo) / (hi - lo))[None], axis=-1)
        dup.update(min_pairwise_param_distance_normalised=float(dist_p[off].min()) if off.any() else None,
                   min_pairwise_skull_rms_mm=float(rms[off].min()) if off.any() else None)
        no_near_dup = bool(not off.any() or rms[off].min() >= gcfg["min_pairwise_skull_rms_mm"] - 1e-6)
    else:                                                        # real scans: topology differs, use Chamfer
        mc = min_pairwise_chamfer(skulls) if len(skulls) > 1 else float("inf")
        dup.update(min_pairwise_skull_chamfer_mm=mc)
        no_near_dup = bool(mc > 0.5)                             # identical scans would be ~0.1 mm

    sev_stats = {}
    for sev in bins:
        m = np.array([r["observation"]["measured_missing_fraction"] for r in samples if r["observation"]["severity"] == sev])
        sev_stats[sev] = dict(count=int(len(m)), bin=bins[sev], measured_min=float(m.min()) if len(m) else None,
                              measured_mean=float(m.mean()) if len(m) else None,
                              measured_max=float(m.max()) if len(m) else None,
                              all_in_bin=bool(len(m) == 0 or (m.min() >= bins[sev][0] and m.max() <= bins[sev][1])))
    npts = np.array([r["observation"]["n_points"] for r in samples])
    pat = {p: sum(r["observation"]["pattern"] == p for r in samples) for p in pc.PATTERNS}
    leak = check_leakage(splits, samples)
    files_ok = all((root / r["pointcloud"]).exists() for r in samples)
    unit_ok = bool(((ext_s[:, 0] > 100) & (ext_s[:, 0] < 200) & (ext_s[:, 1] > 140) & (ext_s[:, 1] < 240)
                    & (ext_s[:, 2] > 100) & (ext_s[:, 2] < 260)).all()) and all(r["units"] == "mm" for r in samples)
    checks = dict(leakage=leak, files_present=files_ok, correspondence_consistent=corr_ok,
                  no_near_duplicate_skulls=no_near_dup, face_narrower_than_skull=bool((ext_f[:, 0] < ext_s[:, 0]).all()),
                  units_plausible_mm=unit_ok,
                  provenance_flags_present=all(r["synthetic"] and r["face_synthetic"] and r["provenance"] for r in samples),
                  no_exact_duplicate_skulls=len(hashes) == len(skulls),
                  all_severity_bins_respected=all(v["all_in_bin"] for v in sev_stats.values()))
    real = bool(samples[0]["real_skull"])
    return dict(
        synthetic=True, real_skull=real, face_synthetic=True, provenance=samples[0]["provenance"], units="mm",
        counts=dict(independent_base_geometries=len(base_ids), observations=len(samples),
                    base_per_split={s: len(splits[s]) for s in SPLIT_NAMES},
                    observations_per_split={s: sum(r["split"] == s for r in samples) for s in SPLIT_NAMES}),
        missingness=sev_stats, patterns=pat,
        point_density=dict(counts={int(k): int((npts == k).sum()) for k in np.unique(npts)},
                           noise_sigma_mm_mean=float(np.mean([r["observation"]["noise_sigma_mm"] for r in samples]))),
        bounding_box_mm=dict(skull_extent_min=ext_s.min(0).tolist(), skull_extent_max=ext_s.max(0).tolist(),
                             face_extent_min=ext_f.min(0).tolist(), face_extent_max=ext_f.max(0).tolist()),
        duplicates=dup, checks=checks,
        all_checks_passed=bool(all((v["passed"] if isinstance(v, dict) else v) for v in checks.values())))


def make_plots(root: Path, records: list[dict]) -> None:
    """Missingness / point-count histograms and one example, saved to visualizations/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist([r["observation"]["measured_missing_fraction"] * 100 for r in records], bins=30)
    ax[0].set(xlabel="measured missing area (%)", ylabel="observations", title="Missingness")
    ax[1].hist([r["observation"]["n_points"] for r in records], bins=[512, 1536, 3072, 4608])
    ax[1].set(xlabel="points per cloud", title="Point density")
    fig.tight_layout(); fig.savefig(out / "missingness_density.png", dpi=120); plt.close(fig)
    r = records[0]
    d = np.load(root / r["pointcloud"])
    sk, _ = mu.load_mesh(str(root / f"raw/generated/cranial_meshes/{base_name(r['base_id'])}.ply"))
    fc = np.load(root / r["face_target"])["face_verts"]
    tag = "real skull + synthetic face" if r["real_skull"] else "synthetic skull + face"
    fig = plt.figure(figsize=(10, 4))
    for j, (title, pts, col) in enumerate([(f"complete skull + face target ({tag})", None, None),
                                           (f"observed input ({r['observation']['measured_missing_fraction']*100:.0f}% missing)", d["points"], "k")]):
        a = fig.add_subplot(1, 2, j + 1, projection="3d")
        if pts is None:
            a.scatter(*sk[::max(1, len(sk) // 4000)].T, s=1, c="gray"); a.scatter(*fc.T, s=1, c="tab:red")
        else:
            a.scatter(*pts.T, s=1, c=col)
        a.set_title(title, fontsize=9); a.view_init(15, 70); a.set_box_aspect((1, 1.2, 0.9))
    fig.savefig(out / "example_sample.png", dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--out", default="UNCERTAFACE_DATA")
    ap.add_argument("--n-base", type=int)
    ap.add_argument("--obs", type=int)
    ap.add_argument("--source", choices=["procedural", "mug500"])
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    rep = build_dataset(cfg, a.out, a.n_base, a.obs, source=a.source)
    print(json.dumps({k: rep[k] for k in ("counts", "all_checks_passed")}, indent=2))


if __name__ == "__main__":
    main()
