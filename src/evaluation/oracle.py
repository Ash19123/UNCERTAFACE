"""Irreducible-error floor of the SYNTHETIC task, computed from the generator.

Face = skull surface displaced along its normal by a thickness
``t = f(position, bone-independent latents)``. A predictor that sees the COMPLETE,
noise-free skull knows everything about the bone; what it cannot know are the
face latents (thickness scale/field, nose, lips, cheeks, unobserved field).
Holding the skull fixed and redrawing only those latents therefore gives, per
face vertex, the unavoidable predictive variance of this synthetic world:

    floor_rmse = sqrt( mean_v  sum_k Var_latents[ y_{v,k} | skull ] )
    oracle_pos_std(v) = sqrt( sum_k Var[ y_{v,k} | skull ] )

No real-anatomy claim: this characterises the generator only. Partial inputs
cannot beat this floor.
"""
from __future__ import annotations

import numpy as np

from ..data import generator as gen

FACE_LATENTS = ("thickness_scale", "thickness_field_amp", "nose_projection", "lip_fullness",
                "cheek_fat", "unobserved_field_amp", "thickness_field_seed", "unobserved_field_seed")


def oracle_variance_real(skull_files, cfg: dict, n_draws: int = 150, seed: int = 0) -> dict:
    """Same floor for REAL skulls: hold each skull (so its face base surface) fixed and redraw only the face latents.

    ``skull_files``: prepared ``A####.npz`` paths (verts mm, faces, scale). Returns ``var (S,V,3)`` in mm,
    pooled ``floor_rmse_mm`` and per-vertex ``pos_std_mm (S,V)``.
    """
    from ..data import face_from_skull as ffs
    from ..data import generator as g
    from ..data.geometry_sources import FACE_KEYS
    rng = np.random.default_rng(seed)
    out = []
    for f in skull_files:
        d = np.load(f)
        s = float(d["scale"])
        base = ffs.envelope_base(s * d["verts"].astype(float))
        faces = []
        for _ in range(n_draws):
            p = {k: v for k, v in g.sample_params(rng, cfg).items() if k in FACE_KEYS}
            faces.append(ffs.apply_thickness(base, p, cfg["min_thickness_mm"])["face"] / s)
        out.append(np.var(np.stack(faces), axis=0))
    var = np.stack(out)
    return dict(var=var, floor_rmse_mm=float(np.sqrt(var.sum(-1).mean())), pos_std_mm=np.sqrt(var.sum(-1)))


def main() -> None:
    """Compute the floor on the validation skulls and save ``reports/oracle_floor.json``.

    ``python -m src.evaluation.oracle [data_root] [dataset_config]`` (defaults: the real-skull dataset).
    """
    import json
    import sys
    from pathlib import Path

    import yaml
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "UNCERTAFACE_DATA_REAL")
    cfg_all = yaml.safe_load(Path(sys.argv[2] if len(sys.argv) > 2 else "configs/dataset_mug500.yaml").read_text())
    cfg = cfg_all["generator"]
    ids = json.loads((root / "splits/validation.json").read_text())["base_ids"]
    metas = [json.loads((root / f"raw/generated/paired_samples/B{b:04d}.json").read_text()) for b in ids]
    if metas[0]["skull_source"] == "mug500":
        proc = Path(cfg_all["mug500"]["processed_dir"])
        files = [proc / f"{m['extra']['case']}.npz" for m in metas]
        o = oracle_variance_real(files, cfg, n_draws=150, seed=1)
    else:
        o = oracle_variance([m["generator_params"] for m in metas], cfg, n_draws=150, seed=1)
    s = o["pos_std_mm"]
    out = dict(floor_rmse_mm=o["floor_rmse_mm"], n_skulls=len(ids), n_draws=150, split="validation",
               source=metas[0]["skull_source"],
               oracle_pos_std_mm=dict(mean=float(s.mean()), min=float(s.min()), max=float(s.max()),
                                      std_across_subjects_at_vertex=float(s.std(0).mean()),
                                      std_across_vertices=float(s.mean(0).std())),
               per_vertex_mean_pos_std_mm=s.mean(0).tolist(),
               note="property of this project's synthetic face model given a complete noise-free skull; not a statement about real anatomy")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/oracle_floor.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("floor_rmse_mm", "oracle_pos_std_mm")}, indent=2))


def oracle_variance(base_params: list[dict], cfg: dict, n_draws: int = 200, seed: int = 0,
                    subdivisions: int = 4) -> dict:
    """Monte-Carlo per-vertex 3D variance of the face given the skull, over face latents.

    ``base_params``: generator parameter dicts of skulls to hold fixed.
    Returns ``var (S,V,3)`` per skull/vertex/axis and the pooled floor RMSE (mm).
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in base_params:
        faces = []
        for _ in range(n_draws):
            q = dict(p)
            draw = gen.sample_params(rng, cfg)
            for k in FACE_LATENTS:
                q[k] = draw[k]
            faces.append(gen.build_subject(0, q, cfg, subdivisions).face_verts)
        out.append(np.var(np.stack(faces), axis=0))               # (V,3)
    var = np.stack(out)
    return dict(var=var, floor_rmse_mm=float(np.sqrt(var.sum(-1).mean())),
                pos_std_mm=np.sqrt(var.sum(-1)))                  # (S,V)


if __name__ == "__main__":
    main()
