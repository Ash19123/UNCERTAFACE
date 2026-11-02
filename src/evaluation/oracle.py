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


def main() -> None:
    """Compute the floor on the validation skulls and save ``reports/oracle_floor.json``."""
    import json
    from pathlib import Path

    import yaml
    cfg = yaml.safe_load(Path("configs/dataset.yaml").read_text())["generator"]
    root = Path("UNCERTAFACE_DATA")
    ids = json.loads((root / "splits/validation.json").read_text())["base_ids"]
    params = [json.loads((root / f"raw/generated/paired_samples/B{b:04d}.json").read_text())["generator_params"] for b in ids]
    o = oracle_variance(params, cfg, n_draws=150, seed=1)
    s = o["pos_std_mm"]
    out = dict(floor_rmse_mm=o["floor_rmse_mm"], n_skulls=len(ids), n_draws=150, split="validation",
               oracle_pos_std_mm=dict(mean=float(s.mean()), min=float(s.min()), max=float(s.max()),
                                      std_across_subjects_at_vertex=float(s.std(0).mean()),
                                      std_across_vertices=float(s.mean(0).std())),
               per_vertex_mean_pos_std_mm=s.mean(0).tolist(),
               note="synthetic-generator property; complete noise-free skull; not a statement about real anatomy")
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
