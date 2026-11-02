"""Ablation study driver: train variants from scratch under one shared protocol, then tabulate.

Variants (all share seed 0, epochs, patience and selection metric unless the variant changes it):
  full               probabilistic model, all loss terms, missingness info, PointNet++ encoder, 1024 pts
  full_seed1/2       identical to `full` with seeds 1, 2 -> a direct estimate of seed-to-seed noise,
                     and together with `full` the 3 members of the deep ensemble
  deterministic      mean-only model, no NLL ("without NLL")
  no_surface         lambda_surface = 0 (no Laplacian / edge / normal regularisation)
  no_missing_info    no missing fraction and no local-density feature
  encoder_pointnet   PointNet baseline encoder instead of hierarchical PointNet++
  points_2048/4096   larger input clouds
"Without calibration" is an evaluation-time ablation (raw vs conformal intervals) reported in the
same table. Checkpoints are selected on validation RMSE for ALL variants so they are comparable.
Differences smaller than the full-vs-full_seed1 gap must not be read as effects.

  python -m src.evaluation.ablation run --epochs 60
  python -m src.evaluation.ablation table
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..training.trainer import Trainer, load_config
from .run_evaluation import evaluate_run

VARIANTS: dict[str, list[str]] = {
    "full": [],
    "full_seed1": ["run.seed=1"],
    "full_seed2": ["run.seed=2"],        # 3 seeds of `full` double as the deep-ensemble members
    "deterministic": ["model.probabilistic=false", "loss.lambda_nll=0.0"],
    "no_surface": ["loss.lambda_surface=0.0"],
    "no_missing_info": ["data.use_missing_info=false", "model.use_missing_info=false"],
    "encoder_pointnet": ["model.encoder.name=pointnet"],
    "points_2048": ["data.n_points=2048"],
    "points_4096": ["data.n_points=4096"],
}
PREFIX = "abl_"


def run_variants(epochs: int, names=None, out_dir="experiments/runs", log=print) -> None:
    """Train every missing variant; a variant whose experiment.json exists is skipped (resume-safe)."""
    for name in names or VARIANTS:
        run = Path(out_dir) / f"{PREFIX}{name}"
        if (run / "experiment.json").exists() and json.loads((run / "experiment.json").read_text())["status"] in ("completed", "early_stopped"):
            log(f"skip {name}: already finished")
            continue
        ov = [f"run.name={PREFIX}{name}", f"run.out_dir={out_dir}", f"optim.epochs={epochs}",
              f"early_stopping.patience={epochs}", "early_stopping.monitor=val_rmse_mm"] + VARIANTS[name]
        log(f"=== training {name} ({epochs} epochs) ===")
        Trainer(load_config("configs/training.yaml", "configs/model.yaml", ov)).fit(log=log)


def _f(x, d=3):
    return "" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def table(out_dir="experiments/runs", split="test", device="cpu", reports="reports") -> list[dict]:
    """Evaluate each finished variant (calibration split -> calibrators, ``split`` -> numbers)."""
    rows = []
    for name in VARIANTS:
        run = Path(out_dir) / f"{PREFIX}{name}"
        if not (run / "experiment.json").exists():
            continue
        exp = json.loads((run / "experiment.json").read_text())
        r = evaluate_run(run, Path(reports) / "results" / f"{PREFIX}{name}", split, device, sweeps=False)
        rc = r["reconstruction"]
        row = dict(variant=name, status=exp["status"], params=exp["model_params"], epochs_run=exp["epochs_run"],
                   best_epoch=exp["best_epoch"], val_rmse_mm=exp["best_validation_metrics"]["val_rmse_mm"],
                   test_vertex_rmse_mm=rc["vertex_rmse"]["mean"], test_chamfer_mm=rc["chamfer"]["mean"],
                   test_surface_error_mm=rc["surface_error"]["mean"], test_hausdorff_mm=rc["hausdorff"]["mean"])
        for sv in ("mild", "moderate", "severe", "extreme"):
            row[f"rmse_{sv}"] = r["reconstruction_by_severity"].get(sv, {}).get("vertex_rmse", {}).get("mean")
        if r["probabilistic"]:
            u, c = r["uncertainty_raw"], r["calibration"]
            row.update(crps_mm=u["crps_mm"], nll=u["nll"], ece_raw=u["coverage"]["ece_coverage"],
                       cov90_raw=u["coverage"]["90"]["observed"],
                       cov90_global=c["global"]["test"]["90"]["pointwise"],
                       half_width90_raw=u["coverage"]["90"]["mean_half_width_mm"],
                       half_width90_global=c["global"]["test"]["90"]["mean_half_width_mm"],
                       sim95_subject_max=c["subject_max"]["test"]["95"]["simultaneous"],
                       sparsification_norm_se=r["sparsification"]["normalised_se"],
                       aurc_mm=r["sparsification"]["aurc_mm"])
        rows.append(row)
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    Path(reports).mkdir(parents=True, exist_ok=True)
    with open(Path(reports) / "ablation_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(_f(r.get(c)) for c in cols) + " |" for r in rows]
    (Path(reports) / "ablation_table.md").write_text("\n".join(md))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run_variants(a.epochs, a.only) if a.cmd == "run" else table(device=a.device)


if __name__ == "__main__":
    main()
