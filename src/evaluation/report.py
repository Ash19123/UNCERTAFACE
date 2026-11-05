"""Generate ``reports/RESULTS.md`` from saved JSON results. No number is typed by hand.

  python -m src.evaluation.report --main probabilistic_scratch [--ensemble ensemble3]

Sections whose source files do not exist are written as "not run", so the report can never
show a result that was not produced. Narrative interpretation lives in ``reports/FINDINGS.md``
(written by a person from these tables) and is appended verbatim if present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _md(rows, cols, fmt="{:.2f}"):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def build_report(main: str, ensemble: str | None = None, reports="reports", data_root="UNCERTAFACE_DATA_REAL") -> str:
    R = Path(reports)
    L = ["# UNCERTAFACE results (real skulls, SYNTHETIC faces)", "",
         "> Every number below is read from saved JSON produced by the code in this repository. Skulls are real "
         "(MUG500+, CC BY 4.0); the face attached to each is SYNTHETIC (an assumed soft-tissue rule). Errors are "
         "measured against that synthetic face, so they say nothing about real faces, forensic validity or identification. "
         "All distances/uncertainties are in millimetres.", ""]
    d = _load(Path(data_root) / "dataset_report.json")
    if d:
        c = d["counts"]
        L += ["## Dataset", f"- version {d.get('dataset_version')}, config sha256 `{str(d.get('config_sha256'))[:12]}`",
              f"- independent base geometries: **{c['independent_base_geometries']}**; observations: **{c['observations']}**",
              f"- bases per split: {c['base_per_split']}; observations per split: {c['observations_per_split']}",
              f"- all integrity checks passed: **{d['all_checks_passed']}**",
              "- measured missingness (mean area fraction): " + ", ".join(f"{k} {v['measured_mean']:.2f}" for k, v in d["missingness"].items()), ""]
    o = _load(R / "oracle_floor.json")
    L += ["## Irreducible-error floor (generator property)"]
    L += [f"- complete noise-free skull, face latents redrawn: floor vertex RMSE = **{o['floor_rmse_mm']:.2f} mm** "
          f"(oracle positional std: mean {o['oracle_pos_std_mm']['mean']:.2f}, range {o['oracle_pos_std_mm']['min']:.2f}-"
          f"{o['oracle_pos_std_mm']['max']:.2f} mm)"] if o else ["- not run"]
    L.append("")
    b = _load(R / "baselines.json")
    L += ["## Reference baseline (no skull evidence)"]
    if b:
        t = b["mean_template"]["test"]
        L += ["- mean training face for every sample, test split (mm): " + ", ".join(f"{k} {v['mean']:.2f}" for k, v in t.items())]
    else:
        L += ["- not run"]
    L.append("")
    r = _load(R / "results" / main / "results.json")
    if not r:
        return "\n".join(L + [f"## Main model `{main}`", "not evaluated yet"])
    L += [f"## Main model `{main}` on the {r['eval_split']} split (n = {r['n_test']} observations)", "### Reconstruction (mm)",
          _md([dict(metric=k, mean=v["mean"], std=v["std"]) for k, v in r["reconstruction"].items()], ["metric", "mean", "std"]), "",
          "Vertex metrics use known correspondence; Chamfer/Hausdorff treat vertices as point sets; point-to-surface/surface error "
          "use distances to the other mesh's triangles.", "", "### By missingness severity (vertex RMSE, mm)",
          _md([dict(severity=k, rmse=v["vertex_rmse"]["mean"], chamfer=v["chamfer"]["mean"], surface_error=v["surface_error"]["mean"])
               for k, v in r["reconstruction_by_severity"].items()], ["severity", "rmse", "chamfer", "surface_error"]), ""]
    mr = r["missingness_relationship"]
    L += ["### Does uncertainty respond to missingness?", "- " + "; ".join(f"{k} = {v:.3f}" for k, v in mr.items()), ""]
    if r["probabilistic"]:
        u = r["uncertainty_raw"]
        L += ["### Prediction intervals: raw vs conformal-calibrated (test targets never used for fitting)",
              f"- raw CRPS = {u['crps_mm']:.3f} mm, NLL = {u['nll']:.3f}, standardised-residual mean/std = {u['z_mean']:.3f}/{u['z_std']:.3f}, "
              f"raw calibration error (mean |gap|) = {u['coverage']['ece_coverage']:.3f}", ""]
        rows = []
        for lvl in ("50", "80", "90", "95"):
            raw = r["calibration"]["raw"][lvl]
            row = dict(nominal=f"{lvl}%", raw_pointwise=raw["pointwise"], raw_width_mm=raw["mean_half_width_mm"])
            for sc in ("global", "per_coordinate", "per_group"):
                row[f"{sc}_pointwise"] = r["calibration"][sc]["test"][lvl]["pointwise"]
            row["global_width_mm"] = r["calibration"]["global"]["test"][lvl]["mean_half_width_mm"]
            row["raw_simultaneous"] = raw["simultaneous"]
            row["subject_max_simultaneous"] = r["calibration"]["subject_max"]["test"][lvl]["simultaneous"]
            rows.append(row)
        L += [_md(rows, list(rows[0]), "{:.3f}"), "",
              "Pointwise = fraction of coordinates inside their interval (marginal). Simultaneous = fraction of test samples whose "
              "EVERY vertex/axis is inside; only `subject_max` calibration targets it, and with few calibration bases the highest levels become "
              f"conservative or infinite (calibration observations: {r['n_calibration']}).", "",
              "### Sparsification and risk-coverage",
              "- " + "; ".join(f"{k} = {v:.3f}" for k, v in r["sparsification"].items()),
              "- risk (mean error, mm) at retained coverage: " + ", ".join(f"{k}: {v:.2f}" for k, v in r["risk_coverage"].items()),
              f"- best-of-{r['best_of_n']['n_samples']} (oracle-selected, NOT a single-prediction result): single {r['best_of_n']['single_prediction_rmse_mm']:.2f} mm, "
              f"mean sampled {r['best_of_n']['mean_sampled_rmse_mm']:.2f} mm, best-of-N {r['best_of_n']['best_of_n_rmse_mm']:.2f} mm", ""]
    for name, rows in r["stress_grouped"].items():
        L += [f"### Test-set breakdown by {name}", _md(rows, [c for c in ("group", "n", "rmse_mm", "mean_pos_std_mm", "cov90_raw") if c in rows[0]]), ""]
    for name, rows in r.get("stress_controlled", {}).items():
        L += [f"### Controlled sweep: {name}", _md(rows, [c for c in ("group", "n", "rmse_mm", "mean_pos_std_mm", "cov90_raw") if c in rows[0]]), ""]
    z = _load(R / "results" / main / "zone_certainty.json")
    if z:
        L += ["### Which face regions are most certain (test split)",
              _md([dict(region=r["zone"], mean_halfwidth_mm=r["mean_halfwidth_mm"], mean_error_mm=r["mean_error_mm"]) for r in z["zones"]], ["region", "mean_halfwidth_mm", "mean_error_mm"], "{:.2f}"), "",
              f"- rank correlation of region certainty vs real error: {z['zone_rank_spearman_halfwidth_vs_error']:.2f} over {len(z['zones'])} regions; per-vertex Spearman {z['vertex_spearman_halfwidth_vs_error']:.2f}",
              f"- error in the 25% most / least certain points: {z['error_in_25pct_most_certain_mm']:.2f} / {z['error_in_25pct_least_certain_mm']:.2f} mm (all points {z['error_all_mm']:.2f} mm)",
              f"- shown per-coordinate half-widths contain {z['coverage_per_coordinate_inside_halfwidth']*100:.1f}% of true coordinates (nominal 90%)",
              f"- plausible faces (K={z['k']}, smoothing {z['corr_mm']} mm, an assumption): mean sample RMSE {z['samples']['mean_sample_rmse_mm']:.2f} mm vs mean prediction {z['samples']['mean_prediction_rmse_mm']:.2f} mm; best-of-K (oracle-selected) {z['samples']['best_of_k_rmse_mm']:.2f} mm", ""]
    L += [f"Figures: `reports/results/{main}/*.png`", ""]
    ab = Path(R / "ablation_table.md")
    L += ["## Ablations", ab.read_text() if ab.exists() else "not run", ""]
    if ensemble:
        e = _load(R / "results" / ensemble / "results.json")
        L += [f"## Deep ensemble `{ensemble}` ({e['n_members']} members)" if e else "## Ensemble", "not run" if not e else
              f"- single members' RMSE (mm): {e['single_member_rmse_mm']}; ensemble mean-prediction RMSE: {e['ensemble_reconstruction']['vertex_rmse']['mean']:.2f} mm",
              ""]
        if e:
            L += [_md([dict(kind=k, **v) for k, v in e["uncertainty"].items()], ["kind", "mean_std_mm", "cov90_raw", "crps_mm", "pooled_spearman_err_vs_std", "normalised_sparsification_error", "aurc_mm"], "{:.3f}"),
                  f"- epistemic share of total variance: {e['epistemic_share_of_total_variance']:.3f}", ""]
    f = Path(R / "FINDINGS.md")
    if f.exists():
        L += ["", f.read_text()]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--ensemble")
    a = ap.parse_args()
    txt = build_report(a.main, a.ensemble)
    Path("reports/RESULTS.md").write_text(txt, encoding="utf-8")
    print(f"wrote reports/RESULTS.md ({len(txt)} chars)")


if __name__ == "__main__":
    main()
