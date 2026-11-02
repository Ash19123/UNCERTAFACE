"""Reconstruction figures from a trained run.

For each chosen sample: observed partial skull | ground-truth face | predicted
face | per-vertex error (mm) | predicted positional uncertainty (mm). Error and
uncertainty share one labelled millimetre colour scale per figure. All values
are physical millimetres (normaliser applied). Everything shown is SYNTHETIC.

Usage:
  python -m src.visualization.results --run experiments/runs/probabilistic --split validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ..data import dataset as ds
from ..data import preprocessing as pp
from ..models.uncertaface import UncertaFace
from ..uncertainty.estimation import gaussian_to_mm


def load_run(run_dir, device: str = "cpu", which: str = "best"):
    """Rebuild model, normaliser and config from a run directory and load a checkpoint."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    root = Path(cfg["run"]["data_root"])
    norm = pp.Normalizer.load(root / "metadata/normalizer.json")
    template = ds.fit_template(root, norm, pp.read_split(root, "train")["base_ids"])
    mcfg = dict(cfg["model"], use_missing_info=bool(cfg["model"]["use_missing_info"] and cfg["data"]["use_missing_info"]))
    model = UncertaFace(mcfg, template, norm.scale).to(device)
    state = torch.load(run_dir / f"{which}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval(), norm, cfg, root


@torch.no_grad()
def predict_sample(model, norm, dataset, i: int, device: str = "cpu") -> dict:
    """Run the model on one dataset item; everything returned is in millimetres."""
    it = dataset[i]
    b = {k: (v[None].to(device) if torch.is_tensor(v) else v) for k, v in it.items()}
    out = model.forward_batch(b)
    phys = gaussian_to_mm(out["mean"].float(), None if out["logvar"] is None else out["logvar"].float(), norm)
    pred, gt = phys["mean_mm"][0].cpu().numpy(), it["face_mm"].numpy()
    res = dict(sample_id=it["sample_id"], points_mm=norm.inverse(it["points"].numpy())[it["mask"].numpy()],
               gt_mm=gt, pred_mm=pred, err_mm=np.linalg.norm(pred - gt, axis=1),
               missing_fraction=float(it["missing_fraction"]))
    if "pos_std_mm" in phys:
        res["unc_mm"] = phys["pos_std_mm"][0].cpu().numpy()
        res["std_axis_mm"] = phys["std_mm"][0].cpu().numpy()
    return res


def pick_by_severity(dataset) -> dict[str, int]:
    """First dataset index of each missingness severity (mild..extreme)."""
    out: dict[str, int] = {}
    for i, r in enumerate(dataset.records):
        out.setdefault(r["observation"]["severity"], i)
    order = ["mild", "moderate", "severe", "extreme"]
    return {s: out[s] for s in order if s in out}


def reconstruction_figure(preds: dict[str, dict], out_png, title: str = "") -> dict:
    """Rows = samples (labelled by severity); returns per-sample summary numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_unc = all("unc_mm" in p for p in preds.values())
    vmax = max(max(p["err_mm"].max(), p.get("unc_mm", p["err_mm"]).max()) for p in preds.values())
    vmax = float(np.ceil(vmax))
    cols = ["observed skull (partial)", "ground truth face", "predicted face", "error |pred - gt| (mm)"]
    if has_unc:
        cols.append("predicted uncertainty (mm)")
    fig = plt.figure(figsize=(3.4 * len(cols), 3.2 * len(preds) + 0.8))
    summary = {}
    for r, (sev, p) in enumerate(preds.items()):
        allp = np.concatenate([p["gt_mm"], p["pred_mm"]])
        lim = np.stack([allp.min(0), allp.max(0)])
        mid, half = lim.mean(0), (lim[1] - lim[0]).max() / 2 * 1.05
        for c, name in enumerate(cols):
            ax = fig.add_subplot(len(preds), len(cols), r * len(cols) + c + 1, projection="3d")
            kw = dict(s=4, depthshade=False)
            if c == 0:
                sc = ax.scatter(*p["points_mm"].T, c="0.3", **kw)
                m = p["points_mm"].mean(0)
                ax.set_xlim(m[0] - 90, m[0] + 90); ax.set_ylim(m[1] - 90, m[1] + 90); ax.set_zlim(m[2] - 90, m[2] + 90)
            else:
                if c == 1:
                    ax.scatter(*p["gt_mm"].T, c="tab:blue", **kw)
                elif c == 2:
                    ax.scatter(*p["pred_mm"].T, c="tab:orange", **kw)
                else:
                    val = p["err_mm"] if c == 3 else p["unc_mm"]
                    sc = ax.scatter(*p["gt_mm"].T, c=val, cmap="viridis", vmin=0, vmax=vmax, **kw)
                ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
                ax.set_zlim(mid[2] - half, mid[2] + half)
            ax.view_init(elev=8, azim=90)               # same camera everywhere: looking at the face (+y)
            ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.text2D(-0.05, 0.5, f"{sev}\n{p['missing_fraction']*100:.0f}% missing", transform=ax.transAxes,
                          ha="right", va="center", fontsize=9)
        summary[sev] = dict(sample_id=p["sample_id"], missing_fraction=p["missing_fraction"],
                            vertex_rmse_mm=float(np.sqrt((p["err_mm"] ** 2).mean())),
                            mean_error_mm=float(p["err_mm"].mean()))
        if has_unc:
            summary[sev].update(mean_uncertainty_mm=float(p["unc_mm"].mean()),
                                spearman_err_vs_unc=float(spearmanr(p["err_mm"], p["unc_mm"])[0]))
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.6])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="viridis"), cax=cax, label="millimetres (mm)")
    fig.suptitle((title + "  " if title else "") + "SYNTHETIC data - not real anatomy", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.91, wspace=0.0, hspace=0.0, top=0.93)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, norm, cfg, root = load_run(a.run, a.device)
    d = ds.UncertafaceDataset(root, a.split, norm, cfg["data"]["n_points"], cfg["data"]["knn_k"],
                              use_missing_info=cfg["data"]["use_missing_info"], seed=cfg["run"]["seed"])
    preds = {sev: predict_sample(model, norm, d, i, a.device) for sev, i in pick_by_severity(d).items()}
    out = a.out or str(Path(a.run) / f"reconstruction_{a.split}.png")
    summ = reconstruction_figure(preds, out, title=f"{cfg['run']['name']} ({a.split})")
    (Path(out).with_suffix(".json")).write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
