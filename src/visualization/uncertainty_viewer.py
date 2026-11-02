"""Matplotlib plots for uncertainty evaluation. Every axis that carries uncertainty or error
is labelled in millimetres; nothing is decorative colour."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    _plt().close(fig)


def reliability_plot(curves: dict[str, tuple], path, title="Reliability of prediction intervals") -> None:
    """``curves``: label -> (nominal levels, observed pointwise coverage)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for label, (nom, obs) in curves.items():
        ax.plot(nom, obs, marker="o", ms=3, label=label)
    ax.set(xlabel="nominal coverage", ylabel="observed pointwise coverage", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    _save(fig, path)


def sparsification_plot(curves: dict, path, title="Sparsification") -> None:
    """``curves`` from ``sparsification.sparsification_curves`` (mm)."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    f = curves["removed_fraction"] * 100
    ax[0].plot(f, curves["uncertainty"], label="ranked by predicted uncertainty")
    ax[0].plot(f, curves["oracle"], label="oracle (ranked by true error)")
    ax[0].plot(f, curves["random"], "--", label="random")
    ax[0].set(xlabel="most uncertain locations removed (%)", ylabel="mean error of retained locations (mm)",
              title=f"{title}\nnormalised SE = {curves['normalised_se']:.2f}")
    ax[0].legend(fontsize=8)
    cov = 100 - f
    ax[1].plot(cov, curves["uncertainty"], label="by uncertainty")
    ax[1].plot(cov, curves["random"], "--", label="random")
    ax[1].invert_xaxis()
    ax[1].set(xlabel="coverage: retained locations (%)", ylabel="risk: mean error (mm)",
              title=f"Risk-coverage (AURC {curves['aurc_mm']:.2f} mm)")
    ax[1].legend(fontsize=8)
    _save(fig, path)


def error_vs_uncertainty_plot(err: np.ndarray, unc: np.ndarray, path, n_bins: int = 20) -> None:
    """Binned mean error vs predicted positional std (mm). Perfect magnitude calibration lies near y = x*c."""
    plt = _plt()
    err, unc = np.asarray(err).ravel(), np.asarray(unc).ravel()
    edges = np.quantile(unc, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(unc, edges[1:-1]), 0, n_bins - 1)
    mu_u = np.array([unc[idx == i].mean() for i in range(n_bins)])
    mu_e = np.array([err[idx == i].mean() for i in range(n_bins)])
    rmse = np.array([np.sqrt((err[idx == i] ** 2).mean()) for i in range(n_bins)])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(mu_u, rmse, "o-", label="RMS error in bin")
    ax.plot(mu_u, mu_e, "s-", label="mean error in bin")
    lim = max(mu_u.max(), rmse.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="error = predicted std")
    ax.set(xlabel="predicted positional std (mm), binned by quantile", ylabel="actual vertex error (mm)",
           title="Error vs predicted uncertainty")
    ax.legend(fontsize=8)
    _save(fig, path)


def spatial_map_plot(template_mm: np.ndarray, mean_err: np.ndarray, mean_unc: np.ndarray, path) -> None:
    """Frontal map (x-z) of per-vertex mean error and mean predicted std over a split, one mm colour scale."""
    plt = _plt()
    vmax = float(np.ceil(max(mean_err.max(), mean_unc.max())))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for a, v, t in zip(ax, (mean_err, mean_unc), ("mean error (mm)", "mean predicted uncertainty (mm)")):
        sc = a.scatter(template_mm[:, 0], template_mm[:, 2], c=v, s=14, cmap="viridis", vmin=0, vmax=vmax)
        a.set(title=t, xlabel="x (mm)", ylabel="z (mm)", aspect="equal")
    fig.colorbar(sc, ax=ax, label="mm", shrink=0.85)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_plot(tables: dict[str, list[dict]], path, metric="rmse_mm", title="") -> None:
    """One bar group per factor: ``tables`` maps factor name -> rows from stress_tests."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables), 3.4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], tables.items()):
        ax.bar([r["group"] for r in rows], [r[metric] for r in rows], color="tab:blue")
        if all("mean_pos_std_mm" in r for r in rows) and metric == "rmse_mm":
            ax.plot([r["group"] for r in rows], [r["mean_pos_std_mm"] for r in rows], "o-", color="tab:orange", label="predicted std")
            ax.legend(fontsize=7)
        ax.set(title=name, ylabel="mm")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


def training_curves(history: list[dict], path) -> None:
    """Train/val loss, validation RMSE (mm), predicted std (mm), learning rate and gradient norm."""
    plt = _plt()
    ep = [h["epoch"] for h in history if "val_loss" in h and np.isfinite(h.get("val_loss", np.nan))]
    h = [x for x in history if x["epoch"] in ep]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].plot(ep, [x["train_loss"] for x in h], label="train (augmented)")
    ax[0].plot(ep, [x["val_loss"] for x in h], label="validation")
    ax[0].set(title="loss", xlabel="epoch"); ax[0].legend(fontsize=8)
    ax[1].plot(ep, [x["val_rmse_mm"] for x in h]); ax[1].set(title="validation vertex RMSE (mm)", xlabel="epoch")
    if "val_pos_std_mm" in h[0]:
        ax[2].plot(ep, [x["val_pos_std_mm"] for x in h], label="mean predicted pos. std")
        ax[2].plot(ep, [x["val_rmse_mm"] for x in h], label="actual RMSE")
        ax[2].set(title="uncertainty vs error (mm)", xlabel="epoch"); ax[2].legend(fontsize=8)
    else:
        ax[2].axis("off")
    ax[3].semilogy(ep, [x["grad_norm_max"] for x in h], label="max grad norm")
    ax3b = ax[3].twinx(); ax3b.plot(ep, [x["lr"] for x in h], "g--"); ax3b.set_ylabel("lr")
    ax[3].set(title="gradient norm / learning rate", xlabel="epoch")
    _save(fig, path)
