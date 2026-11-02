"""Collect model predictions over a dataset split, in millimetres, with sample metadata."""
from __future__ import annotations

import numpy as np
import torch

from ..data import dataset as ds
from ..uncertainty.estimation import gaussian_to_mm


@torch.no_grad()
def predict_cloud(model, norm, points_mm, missing_fraction: float, knn_k: int = 8, device="cpu",
                  n_points: int | None = None) -> dict:
    """Predict from one raw cloud (mm). ``n_points=None`` keeps the cloud's own size (no padding).

    Mirrors the dataset path: normalise, optional fixed size, local-density feature.
    Returns the ``gaussian_to_mm`` dict with a leading batch axis of 1.
    """
    from ..data import preprocessing as pp
    pts = norm.forward(np.asarray(points_mm)).astype(np.float32)
    p, mask = pp.to_fixed_size(pts, n_points or len(pts), np.random.default_rng(0))
    nn = pp.knn_indices(p, min(knn_k, int(mask.sum())), mask)
    kd = np.linalg.norm(p[nn] - p[:, None], axis=-1).mean(1, keepdims=True).astype(np.float32)
    kd[~mask] = 0
    t = lambda x: torch.from_numpy(x)[None].to(device)
    out = model(t(p), t(mask), t(kd), torch.tensor([missing_fraction], dtype=torch.float32, device=device))
    return gaussian_to_mm(out["mean"].float(), None if out["logvar"] is None else out["logvar"].float(), norm)


@torch.no_grad()
def collect(model, norm, dataset: ds.UncertafaceDataset, device="cpu", batch_size: int = 32) -> dict:
    """Run ``model`` over ``dataset`` (no augmentation). All coordinates/uncertainties in mm.

    Returns arrays: ``mean (N,V,3)``, ``std (N,V,3)`` (None if deterministic), ``gt (N,V,3)``,
    ``missing_fraction (N,)``, plus per-sample metadata lists (``severity``, ``pattern``,
    ``n_points``, ``noise_sigma_mm``, ``sampler``, ``base_id``, ``sample_id``).
    """
    model.eval()
    means, stds, gts, miss = [], [], [], []
    for batch in ds.make_loader(dataset, batch_size, shuffle=False):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.forward_batch(b)
        phys = gaussian_to_mm(out["mean"].float(), None if out["logvar"] is None else out["logvar"].float(), norm)
        means.append(phys["mean_mm"].cpu().numpy())
        if "std_mm" in phys:
            stds.append(phys["std_mm"].cpu().numpy())
        gts.append(batch["face_mm"].numpy())
        miss.append(batch["missing_fraction"].numpy())
    o = [r["observation"] for r in dataset.records]
    return dict(mean=np.concatenate(means), std=np.concatenate(stds) if stds else None, gt=np.concatenate(gts),
                missing_fraction=np.array([x["measured_missing_fraction"] for x in o]),
                severity=[x["severity"] for x in o], pattern=[x["pattern"] for x in o],
                n_points=np.array([x["n_points"] for x in o]), noise_sigma_mm=np.array([x["noise_sigma_mm"] for x in o]),
                sampler=[x["sampler"] for x in o], base_id=np.array([r["base_id"] for r in dataset.records]),
                sample_id=[r["sample_id"] for r in dataset.records])
