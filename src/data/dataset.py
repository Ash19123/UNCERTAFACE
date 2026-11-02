"""PyTorch Dataset for UNCERTAFACE (synthetic data).

Each item is a dict of tensors. ``*_mm`` entries are physical millimetres;
unsuffixed coordinate entries are normalised with the train-fitted
``Normalizer``. Predictions in normalised units must be mapped back with the
same normaliser (see ``preprocessing.Normalizer``) before being reported.

Item keys
  points        (n,3)  normalised input cloud (zero rows are padding)
  mask          (n,)   bool, True for real points
  knn_dist      (n,1)  mean distance to the k nearest real neighbours, normalised units
                       (local sparsity cue; 0 on padding)
  missing_fraction ()  measured missing area fraction (ground truth from the generator;
                       for real data this would have to be estimated)
  face          (V,3)  normalised target;  face_mm (V,3) the same in mm
  vertex_visible (Vs,) bool, skull vertices inside the observed region (for visualisation/analysis)
  sample_id, base_id
``use_missing_info=False`` zeroes ``missing_fraction`` and ``knn_dist`` so the
missingness ablation uses exactly the same pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import Augmenter
from .preprocessing import Normalizer, knn_indices, load_sample, to_fixed_size


def read_records(root, split: str) -> list[dict]:
    """Sample records of one split (filtered from metadata/samples.jsonl)."""
    lines = (Path(root) / "metadata/samples.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["split"] == split]


def fit_template(root, norm: Normalizer, base_ids) -> np.ndarray:
    """Mean face (V,3) in NORMALISED units over the given bases (use train only)."""
    faces = [np.load(Path(root) / f"processed/facial_targets/B{b:04d}.npz")["face_verts"] for b in base_ids]
    return norm.forward(np.mean(faces, axis=0)).astype(np.float32)


class UncertafaceDataset(Dataset):
    """Observation -> face-target pairs of one split.

    ``augment`` is only honoured when an ``aug_cfg`` is given. Randomness is
    deterministic: item ``i`` in epoch ``e`` uses ``SeedSequence([seed, i, e])``
    (``set_epoch`` changes augmentation between epochs; evaluation ignores it).
    """

    def __init__(self, root, split: str, norm: Normalizer, n_points: int = 1024, knn_k: int = 8,
                 aug_cfg: dict | None = None, augment: bool = False, use_missing_info: bool = True,
                 seed: int = 0):
        self.root, self.norm = Path(root), norm
        self.records = read_records(root, split)
        if not self.records:
            raise ValueError(f"no samples in split {split!r}")
        self.n_points, self.knn_k, self.use_missing_info, self.seed, self.epoch = n_points, knn_k, use_missing_info, seed, 0
        self.augmenter = Augmenter(aug_cfg) if (augment and aug_cfg and aug_cfg.get("enabled", True)) else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        s = load_sample(self.root, rec)                       # mm, float32
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, i, self.epoch if self.augmenter else 0]))
        pts_mm, face_mm, miss = s["points_mm"], s["face_mm"], s["missing_fraction"]
        if self.augmenter is not None:
            pts_mm, face_mm, info = self.augmenter(pts_mm, face_mm, miss, rng)
            miss = info["missing_fraction"]
        pts, mask = to_fixed_size(self.norm.forward(pts_mm).astype(np.float32), self.n_points, rng)
        k = min(self.knn_k, int(mask.sum()))
        nn = knn_indices(pts, k, mask)
        knn_dist = np.linalg.norm(pts[nn] - pts[:, None], axis=-1).mean(1, keepdims=True).astype(np.float32)
        knn_dist[~mask] = 0.0
        if not self.use_missing_info:
            knn_dist[:] = 0.0
            miss = 0.0
        face_mm = np.asarray(face_mm, dtype=np.float32)
        return dict(points=torch.from_numpy(pts), mask=torch.from_numpy(mask),
                    knn_dist=torch.from_numpy(knn_dist),
                    missing_fraction=torch.tensor(miss, dtype=torch.float32),
                    face=torch.from_numpy(self.norm.forward(face_mm).astype(np.float32)),
                    face_mm=torch.from_numpy(face_mm),
                    vertex_visible=torch.from_numpy(s["vertex_visible"]),
                    sample_id=rec["sample_id"], base_id=rec["base_id"])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0) -> DataLoader:
    """DataLoader with a seeded shuffle. Items have a fixed size, so default collate applies."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, drop_last=False)
