"""Reference baselines that use no skull evidence.

``mean_template``: predict the training-set mean face for every sample. Any useful model must beat it, and the
gap between it and the irreducible floor (``oracle.py``) is the room a skull-conditioned model has to improve.

  python -m src.evaluation.baselines            # writes reports/baselines.json (validation and test)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import dataset as ds
from ..data import preprocessing as pp
from . import reconstruction_metrics as rm


def mean_template_metrics(root="UNCERTAFACE_DATA_REAL") -> dict:
    root = Path(root)
    nfile = root / "metadata/normalizer.json"
    if not nfile.exists():                       # the trainer saves one per run; here fit on the train split
        pp.fit_normalizer_from_train(root).save(nfile)
    norm = pp.Normalizer.load(nfile)
    tpl = norm.inverse(ds.fit_template(root, norm, pp.read_split(root, "train")["base_ids"]))
    faces = np.load(root / "processed/correspondence/correspondence.npz")["face_faces"]
    out = {}
    for split in ("validation", "test"):
        d = ds.UncertafaceDataset(root, split, norm, cache=True)
        gt = np.stack([d[i]["face_mm"].numpy() for i in range(len(d))])
        pred = np.broadcast_to(tpl, gt.shape)
        out[split] = rm.summarise(rm.evaluate_batch(pred, gt, faces))
    return out


def main() -> None:
    res = dict(mean_template=mean_template_metrics(),
               note="training-set mean face for every sample; uses no skull evidence")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baselines.json").write_text(json.dumps(res, indent=2))
    t = res["mean_template"]["test"]
    print({k: round(v["mean"], 3) for k, v in t.items()})


if __name__ == "__main__":
    main()
