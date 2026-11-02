"""Clean + align every downloaded MUG500+ skull (see src/data/real_skulls.py).

  python scripts/prepare_mug500.py [--raw external_data/mug500/raw] [--out external_data/mug500/processed]

Writes ``reference.npz`` (from case A0001) and one ``A####.npz`` per case, plus ``processed.json`` (per-case
alignment residuals, scales, crop counts). Outputs stay in external_data/ (git-ignored).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data import real_skulls as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="external_data/mug500/raw")
    ap.add_argument("--out", default="external_data/mug500/processed")
    ap.add_argument("--reference", default="A0001")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_path = raw / a.reference / f"{a.reference}_clear.stl"
    if not (out / "reference.npz").exists():
        rv, rf = rs.build_reference(ref_path)
        np.savez_compressed(out / "reference.npz", verts=rv.astype(np.float32), faces=rf.astype(np.int32))
    ref = np.load(out / "reference.npz")["verts"].astype(np.float64)
    log = json.loads((out / "processed.json").read_text()) if (out / "processed.json").exists() else {}
    for d in sorted(raw.glob("A[0-9][0-9][0-9][0-9]")):
        stl = d / f"{d.name}_clear.stl"
        if not stl.exists() or (out / f"{d.name}.npz").exists():
            continue
        try:
            r = rs.prepare_case(stl, ref)
        except ValueError as e:
            log[d.name] = dict(status="rejected", reason=str(e))
            print("REJECTED", d.name, e)
            continue
        np.savez_compressed(out / f"{d.name}.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                            scale=r["scale"], align_rms_mm=r["align_rms_mm"])
        log[d.name] = dict(status="ok", scale=r["scale"], align_rms_mm=r["align_rms_mm"], n_verts=len(r["verts"]),
                           n_cropped_verts=r["n_cropped_verts"], n_small_components=r["n_small_components"])
        print(d.name, {k: round(v, 3) if isinstance(v, float) else v for k, v in log[d.name].items()})
    (out / "processed.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
