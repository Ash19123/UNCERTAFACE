"""Load, clean, decimate and canonically align REAL CT-derived skull meshes (MUG500+, CC BY 4.0).

These meshes are real, anonymised patient-derived skull geometry. They live in ``external_data/`` (git-ignored) and are
never redistributed by this repository. Processing steps (all deterministic):

  1. ``read_stl``            parse a binary STL into triangles (NumPy only; the file is never executed).
  2. ``decimate``            vertex-clustering decimation to ~N vertices (shared vertices, degenerate faces dropped).
  3. ``keep_main_parts``     drop small connected components (CT speckle, floating bone fragments).
  4. ``align_to_reference``  rigid + uniform-scale ICP (trimmed) onto a reference skull so every skull is expressed in
                             the same frame (+x right, +y anterior, +z superior). Real size is kept in mm; the fitted
                             scale is stored and used only to place the anatomical face grid.
  5. ``crop_neck``           remove cervical-spine remnants using the aligned frame.

Coordinates are millimetres in the aligned frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_STL_REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_stl(path) -> np.ndarray:
    """Binary STL -> triangles ``(n, 3, 3)`` float64. Raises on size mismatch (corrupt/ASCII files)."""
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        raw = f.read()
    if len(raw) != n * _STL_REC.itemsize:
        raise ValueError(f"{path}: STL size mismatch (header says {n} triangles)")
    return np.frombuffer(raw, dtype=_STL_REC, count=n)["v"].astype(np.float64)


def cluster_decimate(tri: np.ndarray, cell: float):
    """Vertex-clustering decimation with grid cell ``cell`` mm. Returns ``(verts, faces)``."""
    v = tri.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv, minlength=len(first)).astype(float)
    cv = np.stack([np.bincount(inv, weights=v[:, k], minlength=len(first)) for k in range(3)], 1) / cnt[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(np.sort(f, axis=1), axis=0) if False else f
    return cv, f


def decimate(tri: np.ndarray, target_verts: int = 30000):
    """Cluster-decimate to roughly ``target_verts`` vertices by bisection on the cell size."""
    lo, hi = 0.4, 6.0
    best = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        cv, f = cluster_decimate(tri, mid)
        used = len(np.unique(f))
        best = (cv, f, mid)
        if used > target_verts:
            lo = mid
        else:
            hi = mid
    cv, f, cell = best
    keep = np.unique(f)
    remap = -np.ones(len(cv), int)
    remap[keep] = np.arange(len(keep))
    return cv[keep], remap[f], cell


def components(faces: np.ndarray, n: int):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(a, directed=False)


def keep_main_parts(verts, faces, min_frac: float = 0.02):
    """Keep connected components holding at least ``min_frac`` of the faces; returns new (verts, faces, n_removed)."""
    ncomp, lab = components(faces, len(verts))
    fcomp = lab[faces[:, 0]]
    cnt = np.bincount(fcomp, minlength=ncomp)
    keep_c = cnt >= min_frac * len(faces)
    kf = faces[keep_c[fcomp]]
    used = np.unique(kf)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[kf], int((~keep_c).sum())


# ---------------------------------------------------------------- alignment
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform ``dst ~ s R src + t`` (Umeyama 1991). Returns ``(s, R, t)``."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    Z = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Z[2, 2] = -1
    R = U @ Z @ Vt
    s = (sig * np.diag(Z)).sum() / S.var(0).sum() if with_scale else 1.0
    return float(s), R, mu_d - s * R @ mu_s


def icp_similarity(src, dst, init_R=None, iters: int = 40, trim: float = 0.8, with_scale: bool = True):
    """Trimmed ICP of ``src`` onto ``dst`` (robust to leftover neck/speckle). Returns ``(s, R, t, rms)``."""
    tree = cKDTree(dst)
    R = np.eye(3) if init_R is None else init_R
    s, t = 1.0, dst.mean(0) - (R @ src.mean(0))
    for _ in range(iters):
        p = s * src @ R.T + t
        d, i = tree.query(p)
        k = d <= np.quantile(d, trim)
        s, R, t = umeyama(src[k], dst[i[k]], with_scale)
    p = s * src @ R.T + t
    d, _ = tree.query(p)
    return s, R, t, float(np.sqrt((np.sort(d)[: int(trim * len(d))] ** 2).mean()))


REF_TARGET_VERTS = 70000


def build_reference(stl_path, target_verts: int = REF_TARGET_VERTS):
    """Reference skull in the canonical frame: +y anterior (CT scan rotated 180 deg about z), origin at the
    bbox mid x/y and 70 mm below the vault top (inside the cranium). Returns ``(verts, faces)``."""
    v, f, _ = decimate(read_stl(stl_path), target_verts)
    v, f, _ = keep_main_parts(v, f)
    v = v @ _rot_z(180).T
    return v - np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].max() - 70.0]), f


def prepare_case(stl_path, ref_verts: np.ndarray, target_verts: int = REF_TARGET_VERTS, crop_mm: float = 30.0,
                 max_rms: float = 8.0, scale_range: tuple[float, float] = (0.8, 1.25)) -> dict:
    """Clean and canonically align one skull. Returns verts in the aligned frame in the skull's OWN millimetres.

    ``scale`` maps them into the reference frame (``ref = scale * verts``); it is used only to place the face grid.
    Raises ``ValueError`` if the alignment residual exceeds ``max_rms`` mm or the fitted scale is implausible
    (a free-scale ICP can shrink a corrupt mesh onto the reference, so the residual alone is not enough).
    """
    tri = stl_path if isinstance(stl_path, np.ndarray) else read_stl(stl_path)      # path, or (n, 3, 3) triangles
    v, f, cell = decimate(tri, target_verts)
    v, f, n_small = keep_main_parts(v, f)
    s, R, t, rms = align_to_reference(v, ref_verts)
    if rms > max_rms or not (scale_range[0] <= s <= scale_range[1]):
        raise ValueError(f"{stl_path}: bad alignment (residual {rms:.1f} mm, max {max_rms}; scale {s:.2f}, "
                         f"allowed {scale_range})")
    v_mm = v @ R.T + t / s                                               # rigid, real size
    d, _ = cKDTree(ref_verts).query(s * v_mm)
    keep = d <= crop_mm                                                  # drops neck/spine remnants far from the reference
    kf = f[keep[f].all(axis=1)]
    used = np.unique(kf)
    remap = -np.ones(len(v_mm), int)
    remap[used] = np.arange(len(used))
    v_mm, kf = v_mm[used], remap[kf]
    v_mm, kf, _ = keep_main_parts(v_mm, kf)
    return dict(verts=v_mm.astype(np.float32), faces=kf.astype(np.int32), scale=s, align_rms_mm=rms, cell_mm=cell,
                n_cropped_verts=int((~keep).sum()), n_small_components=n_small)


REGION_NAMES = ["midface", "frontal", "parietal", "occipital", "temporal_L", "temporal_R", "mandible"]


def region_labels(verts_ref: np.ndarray) -> np.ndarray:
    """Coarse anatomical-style region per vertex from reference-frame coordinates (used by the 'region' occlusion pattern)."""
    x, y, z = verts_ref.T
    lab = np.full(len(x), 2)                                             # parietal (default: top/back of the vault)
    lab[(np.abs(x) > 48) & (z < 35)] = 4
    lab[(x > 48) & (z < 35)] = 5
    lab[y < -45] = 3
    lab[(y > 30) & (z >= -5)] = 1
    lab[(y > 38) & (z < -5) & (z > -78)] = 0
    lab[(z <= -78) | ((z < -30) & (y < -10) & (np.abs(x) > 40) & (z > -100))] = 6
    return lab.astype(np.int64)


def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def align_to_reference(verts, ref_verts, n_sub: int = 8000, seed: int = 0):
    """Rigid+scale align ``verts`` to ``ref_verts`` trying the CT-convention flips; returns ``(s, R, t, rms)``.

    Candidate initial rotations: 0 and 180 degrees about z (CT scans differ in anterior direction). The one with the
    smallest trimmed residual wins.
    """
    rng = np.random.default_rng(seed)
    a = verts[rng.choice(len(verts), min(n_sub, len(verts)), replace=False)]
    b = ref_verts[rng.choice(len(ref_verts), min(n_sub, len(ref_verts)), replace=False)]
    best = None
    for rz in (0, 180):
        s, R, t, rms = icp_similarity(a, b, _rot_z(rz))
        if best is None or rms < best[3]:
            best = (s, R, t, rms)
    return best
