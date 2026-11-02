"""Synthetic soft-tissue face draped over a REAL (aligned) skull.

The skull is real CT-derived geometry; the FACE is synthetic: it is built from the skull, not measured.

Everything is computed in a common REFERENCE frame (aligned skull coordinates multiplied by the fitted scale ``s``)
so the face grid lands on the same anatomy for every skull, then divided by ``s`` to return to the skull's own
millimetres.

  1. ``front_envelope``   front-most surface y(x, z) of the skull: vertices are splatted on a fine (x, z) raster
                          (max y per cell), holes filled, then grey-scale CLOSING bridges the orbits, nasal opening
                          and the gap between the teeth (soft tissue spans them) and Gaussian smoothing removes
                          bony detail. The result is sampled on the face grid (5 mm spacing, oval mask).
  2. ``thickness``        soft-tissue thickness t(x, z) = max(t_min, s_t * t0 * (1 + a g) + nose + lips + cheeks + u),
                          with t0 a normalised-RBF blend of ASSUMED regional anchor thicknesses (not clinical data) and
                          the remaining terms drawn independently of the bone (irreducible ambiguity by design).
  3. ``build_face``       face = base + t * n, n = vertex normals of the base grid mesh.

Assumptions and limits: the envelope is a frontal heightfield (valid for the frontal patch only); the face is
skull + normal * t, not real facial anatomy; anchor positions were set on the reference skull by eye.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing, distance_transform_edt

from . import mesh_utils as mu

FACE_X = np.arange(-55.0, 55.1, 5.0)
FACE_Z = np.arange(30.0, -130.1, -5.0)
RES = 1.5


def splat_depth(verts: np.ndarray, xr=(-90.0, 90.0), zr=(-140.0, 70.0), res: float = RES):
    """Max-y raster of vertices over (x, z); empty cells filled from the nearest filled cell."""
    nx, nz = int((xr[1] - xr[0]) / res) + 1, int((zr[1] - zr[0]) / res) + 1
    ix = np.clip(np.rint((verts[:, 0] - xr[0]) / res).astype(int), 0, nx - 1)
    iz = np.clip(np.rint((verts[:, 2] - zr[0]) / res).astype(int), 0, nz - 1)
    H = np.full((nz, nx), -np.inf)
    np.maximum.at(H, (iz, ix), verts[:, 1])
    empty = ~np.isfinite(H)
    if empty.any():
        _, (ri, ci) = distance_transform_edt(empty, return_indices=True)
        H = H[ri, ci]
    return H, xr, zr


def front_envelope(verts: np.ndarray, close_mm: float = 22.0, smooth_mm: float = 5.0, res: float = RES):
    """Closed + smoothed front-most surface; returns ``(H, xr, zr)`` (H indexed [z, x], mm)."""
    H, xr, zr = splat_depth(verts, res=res)
    r = int(round(close_mm / res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    Hc = grey_closing(H, footprint=disk, mode="nearest")
    return gaussian_filter(Hc, sigma=smooth_mm / res, mode="nearest"), xr, zr


def sample_on_grid(H, xr, zr, res: float = RES):
    """Bilinear samples of the envelope at the oval-masked face grid; returns (verts (V,3), mask, rc)."""
    from scipy.ndimage import map_coordinates
    X, Z = np.meshgrid(FACE_X, FACE_Z, indexing="xy")
    mask = oval_mask(X, Z)
    cx = (X - xr[0]) / res
    cz = (Z - zr[0]) / res
    y = map_coordinates(H, [cz, cx], order=1, mode="nearest")
    return np.stack([X[mask], y[mask], Z[mask]], axis=1), mask


def oval_mask(X, Z):
    """Face outline: an oval over the frontal bone, midface and chin (reference-frame mm)."""
    return (X / 58.0) ** 2 + ((Z + 50.0) / 82.0) ** 2 <= 1.0


def grid_triangles(mask: np.ndarray):
    """Triangles of the masked grid (indices into the masked vertex list), +y-facing winding."""
    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    tris = []
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            a, b, c2, d = idx[r, c], idx[r, c + 1], idx[r + 1, c], idx[r + 1, c + 1]
            if min(a, b, c2, d) >= 0:
                tris += [[a, c2, b], [b, c2, d]]
    return np.array(tris, int)


# ------------------------------------------------------------------ thickness
# (x, z, assumed thickness mm) in the reference frame; set by eye on the reference skull (see docs)
ANCHORS = [(0.0, 12.0, 4.5),     # forehead
           (0.0, -14.0, 5.0),    # glabella / nasal root
           (0.0, -38.0, 3.5),    # nose over the nasal bones (tip projection is a separate latent)
           (37.0, -36.0, 6.0),   # over the orbit (lids/globe)
           (-37.0, -36.0, 6.0),
           (42.0, -64.0, 12.0),  # cheek
           (-42.0, -64.0, 12.0),
           (0.0, -68.0, 8.0),    # upper lip / philtrum
           (0.0, -84.0, 10.0),   # mouth
           (0.0, -116.0, 8.0)]   # chin
SIGMA = 14.0
BACKGROUND_T, BACKGROUND_W = 6.0, 0.02


def _bump(x, z, cx, cz, sig):
    return np.exp(-0.5 * ((x - cx) ** 2 + (z - cz) ** 2) / sig ** 2)


def base_thickness(x, z):
    num = BACKGROUND_W * BACKGROUND_T * np.ones_like(x)
    den = BACKGROUND_W * np.ones_like(x)
    for cx, cz, t in ANCHORS:
        w = _bump(x, z, cx, cz, SIGMA)
        num, den = num + w * t, den + w
    return num / den


def random_field_2d(seed: int, x, z, k: int = 6):
    """Smooth random field over (x, z), normalised to max |f| = 1 on the grid."""
    rng = np.random.default_rng(int(seed))
    cx, cz = rng.uniform(-50, 50, k), rng.uniform(-120, 30, k)
    w, sg = rng.uniform(-1, 1, k), rng.uniform(15, 35, k)
    f = sum(w[i] * _bump(x, z, cx[i], cz[i], sg[i]) for i in range(k))
    return f / max(np.abs(f).max(), 1e-9)


def thickness(x, z, p: dict, t_min: float = 1.0):
    """Soft-tissue thickness (reference-frame mm) from face latents ``p`` (same names as the procedural generator)."""
    g = random_field_2d(p["thickness_field_seed"], x, z)
    u = random_field_2d(p["unobserved_field_seed"], x, z)
    t = p["thickness_scale"] * base_thickness(x, z) * (1 + p["thickness_field_amp"] * g)
    t += p["nose_projection"] * _bump(x, z, 0.0, -52.0, 8.0)
    t += p["lip_fullness"] * _bump(x, z, 0.0, -84.0, 9.0)
    t += p["cheek_fat"] * np.maximum(_bump(x, z, 42.0, -64.0, 14.0), _bump(x, z, -42.0, -64.0, 14.0))
    t += p["unobserved_field_amp"] * u
    return np.maximum(t, t_min)


def envelope_base(skull_ref: np.ndarray) -> dict:
    """Bone-side base surface of the face for a skull (reference-frame coords): base verts, triangles, mask, normals."""
    H, xr, zr = front_envelope(skull_ref)
    base, mask = sample_on_grid(H, xr, zr)
    faces = grid_triangles(mask)
    n = mu.vertex_normals(base, faces)
    n[n[:, 1] < 0] *= -1                                       # frontal patch: normals must point anteriorly
    return dict(base=base, faces=faces, mask=mask, normals=n)


def apply_thickness(b: dict, p: dict, t_min: float = 1.0) -> dict:
    """Face = base + t * n for the face latents ``p`` over a precomputed ``envelope_base``."""
    t = thickness(b["base"][:, 0], b["base"][:, 2], p, t_min)
    return dict(base=b["base"], face=b["base"] + b["normals"] * t[:, None], thickness=t, faces=b["faces"], mask=b["mask"])


def build_face(skull_ref: np.ndarray, p: dict, t_min: float = 1.0) -> dict:
    """Face for a skull given in the reference frame (scaled coords). Returns base, face, thickness, faces (ref frame)."""
    return apply_thickness(envelope_base(skull_ref), p, t_min)
