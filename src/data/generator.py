"""Procedural SYNTHETIC skull/face pair generator (research prototype).

This is a simplified geometric model, NOT an anatomical simulation.

Coordinates (mm, origin at the cranial centre): +x right, +y anterior, +z superior.

Skull
    A deformed ellipsoid on a fixed icosphere topology. Vertex ``i`` always
    has the same unit direction ``d_i``, so vertex ``i`` is in correspondence
    across every subject. Radial displacement = sum of Gaussian "bumps" on the
    sphere of directions (brow, forehead, occiput, zygoma, chin, jaw angles, a
    nasal-aperture indentation) plus a smooth random field with a controllable
    left-right asymmetric part.

Face
    The facial-region subset of the skull vertices, displaced along the skull
    vertex normals by a soft-tissue thickness field

        t(v) = max(t_min, s * t0(d_v) * (1 + a * g(d_v))
                          + nose(d_v) + lips(d_v) + cheeks(d_v) + u(d_v))

    ``t0`` interpolates assumed regional anchor thicknesses. The parameters
    ``s, a, g, nose, lips, cheeks, u`` are drawn independently of the bone
    parameters, so part of the face is NOT determined by the skull. That
    irreducible part is the ground-truth aleatoric ambiguity of this synthetic
    world. The face is an open patch (not a full head) and the "face" is simply
    skull + normal * t; it is not real facial anatomy.

Everything is deterministic given ``(seed, base_id)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import mesh_utils as mu

REGION_NAMES = ["face", "frontal", "parietal", "occipital", "temporal_L", "temporal_R", "jaw"]
FACE_REGION = 0

# Anchor (direction, assumed thickness in mm) for the base soft-tissue map.
# These are assumed prototype values, not clinical measurements.
_THICKNESS_ANCHORS = [
    ((0.0, 0.85, 0.50), 4.5),    # forehead
    ((0.0, 1.00, 0.10), 3.0),    # glabella / nasal bridge
    ((0.0, 1.00, -0.15), 3.5),   # nose
    ((0.50, 0.80, -0.25), 10.0),  # cheek R
    ((-0.50, 0.80, -0.25), 10.0),  # cheek L
    ((0.0, 0.95, -0.45), 10.0),  # mouth
    ((0.0, 0.70, -0.80), 8.0),   # chin
]
_ANCHOR_SIGMA = 0.35   # rad
_BACKGROUND_T = 6.0    # mm, far from every anchor
_BACKGROUND_W = 0.02


# --------------------------------------------------------------------------
# Spherical helpers
# --------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def bump(d: np.ndarray, center, sigma: float) -> np.ndarray:
    """Gaussian bump in geodesic angle: ``exp(-0.5 (angle(d, c) / sigma)^2)``."""
    ang = np.arccos(np.clip(d @ _unit(center), -1.0, 1.0))
    return np.exp(-0.5 * (ang / sigma) ** 2)


def sym_bump(d: np.ndarray, center, sigma: float) -> np.ndarray:
    """Bump at ``center`` plus its left-right mirror (x -> -x), unit peak."""
    c = np.asarray(center, dtype=np.float64)
    m = c * np.array([-1.0, 1.0, 1.0])
    if np.allclose(c, m):
        return bump(d, c, sigma)
    return np.maximum(bump(d, c, sigma), bump(d, m, sigma))


def _mirror(d: np.ndarray) -> np.ndarray:
    return d * np.array([-1.0, 1.0, 1.0])


_REF_DIRS = None


def _ref_dirs() -> np.ndarray:
    global _REF_DIRS
    if _REF_DIRS is None:
        _REF_DIRS = mu.icosphere(3)[0]
    return _REF_DIRS


def make_random_field(seed: int, k: int = 6, sigma_range=(0.5, 0.9)):
    """Smooth random scalar field on the sphere, normalised to ``max |f| = 1``.

    Returns ``f(d)`` valid for any direction array (so mirrored evaluation
    is possible). Normalisation uses a fixed reference grid.
    """
    rng = np.random.default_rng(int(seed))
    centers = rng.normal(size=(k, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    w = rng.uniform(-1.0, 1.0, size=k)
    sig = rng.uniform(*sigma_range, size=k)

    def raw(d):
        return sum(w[i] * bump(d, centers[i], sig[i]) for i in range(k))

    ref = _ref_dirs()
    scale = max(np.abs(raw(ref)).max(), 1e-9)
    return lambda d: raw(d) / scale


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator, cfg: dict) -> dict:
    """Draw one parameter set uniformly within the configured ranges.

    ``cfg`` is the ``generator`` section of dataset.yaml. Three integer seeds
    (for the shape field, thickness field and unobserved-thickness field) are
    stored so the full geometry is reproducible from the parameter dict.
    """
    p: dict = {}
    for grp in ("skull", "face"):
        for name, (lo, hi) in cfg[grp].items():
            p[name] = float(rng.uniform(lo, hi))
    for name in ("field_seed", "thickness_field_seed", "unobserved_field_seed"):
        p[name] = int(rng.integers(0, 2**31 - 1))
    return p


def param_vector(p: dict, cfg: dict) -> np.ndarray:
    """Continuous parameters as a vector in ``cfg`` order (for diversity checks)."""
    return np.array([p[n] for grp in ("skull", "face") for n in cfg[grp]])


def param_bounds_violations(p: dict, cfg: dict) -> list[str]:
    """Names of parameters outside their configured range."""
    return [n for grp in ("skull", "face") for n, (lo, hi) in cfg[grp].items()
            if not (lo - 1e-9 <= p[n] <= hi + 1e-9)]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def region_labels(d: np.ndarray) -> np.ndarray:
    """Integer region label per direction (see ``REGION_NAMES``); first match wins."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    lab = np.full(len(d), 2, dtype=np.int64)  # parietal default
    lab[(np.abs(x) > 0.6)] = 4
    lab[(x > 0.6)] = 5
    lab[z < -0.55] = 6
    lab[y < -0.35] = 3
    lab[(y > 0.2) & (z >= 0.5)] = 1
    lab[(y > 0.35) & (z < 0.5) & (np.abs(x) <= 0.8)] = FACE_REGION  # |x| cap keeps the patch frontal
    return lab


def _radial_factor(d: np.ndarray, p: dict) -> np.ndarray:
    """Multiplicative radial deformation ``1 + sum(amp_i * bump_i)``."""
    f = np.zeros(len(d))
    f += p["forehead_bulge"] * bump(d, (0, 0.85, 0.5), 0.45)
    f += p["occipital_bulge"] * bump(d, (0, -0.9, 0.2), 0.5)
    f += p["brow_ridge"] * (bump(d, (0, 0.92, 0.28), 0.22) + 0.6 * sym_bump(d, (0.45, 0.85, 0.28), 0.2))
    f += p["zygomatic"] * sym_bump(d, (0.8, 0.5, -0.15), 0.3)
    f += p["chin_projection"] * bump(d, (0, 0.75, -0.7), 0.3)
    f += p["jaw_width"] * sym_bump(d, (0.85, -0.2, -0.65), 0.3)
    f -= p["nasal_aperture_depth"] * bump(d, (0, 1.0, -0.15), 0.18)
    field = make_random_field(p["field_seed"])
    fd, fm = field(d), field(_mirror(d))
    sym, anti = 0.5 * (fd + fm), 0.5 * (fd - fm)
    f += p["field_amp"] * (sym + p["asymmetry"] * anti)
    return 1.0 + f


def base_thickness(d: np.ndarray) -> np.ndarray:
    """Assumed regional soft-tissue thickness ``t0`` (mm), normalised RBF blend."""
    num = _BACKGROUND_W * _BACKGROUND_T * np.ones(len(d))
    den = _BACKGROUND_W * np.ones(len(d))
    for c, t in _THICKNESS_ANCHORS:
        w = bump(d, c, _ANCHOR_SIGMA)
        num += w * t
        den += w
    return num / den


def thickness_field(d: np.ndarray, p: dict, min_thickness: float) -> np.ndarray:
    """Soft-tissue thickness (mm) at directions ``d`` for parameters ``p``."""
    g = make_random_field(p["thickness_field_seed"])(d)
    u = make_random_field(p["unobserved_field_seed"])(d)
    t = p["thickness_scale"] * base_thickness(d) * (1.0 + p["thickness_field_amp"] * g)
    t += p["nose_projection"] * bump(d, (0, 1.0, -0.15), 0.18)
    t += p["lip_fullness"] * bump(d, (0, 0.95, -0.45), 0.2)
    t += p["cheek_fat"] * sym_bump(d, (0.50, 0.80, -0.25), 0.35)
    t += p["unobserved_field_amp"] * u
    return np.maximum(t, min_thickness)


@dataclass
class Subject:
    """One synthetic skull/face pair. Everything is in mm."""
    base_id: int
    params: dict
    skull_verts: np.ndarray       # (V, 3)
    skull_faces: np.ndarray       # (F, 3) shared topology
    region: np.ndarray            # (V,) int labels, see REGION_NAMES
    face_vert_index: np.ndarray   # (Vf,) skull vertex index of each face vertex
    face_verts: np.ndarray        # (Vf, 3) ground-truth face, vertex-wise corresponding
    face_faces: np.ndarray        # (Ff, 3) indices into face_verts
    thickness: np.ndarray         # (Vf,) ground-truth soft-tissue thickness in mm
    synthetic: bool = True


def build_subject(base_id: int, params: dict, cfg: dict, subdivisions: int = 4) -> Subject:
    """Deterministically build the skull and face for a parameter set."""
    d, faces = mu.icosphere(subdivisions)
    axes = np.array([params["semi_width"], params["semi_length"], params["semi_height"]])
    skull = d * axes * _radial_factor(d, params)[:, None]
    region = region_labels(d)
    vn = mu.vertex_normals(skull, faces)

    in_face = region == FACE_REGION
    sub = faces[in_face[faces].all(axis=1)]
    fidx = np.unique(sub)      # region vertices without a complete triangle are dropped (same for every subject)
    t = thickness_field(d[fidx], params, cfg["min_thickness_mm"])
    face = skull[fidx] + vn[fidx] * t[:, None]

    remap = -np.ones(len(d), dtype=np.int64)
    remap[fidx] = np.arange(len(fidx))
    return Subject(base_id, params, skull, faces, region, fidx, face, remap[sub], t)


def generate_base(base_id: int, seed: int, cfg: dict, subdivisions: int = 4, attempt: int = 0) -> Subject:
    """Generate base geometry ``base_id``; independent of generation order.

    ``attempt > 0`` draws a fresh parameter set for the same id; the dataset
    builder uses it to replace candidates rejected as near-duplicates.
    ``attempt = 0`` reproduces the original seeding exactly.
    """
    key = [int(seed), int(base_id)] + ([int(attempt)] if attempt else [])
    rng = np.random.default_rng(np.random.SeedSequence(key))
    return build_subject(base_id, sample_params(rng, cfg), cfg, subdivisions)
