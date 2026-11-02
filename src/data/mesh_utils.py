"""Triangle-mesh utilities for UNCERTAFACE.

All functions are NumPy-based and operate on ``verts`` (V, 3) float arrays and
``faces`` (F, 3) integer arrays. Units are whatever the caller supplies; the
data pipeline uses millimetres.

Contents: icosphere construction, areas/normals, topology validation and
cleaning, mesh I/O, area-weighted surface sampling, farthest point sampling,
and measurement of the surface area fraction removed by a vertex mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def icosphere(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere with ``10 * 4**s + 2`` vertices and ``20 * 4**s`` faces.

    Built by recursively splitting each triangle and projecting new midpoints
    onto the unit sphere. Faces are counter-clockwise seen from outside, so
    face normals point outward.
    """
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)

    for _ in range(int(subdivisions)):
        verts_list = list(verts)
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                m = verts_list[a] + verts_list[b]
                verts_list.append(m / np.linalg.norm(m))
                cache[key] = len(verts_list) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts, faces = np.array(verts_list), np.array(new_faces, dtype=np.int64)
    return verts, faces


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def face_normals(verts: np.ndarray, faces: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Per-face normals; unnormalised normals have length ``2 * area``."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    if normalize:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return n


def face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face area: ``0.5 * |(v1 - v0) x (v2 - v0)|``."""
    return 0.5 * np.linalg.norm(face_normals(verts, faces, normalize=False), axis=1)


def surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Total surface area."""
    return float(face_areas(verts, faces).sum())


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (unit length; zero for isolated vertices)."""
    fn = face_normals(verts, faces, normalize=False)  # length = 2*area -> area weighting
    vn = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where(norm > 0, vn / np.maximum(norm, 1e-20), 0.0)


def vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric vertex area: one third of each incident face's area."""
    a = face_areas(verts, faces) / 3.0
    out = np.zeros(len(verts))
    for k in range(3):
        np.add.at(out, faces[:, k], a)
    return out


# --------------------------------------------------------------------------
# Validation and cleaning
# --------------------------------------------------------------------------
@dataclass
class MeshReport:
    """Topology and validity summary of a mesh."""
    n_verts: int
    n_faces: int
    n_nonfinite_verts: int = 0
    n_out_of_range_faces: int = 0
    n_degenerate_faces: int = 0
    n_duplicate_faces: int = 0
    n_unreferenced_verts: int = 0
    n_nonmanifold_edges: int = 0
    n_boundary_edges: int = 0
    euler_characteristic: int = 0
    is_watertight: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _edge_counts(faces: np.ndarray) -> np.ndarray:
    """Return counts of undirected-edge occurrences."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def validate_mesh(verts: np.ndarray, faces: np.ndarray, area_eps: float = 1e-12) -> MeshReport:
    """Inspect a mesh without modifying it.

    Degenerate means area below ``area_eps`` or repeated vertex indices.
    Watertight means every edge is shared by exactly two faces.
    """
    rep = MeshReport(n_verts=len(verts), n_faces=len(faces))
    finite = np.isfinite(verts).all(axis=1)
    rep.n_nonfinite_verts = int((~finite).sum())
    in_range = ((faces >= 0) & (faces < len(verts))).all(axis=1)
    rep.n_out_of_range_faces = int((~in_range).sum())
    ok = faces[in_range]
    if len(ok):
        repeated = (ok[:, 0] == ok[:, 1]) | (ok[:, 1] == ok[:, 2]) | (ok[:, 0] == ok[:, 2])
        safe_verts = np.where(np.isfinite(verts), verts, 0.0)
        small = face_areas(safe_verts, ok) < area_eps
        rep.n_degenerate_faces = int((repeated | small).sum())
        rep.n_duplicate_faces = int(len(ok) - len(np.unique(np.sort(ok, axis=1), axis=0)))
        rep.n_unreferenced_verts = int(len(verts) - len(np.unique(ok)))
        counts = _edge_counts(ok)
        rep.n_boundary_edges = int((counts == 1).sum())
        rep.n_nonmanifold_edges = int((counts > 2).sum())
        rep.is_watertight = bool((counts == 2).all())
        n_edges = len(counts)
        rep.euler_characteristic = int(len(np.unique(ok)) - n_edges + len(ok))
    if rep.n_nonfinite_verts or rep.n_out_of_range_faces:
        rep.notes.append("invalid vertices/faces present")
    return rep


def clean_mesh(verts: np.ndarray, faces: np.ndarray, area_eps: float = 1e-12
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove non-finite vertices, bad/degenerate/duplicate faces, unused vertices.

    Returns ``(verts, faces, vert_index)`` where ``vert_index[i]`` is the index
    in the *input* mesh of output vertex ``i``. Callers that rely on vertex
    correspondence (e.g. face templates) must check ``len(vert_index)`` equals
    the input size, otherwise correspondence has been broken.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    finite = np.isfinite(verts).all(axis=1)
    keep = (faces >= 0).all(axis=1) & (faces < len(verts)).all(axis=1)
    keep &= finite[np.clip(faces, 0, len(verts) - 1)].all(axis=1)
    faces = faces[keep]
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    faces = faces[~repeated]
    faces = faces[face_areas(verts, faces) >= area_eps]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], used


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def save_mesh(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a mesh (format from extension, e.g. .ply/.obj) without any merging."""
    import trimesh
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh as ``(verts float64, faces int64)`` with vertex order preserved."""
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int,
                   rng: np.random.Generator, return_face_idx: bool = False):
    """Area-weighted uniform sampling of ``n`` points on the surface.

    A face is drawn with probability proportional to its area, then a point is
    drawn uniformly inside it using ``p = (1-sqrt(r1)) v0 + sqrt(r1)(1-r2) v1
    + sqrt(r1) r2 v2``, which is uniform over the triangle.
    """
    areas = face_areas(verts, faces)
    prob = areas / areas.sum()
    fidx = rng.choice(len(faces), size=n, p=prob)
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    tri = verts[faces[fidx]]
    pts = (1 - r1) * tri[:, 0] + r1 * (1 - r2) * tri[:, 1] + r1 * r2 * tri[:, 2]
    return (pts, fidx) if return_face_idx else pts


def sample_vertices(verts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sampling of mesh *vertices* (not area-uniform if tessellation is uneven).

    Samples without replacement when ``n <= len(verts)``, otherwise with.
    """
    idx = rng.choice(len(verts), size=n, replace=n > len(verts))
    return verts[idx]


def farthest_point_sampling(points: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Greedy farthest point sampling; returns the ``n`` selected indices.

    O(n * P) time. Deterministic given ``start``.
    """
    P = len(points)
    if n > P:
        raise ValueError(f"cannot sample {n} points from {P}")
    idx = np.empty(n, dtype=np.int64)
    dist = np.full(P, np.inf)
    cur = start
    for i in range(n):
        idx[i] = cur
        d = ((points - points[cur]) ** 2).sum(axis=1)
        dist = np.minimum(dist, d)
        cur = int(dist.argmax())
    return idx


# --------------------------------------------------------------------------
# Missingness measurement
# --------------------------------------------------------------------------
def missing_area_fraction(verts: np.ndarray, faces: np.ndarray, visible: np.ndarray) -> float:
    """Fraction of surface area that is *not* observed, from a per-vertex mask.

    Each vertex owns its barycentric area (one third of incident faces), so the
    result is the area-weighted fraction of vertices with ``visible == False``.
    This is a measurement on the actual mesh, not a nominal target.
    """
    va = vertex_areas(verts, faces)
    return float(va[~np.asarray(visible, dtype=bool)].sum() / va.sum())
