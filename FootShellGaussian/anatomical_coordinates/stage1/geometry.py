"""Surface sampling and exact point-to-mesh distance.

The distance is the unsigned one on purpose. Stage 1 trains every model on
the same targets, and on these CAD meshes an inside test is not reliable
everywhere (see ``signs``): where an upper is a single sheet, the winding
number calls the air inside the shoe material. An unsigned distance is
well defined on any triangle soup.

Exactness: triangles wider than ``max_radius`` are split first (the surface is
unchanged), so the true closest triangle's centroid lies within the query's
distance plus that radius. The candidate search takes the ``k`` nearest
centroids and the error it can leave is bounded by the split radius, not by
the mesh's own triangle size.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..coordinate_mapping.material import subdivide_large


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0):
    """Area-uniform points, with the face each lies on and its unit normal."""

    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    keep = np.nonzero(area > 1e-16)[0]
    generator = np.random.default_rng(seed)
    face = generator.choice(keep, size=count, p=area[keep] / area[keep].sum())
    u, v = generator.random(count), generator.random(count)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    t = tri[face]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return points, face, normal[face] / area[face, None]


def closest_on_candidates(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest of its candidate triangles.

    ``corners`` is (N, K, 3, 3). Ericson's region test (Real-Time Collision
    Detection, 5.1.5), the same cascade ``FiberTracer.project_to_surface``
    uses, applied to arbitrary candidate triangles.
    """

    a, b, c = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
    ab, ac = b - a, c - a
    p = points[:, None, :]
    ap, bp, cp = p - a, p - b, p - c
    d1 = np.einsum("nkj,nkj->nk", ab, ap); d2 = np.einsum("nkj,nkj->nk", ac, ap)
    d3 = np.einsum("nkj,nkj->nk", ab, bp); d4 = np.einsum("nkj,nkj->nk", ac, bp)
    d5 = np.einsum("nkj,nkj->nk", ab, cp); d6 = np.einsum("nkj,nkj->nk", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    total = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1.0)
    v = vb / total
    w = vc / total

    def override(mask, vv, ww):
        nonlocal v, w
        v = np.where(mask, vv, v)
        w = np.where(mask, ww, w)

    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where((d1 - d3) != 0, d1 / (d1 - d3), 0.0)
        override((vc <= 0) & (d1 >= 0) & (d3 <= 0), e, 0.0)
        e = np.where((d2 - d6) != 0, d2 / (d2 - d6), 0.0)
        override((vb <= 0) & (d2 >= 0) & (d6 <= 0), 0.0, e)
        span = (d4 - d3) + (d5 - d6)
        e = np.where(span != 0, (d4 - d3) / span, 0.0)
        override((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), 1.0 - e, e)
    override((d1 <= 0) & (d2 <= 0), 0.0, 0.0)
    override((d3 >= 0) & (d4 <= d3), 1.0, 0.0)
    override((d6 >= 0) & (d5 <= d6), 0.0, 1.0)
    nearest = a + v[..., None] * ab + w[..., None] * ac
    return np.linalg.norm(nearest - p, axis=-1).min(axis=1)


class DistanceField:
    """Exact unsigned distance to one mesh, for many queries."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 max_radius: float = 1.0 / 262.5, neighbours: int = 32) -> None:
        v, f = subdivide_large(vertices, faces, max_radius)
        self.corners = v[f]
        self.tree = cKDTree(self.corners.mean(axis=1))
        self.k = min(neighbours, len(f))

    def __call__(self, points: np.ndarray, chunk: int = 32768) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        out = np.empty(len(points))
        for begin in range(0, len(points), chunk):
            block = points[begin:begin + chunk]
            _, index = self.tree.query(block, k=self.k, workers=-1)
            index = np.atleast_2d(index)
            out[begin:begin + chunk] = closest_on_candidates(block, self.corners[index])
        return out
