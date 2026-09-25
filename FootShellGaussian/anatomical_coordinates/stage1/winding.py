"""Generalized winding numbers, on the GPU.

An inside test is needed to give training samples a sign, and the shoe meshes
have thousands of open edges, so ray parity is unreliable. The generalized
winding number (Jacobson, Kavan and Sorkine-Hornung, "Robust inside-outside
segmentation using generalized winding numbers", SIGGRAPH 2013) sums the solid
angle every triangle subtends at the query point, divided by 4 pi. For a closed,
consistently oriented surface it is exactly 1 inside and 0 outside; a hole only
blurs it locally, towards 0.5 near the hole, instead of flipping whole regions.

It does rely on consistent face orientation. A mesh whose faces point inwards
gives -1 inside, which is why ``winding_numbers`` reports the raw value and
``inside`` decides the orientation from the data.

Brute force over all triangles is used on purpose: at these mesh sizes (up to
~480k triangles) a GPU sums them fast enough that a Barnes-Hut tree
(Barill et al., "Fast winding numbers for soups and clouds", 2018) would only
add approximation error.
"""

from __future__ import annotations

import numpy as np
import torch


def winding_numbers(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: torch.device | str = "cuda",
    point_chunk: int = 2048,
    face_chunk: int = 65536,
) -> np.ndarray:
    """Raw generalized winding number of the triangle soup at every point.

    Solid angle of one triangle, Van Oosterom and Strackee (1983)::

        tan(omega / 2) = det[a b c] /
            (|a||b||c| + (a.b)|c| + (b.c)|a| + (c.a)|b|)

    with a, b, c the corners relative to the query point.
    """

    device = torch.device(device)
    tri = torch.as_tensor(
        np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)],
        dtype=torch.float32, device=device,
    )
    pts = torch.as_tensor(np.asarray(points, dtype=np.float64), dtype=torch.float32,
                          device=device)
    out = torch.zeros(len(pts), dtype=torch.float64, device=device)
    for begin in range(0, len(pts), point_chunk):
        p = pts[begin:begin + point_chunk]
        total = torch.zeros(len(p), dtype=torch.float64, device=device)
        for start in range(0, len(tri), face_chunk):
            t = tri[start:start + face_chunk]
            a = t[None, :, 0] - p[:, None]
            b = t[None, :, 1] - p[:, None]
            c = t[None, :, 2] - p[:, None]
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            det = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
            denom = (la * lb * lc + (a * b).sum(-1) * lc
                     + (b * c).sum(-1) * la + (c * a).sum(-1) * lb)
            total += torch.atan2(det, denom).sum(dim=1).double()
        out[begin:begin + point_chunk] = total * (2.0 / (4.0 * np.pi))
    return out.cpu().numpy()


def orientation(vertices: np.ndarray, faces: np.ndarray, device="cuda") -> float:
    """+1 if faces point outwards, -1 if inwards, judged from far and near.

    The winding number is 0 far away regardless of orientation, so it is read
    a small step to both sides of sampled faces. The side with the larger
    magnitude is the inside, and the median sign there says which way the
    faces point. Reading only against the normal would land outside when the
    faces point inwards, and report the wrong orientation.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    tri = vertices[np.asarray(faces)]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    keep = area > 1e-14
    generator = np.random.default_rng(0)
    pick = generator.choice(np.nonzero(keep)[0], size=min(2048, keep.sum()),
                            p=area[keep] / area[keep].sum())
    centre = tri[pick].mean(axis=1)
    unit = normal[pick] / area[pick, None]
    step = 1e-3 * float(np.ptp(vertices, axis=0).max())
    behind = winding_numbers(centre - step * unit, vertices, faces, device)
    ahead = winding_numbers(centre + step * unit, vertices, faces, device)
    inner = np.where(np.abs(behind) >= np.abs(ahead), behind, ahead)
    return 1.0 if np.median(inner) >= 0.0 else -1.0


def inside(points, vertices, faces, device="cuda", sign: float | None = None):
    """Winding number with the orientation fixed so that inside reads ~1."""

    sign = orientation(vertices, faces, device) if sign is None else sign
    return sign * winding_numbers(points, vertices, faces, device)
