"""Anatomical fibers: the curves that run outward from the foot.

A fiber starts on the foot's surface and travels out through the surrounding
volume, so every point along it shares the same position-on-the-foot and differs
only in how far out it sits. Two shoes are compared by looking along the same
fiber.

The older implementation integrated a direction field through each shoe's own
deformed mesh, cell by cell, and had to exclude cells where that integration
could not be certified. Under a flow map none of that is per-shoe: the canonical
fibers are traced **once**, in the canonical anatomy, and then carried through
each shoe's map. A smooth invertible map takes a curve to a curve, so pushing a
traced fiber forward is an evaluation rather than another integration.

The direction field itself is not recomputed. It is the field solved for during
the original fiber work (``fiber_field.npz``), read unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lookup import CanonicalSemantics


@dataclass
class CanonicalFibers:
    """Fibers in the canonical anatomy, shared by every shoe."""

    origins: np.ndarray       # (F, 3) start points on the foot surface
    curves: np.ndarray        # (F, S, 3) sampled points along each fiber
    progress: np.ndarray      # (F, S) outward progress in [0, 1]
    face_indices: np.ndarray  # (F,) which inner-boundary triangle each starts on
    #: 0 reached the envelope, 1 left the domain early, 2 direction too weak,
    #: 3 ran out of samples. Reported so coverage is never overstated.
    stopped: np.ndarray


def load_direction_field(path: Path, vertex_count: int) -> np.ndarray:
    data = np.load(path)
    directions = np.asarray(data["directions"], dtype=np.float64)
    if directions.shape != (vertex_count, 3):
        raise ValueError(
            f"direction field has {directions.shape}, expected ({vertex_count}, 3)"
        )
    return directions


def trace_canonical(
    semantics: CanonicalSemantics,
    directions: np.ndarray,
    count: int = 2048,
    samples: int = 48,
    step: float = 0.01,
    seed: int = 0,
) -> CanonicalFibers:
    """Trace fibers outward from the canonical foot surface.

    Done once. The canonical anatomy never changes, so neither do these.
    """

    generator = np.random.default_rng(seed)
    faces = semantics.inner_vertex_indices[semantics.inner_faces]
    triangles = semantics.vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    keep = areas > 0.0
    picked = generator.choice(
        np.nonzero(keep)[0], size=count, p=areas[keep] / areas[keep].sum()
    )
    u = generator.random(count)
    v = generator.random(count)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    corners = triangles[picked]
    origins = (
        corners[:, 0]
        + u[:, None] * (corners[:, 1] - corners[:, 0])
        + v[:, None] * (corners[:, 2] - corners[:, 0])
    )

    curves = np.full((count, samples, 3), np.nan)
    progress = np.full((count, samples), np.nan)
    stopped = np.zeros(count, dtype=np.int8)  # 0 reached the envelope
    current = origins.copy()
    alive = np.ones(count, dtype=bool)
    curves[:, 0] = current
    for index in range(1, samples):
        located = semantics.locate(current)
        moving = alive & located.found
        if not moving.any():
            stopped[alive] = 1  # left the domain
            break
        corner_ids = semantics.tetrahedra[located.tetrahedron[moving]]
        # Direction is interpolated from the canonical vertex field, so the
        # curve is continuous across cell faces by construction.
        velocity = np.einsum(
            "ni,nij->nj", located.barycentric[moving], directions[corner_ids]
        )
        norm = np.linalg.norm(velocity, axis=1, keepdims=True)
        weak = norm[:, 0] < 1e-9
        velocity = np.where(weak[:, None], 0.0, velocity / np.maximum(norm, 1e-12))
        advanced = current[moving] + step * velocity
        current = current.copy()
        current[moving] = advanced
        progress[moving, index] = semantics.harmonic(
            semantics.locate(advanced)
        )
        curves[moving, index] = advanced
        lost = alive.copy()
        lost[moving] = False
        stopped[lost] = 1
        alive = moving
        alive_indices = np.nonzero(alive)[0]
        weak_stop = alive_indices[weak] if weak.any() else np.array([], dtype=np.int64)
        stopped[weak_stop] = 2  # direction field too weak to continue
        alive[weak_stop] = False
        if not alive.any():
            break
    progress[:, 0] = 0.0
    # Reaching the outer envelope also means leaving the tetrahedral domain, so
    # the two have to be told apart after the fact: a fiber whose last valid
    # sample is at full outward progress finished, it did not fail.
    for index in range(count):
        valid = progress[index][np.isfinite(progress[index])]
        if valid.size and valid[-1] >= 0.98:
            stopped[index] = 0
        elif stopped[index] == 0:
            stopped[index] = 3  # ran out of samples before the envelope
    return CanonicalFibers(origins, curves, progress, picked, stopped)


def push_through(fibers: CanonicalFibers, flow) -> np.ndarray:
    """Carry canonical fibers into one shoe. ``flow`` maps (N,3) -> (N,3)."""

    shape = fibers.curves.shape
    flat = fibers.curves.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan)
    out[finite] = flow(flat[finite])
    return out.reshape(shape)


def write_polydata(path: Path, curves: np.ndarray, progress: np.ndarray) -> None:
    """Write fibers as VTK XML PolyData lines, with progress as point data.

    Written directly rather than through a VTK binding: the format is a short
    XML document and this avoids adding a dependency for one output file.
    """

    points: list[str] = []
    connectivity: list[str] = []
    offsets: list[str] = []
    values: list[str] = []
    cursor = 0
    for index in range(curves.shape[0]):
        curve = curves[index]
        good = np.isfinite(curve).all(axis=1)
        if good.sum() < 2:
            continue
        for point, value in zip(curve[good], progress[index][good]):
            points.append(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
            values.append(f"{0.0 if not np.isfinite(value) else value:.6f}")
            connectivity.append(str(cursor))
            cursor += 1
        offsets.append(str(cursor))
    Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{cursor}" NumberOfLines="{len(offsets)}">\n'
        "      <PointData Scalars=\"outward_progress\">\n"
        '        <DataArray type="Float32" Name="outward_progress" format="ascii">\n'
        f"          {' '.join(values)}\n"
        "        </DataArray>\n"
        "      </PointData>\n"
        "      <Points>\n"
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        f"          {' '.join(points)}\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "      <Lines>\n"
        '        <DataArray type="Int32" Name="connectivity" format="ascii">\n'
        f"          {' '.join(connectivity)}\n"
        "        </DataArray>\n"
        '        <DataArray type="Int32" Name="offsets" format="ascii">\n'
        f"          {' '.join(offsets)}\n"
        "        </DataArray>\n"
        "      </Lines>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )
