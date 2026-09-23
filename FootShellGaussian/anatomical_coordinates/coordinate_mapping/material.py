"""What the footwear does at each place on the foot.

Once every point has an anatomical address, the natural next question is the
inverse one: pick a place on the foot and ask what the shoe does *there*. That
turns a shoe from a mesh into three functions over the canonical foot -

    g(u, v)        is this part of the anatomy covered at all?
    delta_in(u,v)  how far out along the fiber does material start?
    delta_out(u,v) how far out does it end?

and those are what the representation's base envelope is built from:

    B(u, v, r) = max{ g, delta_in - r, r - delta_out }

with ``B < 0`` inside the primary material volume. Every shoe is then described
on the same domain, which is what makes 27 different meshes comparable.

The measurement is made by walking each fiber and recording where it meets the
footwear surface. Crossings, not containment: the shoe meshes in this dataset
are not watertight - several thousand open edges each, and at least one with
zero signed volume - so there is no reliable inside test to make. A segment
meeting a triangle is well defined whatever the mesh's topology, and the first
and last crossing along a fiber are exactly the two extents wanted.

Keeping the *whole* crossing list rather than only its ends is deliberate. A
fiber through a sandal strap over an open dorsum crosses material twice with
nothing in between, and a fiber through nothing at all crosses zero times.
Those are the set-valued correspondences the representation asks for, and
throwing away the middle of the list would discard them.

Two conventions, chosen once and worth stating:

``r`` is arclength along the fiber in *canonical* space. It has to be, because
``B`` is evaluated at the anatomical coordinate of a point, so its units are
whatever the coordinate map produces. Physical distance in millimetres is also
recorded alongside, for reading, and because a clearance prior is easier to
argue about in millimetres than in canonical units.

Material starting *inside* the foot gives a negative ``delta_in``. That is not
an error to be clipped away: a rigid shoe on an undeformable foot model must
overlap somewhere, and where it does, the overlap is the measurement. It is
recorded separately from the crossings, because it cannot come from them - a
fiber that begins already inside material has no entry crossing to find.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .address import FiberTracer
from .lookup import CanonicalSemantics

#: One normalized unit, in millimetres.
MILLIMETRES = 262.5


@dataclass
class Sites:
    """The places on the canonical foot at which everything is measured.

    One per triangle of the inner surface, at its centroid. Fixed, shared by
    every shoe, and the same on every run - so a field measured on one shoe is
    directly comparable with the same field on another, entry by entry.
    """

    face: np.ndarray          # (N,) inner-surface triangle
    barycentric: np.ndarray   # (N, 3) always the centroid
    label: np.ndarray         # (N,) anatomical region
    point: np.ndarray         # (N, 3) the site in canonical space


def anatomical_sites(semantics: CanonicalSemantics, tracer: FiberTracer) -> Sites:
    """Every inner-surface triangle, at its centroid."""

    count = len(tracer.triangles)
    face = np.arange(count, dtype=np.int64)
    barycentric = np.full((count, 3), 1.0 / 3.0)
    point = tracer.triangles.mean(axis=1)
    return Sites(face, barycentric, semantics.inner_face_labels.copy(), point)


def crossings(
    curves: np.ndarray,
    canonical_length: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    neighbours: int = 64,
    chunk: int = 32768,
    merge_tolerance: float = 1.0e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each fiber meets a surface.

    ``curves`` are the fibers in the same frame as the mesh, sampled; entries
    past the end of a fiber are NaN. ``canonical_length`` is the arclength at
    each sample in canonical space, carried through so a crossing can be
    reported in the coordinate's own units as well as in millimetres.

    Returns three parallel arrays, sorted by site and then by distance: which
    site the crossing belongs to, its ``r``, and its distance in millimetres.
    """

    triangles = vertices[faces]
    tree = cKDTree(triangles.mean(axis=1))
    origin = triangles[:, 0]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]

    start, finish = curves[:, :-1], curves[:, 1:]
    usable = np.isfinite(start).all(-1) & np.isfinite(finish).all(-1)
    site, segment = np.nonzero(usable)
    if site.size == 0:
        empty = np.zeros(0)
        return np.zeros(0, dtype=np.int64), empty, empty

    # Physical distance along each fiber, accumulated in the mesh's own frame.
    lengths = np.linalg.norm(finish - start, axis=-1)
    lengths = np.where(np.isfinite(lengths), lengths, 0.0)
    physical = np.concatenate(
        (np.zeros((len(curves), 1)), np.cumsum(lengths, axis=1)), axis=1
    )

    hit_site, hit_r, hit_mm = [], [], []
    for begin in range(0, len(site), chunk):
        rows = slice(begin, begin + chunk)
        a, b = site[rows], segment[rows]
        head, tail = start[a, b], finish[a, b]
        direction = tail - head
        _, candidates = tree.query(
            0.5 * (head + tail), k=min(neighbours, len(triangles)), workers=-1
        )
        candidates = np.atleast_2d(np.asarray(candidates, dtype=np.int64))

        spread = np.broadcast_to(direction[:, None, :], (len(a), candidates.shape[1], 3))
        pvec = np.cross(spread, edge_b[candidates])
        determinant = np.einsum("nkj,nkj->nk", edge_a[candidates], pvec)
        alive = np.abs(determinant) > 1e-16
        inverse = np.divide(
            1.0, determinant, out=np.zeros_like(determinant), where=alive
        )
        tvec = head[:, None, :] - origin[candidates]
        u = np.einsum("nkj,nkj->nk", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge_a[candidates])
        v = np.einsum("nj,nkj->nk", direction, qvec) * inverse
        t = np.einsum("nkj,nkj->nk", edge_b[candidates], qvec) * inverse
        met = (
            alive & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
            & (t >= 0.0) & (t <= 1.0)
        )
        # Every crossing is kept, not just the first: one segment can pass
        # through both faces of a thin strap.
        where, slot = np.nonzero(met)
        if where.size == 0:
            continue
        fraction = t[where, slot]
        row, column = a[where], b[where]
        hit_site.append(row)
        hit_r.append(
            canonical_length[row, column]
            + fraction * (canonical_length[row, column + 1]
                          - canonical_length[row, column])
        )
        hit_mm.append(
            (physical[row, column] + fraction * lengths[row, column]) * MILLIMETRES
        )

    if not hit_site:
        empty = np.zeros(0)
        return np.zeros(0, dtype=np.int64), empty, empty
    hit_site = np.concatenate(hit_site)
    hit_r = np.concatenate(hit_r)
    hit_mm = np.concatenate(hit_mm)

    order = np.lexsort((hit_mm, hit_site))
    hit_site, hit_r, hit_mm = hit_site[order], hit_r[order], hit_mm[order]
    # A crossing on a shared edge is found from both triangles; the same place
    # met twice is one crossing.
    if len(hit_site) > 1:
        same = (hit_site[1:] == hit_site[:-1]) & (
            np.abs(hit_mm[1:] - hit_mm[:-1]) <= merge_tolerance * MILLIMETRES
        )
        keep = np.concatenate(([True], ~same))
        hit_site, hit_r, hit_mm = hit_site[keep], hit_r[keep], hit_mm[keep]
    return hit_site, hit_r, hit_mm


@dataclass
class MaterialFields:
    """The three fields, plus everything needed to defend them."""

    sites: Sites
    covered: np.ndarray        # (N,) bool - the sign of g
    delta_in: np.ndarray       # (N,) canonical arclength, NaN where uncovered
    delta_out: np.ndarray      # (N,)
    delta_in_mm: np.ndarray    # (N,) the same, in millimetres of real space
    delta_out_mm: np.ndarray   # (N,)
    layers: np.ndarray         # (N,) how many crossings - 2 is one wall
    penetration_mm: np.ndarray  # (N,) how far material reaches inside the foot
    reach: np.ndarray          # (N,) how far the fiber itself got, canonical
    crossing_site: np.ndarray   # the full crossing list, flat and sorted
    crossing_r: np.ndarray
    crossing_mm: np.ndarray

    def signed_delta_in(self) -> np.ndarray:
        """``delta_in`` with material inside the foot counted as negative.

        A fiber beginning inside material has no entry crossing, so the inward
        extent cannot be read from the crossings; it comes from the measured
        penetration instead. Where both exist, the penetration wins - it is the
        one that says material reaches past the skin.
        """

        out = self.delta_in.copy()
        buried = self.penetration_mm > 0.0
        out[buried] = -self.penetration_mm[buried] / MILLIMETRES
        return out


def base_field(
    g: np.ndarray, delta_in: np.ndarray, delta_out: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """``B(u, v, r) = max{g, delta_in - r, r - delta_out}``; negative in material.

    Written as a plain function of four arrays rather than a method, because
    the learned version substitutes networks for the first three and has to
    compose the same way.
    """

    return np.maximum(np.maximum(g, delta_in - r), r - delta_out)


def penetration_per_site(
    addresses: Path | dict, site_count: int, inside_code: int = 1
) -> np.ndarray:
    """How far footwear reaches inside the foot, at each site.

    Read from the address stage's own output. Points that land inside the
    anatomy were given the nearest triangle and a negative arclength there, so
    grouping them by that triangle puts the overlap where it belongs. The
    deepest point at a site wins, because the question is how far material
    reaches, not how far it reaches on average.
    """

    data = np.load(addresses) if not isinstance(addresses, dict) else addresses
    outcome = np.asarray(data["outcome"])
    face = np.asarray(data["face"])
    arclength = np.asarray(data["arclength"], dtype=np.float64)
    buried = (outcome == inside_code) & (face >= 0) & np.isfinite(arclength)
    out = np.zeros(site_count)
    if buried.any():
        depth = -arclength[buried] * MILLIMETRES
        np.maximum.at(out, face[buried], np.maximum(depth, 0.0))
    return out
