"""Does one address mean the same anatomical place in every shoe?

That is the claim the shared coordinate system makes, and it is checkable
without any new geometry. Take an address, put it into all 27 shoes, and ask
each shoe what it just received:

    A --place in shoe i--> a point in shoe i --query shoe i--> A'

If the coordinate system holds, A' is A for every shoe. The physical points
differ between shoes - that is the whole point, the shoes are different shapes -
but the address they carry does not.

Reported per address: the largest distance between the point it lands on in one
shoe and in another (how much geometry the address spans), and the largest
disagreement between the address sent in and the address read back (how much
meaning it loses on the way).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .address import ADDRESSED, AddressBook, FiberTracer


@dataclass
class CrossShoeReport:
    shoes: list[str]
    #: (A,) how far apart the same address lands across shoes, in mm.
    spread_mm: np.ndarray
    #: (A, S) distance from the address sent in to the one read back, in mm.
    drift_mm: np.ndarray
    #: (A, S) whether the shoe returned a usable address at all.
    returned: np.ndarray
    #: (A, S, 3) where the address landed in each shoe.
    placed: np.ndarray


def check(
    books: dict[str, AddressBook],
    tracer: FiberTracer,
    face: np.ndarray,
    barycentric: np.ndarray,
    r: np.ndarray,
    **trace,
) -> CrossShoeReport:
    """Send one set of addresses through every shoe and back."""

    names = list(books)
    count = len(face)
    placed = np.full((count, len(names), 3), np.nan)
    drift = np.full((count, len(names)), np.nan)
    returned = np.zeros((count, len(names)), dtype=bool)

    sent = np.einsum("ni,nij->nj", barycentric, tracer.triangles[face])
    for slot, name in enumerate(names):
        book = books[name]
        points = book.place(face, barycentric, r, **trace)
        placed[:, slot] = points
        live = np.isfinite(points).all(axis=1)
        if not live.any():
            continue
        back = book.query(points[live], exact=True, **trace)
        ok = back.outcome == ADDRESSED
        rows = np.nonzero(live)[0]
        returned[rows[ok], slot] = True
        # Compare the addresses as the points they name on the canonical foot,
        # because two triangles that share an edge can describe the same place.
        got = back.surface_point(tracer)
        drift[rows[ok], slot] = (
            np.linalg.norm(got[ok] - sent[rows[ok]], axis=1) * 262.5
        )

    spread = np.full(count, np.nan)
    for index in range(count):
        seen = placed[index][np.isfinite(placed[index]).all(axis=1)]
        if len(seen) > 1:
            gaps = np.linalg.norm(seen[:, None, :] - seen[None, :, :], axis=-1)
            spread[index] = gaps.max() * 262.5
    return CrossShoeReport(names, spread, drift, returned, placed)
