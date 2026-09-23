"""Anatomical addresses: which point on the foot, and how far out from it.

An address answers, for any point in the space around the foot, the question
"where on the foot is this, and how far out?". It has two halves:

    (u, v)   a triangle of the foot surface and a position inside it
    r        outward progress along the fiber through that point, 0 on the
             skin and 1 on the outer envelope

The outward half is a field lookup - ``harmonic_r`` is stored per canonical
vertex and interpolates. The surface half is not, because "which point on the
foot" is defined by following the fiber through a point all the way back to the
skin. That is the correspondence map, and this module computes it.

Two ways to get it, both classical:

    tracing        integrate the direction field backwards from the query
                   point until it meets the skin. Exact up to the integrator,
                   and costs an integration per query.
    correspondence solve for the whole map at once, then interpolate. The
                   correspondence map phi is constant along fibers, so it
                   satisfies the transport equation W . grad phi = 0 with
                   phi(x) = x on the skin (Yezzi and Prince, "An Eulerian PDE
                   approach for computing tissue thickness", IEEE TMI 2003;
                   the correspondence and gridding form is their ECCV 2002).

The pure Eulerian solve carries numerical diffusion: each upwind step mixes
neighbouring fibers a little, and over a long fiber that smears the
correspondence. So this uses the hybrid Rocha et al. describe - trace once from
every canonical *vertex* with the accurate integrator, store the result as a
table, and interpolate between vertices at query time. The traces are
Lagrangian so they do not diffuse; the lookup is Eulerian so it is O(1).

That trade only works because the canonical anatomy is shared. Under the older
tetrahedral cage the volume was per shoe and this table would have had to be
rebuilt 27 times; under a flow map the canonical mesh is the only mesh, so the
table is built once and every shoe reads it.

``trace_in`` remains available per query and is what the accuracy audit
compares against, so the interpolated answer is never trusted on argument
alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .lookup import CanonicalSemantics

#: Landing outcomes. Anything other than ``REACHED`` has no address.
REACHED, LEFT_DOMAIN, WEAK_DIRECTION, OUT_OF_STEPS = 0, 1, 2, 3

STATUS_NAMES = {
    REACHED: "reached the skin",
    LEFT_DOMAIN: "left the domain",
    WEAK_DIRECTION: "direction field too weak",
    OUT_OF_STEPS: "ran out of steps",
}


@dataclass
class Landing:
    """Where a set of fibers met the inner surface."""

    face: np.ndarray          # (N,) index into inner_faces, -1 where none
    barycentric: np.ndarray   # (N, 3) weights on that triangle
    point: np.ndarray         # (N, 3) the landing point itself
    arclength: np.ndarray     # (N,) fiber length from the query point to it
    label: np.ndarray         # (N,) which anatomical region, -1 where none
    status: np.ndarray        # (N,) one of the codes above

    @property
    def found(self) -> np.ndarray:
        return self.status == REACHED


class FiberTracer:
    """Integrates the canonical direction field, in either direction.

    The field is read unchanged from the original fiber work and interpolated
    barycentrically, which makes it continuous across cell faces, so a curve
    does not kink where it crosses from one tetrahedron into the next.
    """

    def __init__(
        self,
        semantics: CanonicalSemantics,
        directions: np.ndarray,
        device=None,
    ) -> None:
        self.semantics = semantics
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        self.directions = np.divide(
            directions, norm, out=np.zeros_like(directions), where=norm > 1e-9
        )
        self.device = device
        faces = semantics.inner_vertex_indices[semantics.inner_faces]
        self.triangles = semantics.vertices[faces]
        self.face_labels = semantics.inner_face_labels
        self._surface_tree = cKDTree(self.triangles.mean(axis=1))
        self._edge_a = self.triangles[:, 1] - self.triangles[:, 0]
        self._edge_b = self.triangles[:, 2] - self.triangles[:, 0]
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Which tetrahedron lies across each face of each tetrahedron.

        An integration step is short compared with a cell, so after a step a
        point is almost always still in the cell it was in, or in one next to
        it. Testing that handful of cells first turns the common case into
        pure arithmetic and leaves the spatial index to handle the rest.
        """

        tets = self.semantics.tetrahedra
        faces = np.stack(
            (tets[:, [1, 2, 3]], tets[:, [0, 2, 3]],
             tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]),
            axis=1,
        ).reshape(-1, 3)
        key = np.sort(faces, axis=1)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        ordered = key[order]
        same = np.all(ordered[:-1] == ordered[1:], axis=1)
        pairs = np.nonzero(same)[0]
        neighbours = np.full(len(faces), -1, dtype=np.int64)
        left, right = order[pairs], order[pairs + 1]
        neighbours[left] = right // 4
        neighbours[right] = left // 4
        self.neighbours = neighbours.reshape(-1, 4)

    def _walk_candidates(self, hint: np.ndarray) -> np.ndarray:
        """The hinted cell, its face neighbours, and theirs."""

        first = self.neighbours[hint]                        # (N, 4)
        second = self.neighbours[np.maximum(first, 0)]        # (N, 4, 4)
        second = np.where(first[:, :, None] >= 0, second, -1).reshape(len(hint), -1)
        return np.concatenate((hint[:, None], first, second), axis=1)

    def _barycentric(
        self, points: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Test each point against its own candidate cells, on the GPU."""

        import torch

        semantics = self.semantics
        device = self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not hasattr(self, "_origin_gpu"):
            self._origin_gpu = torch.as_tensor(
                semantics.origin, dtype=torch.float64, device=device
            )
            self._inverse_gpu = torch.as_tensor(
                semantics.inverse, dtype=torch.float64, device=device
            )
        found = np.full(len(points), -1, dtype=np.int64)
        weights = np.zeros((len(points), 4))
        block = 65536
        for start in range(0, len(points), block):
            stop = min(start + block, len(points))
            pts = torch.as_tensor(
                points[start:stop], dtype=torch.float64, device=device
            )
            cand = torch.as_tensor(candidates[start:stop], device=device)
            valid = cand >= 0
            safe = cand.clamp(min=0)
            relative = pts[:, None, :] - self._origin_gpu[safe]
            full = torch.einsum("bkij,bkj->bki", self._inverse_gpu[safe], relative)
            full = torch.cat((1.0 - full.sum(dim=-1, keepdim=True), full), dim=-1)
            inside = valid & (full >= -semantics.tolerance).all(dim=-1)
            # Deterministic on a shared face: smallest cell id, as ``locate``.
            ranked = torch.where(inside, safe, torch.full_like(safe, 2**31))
            best, slot = ranked.min(dim=1)
            hit = best < 2**31
            rows = torch.arange(stop - start, device=device)
            found[start:stop] = torch.where(
                hit, best, torch.full_like(best, -1)
            ).cpu().numpy()
            weights[start:stop] = full[rows, slot].cpu().numpy()
        return found, weights, found >= 0

    def _locate(
        self, points: np.ndarray, hint: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cell and barycentric weights, trying the hinted neighbourhood first.

        Identical answers to ``locate_fast`` - the same barycentric test
        decides, only the order in which cells are offered to it changes.
        """

        count = len(points)
        tetrahedron = np.full(count, -1, dtype=np.int64)
        barycentric = np.zeros((count, 4))
        pending = np.arange(count)
        if hint is not None and count:
            usable = hint >= 0
            if usable.any():
                rows = pending[usable]
                found, weights, hit = self._barycentric(
                    points[rows], self._walk_candidates(hint[usable])
                )
                tetrahedron[rows] = found
                barycentric[rows] = weights
                pending = np.concatenate((pending[~usable], rows[~hit]))
        if pending.size:
            result = self.semantics.locate_fast(
                points[pending], device=self.device, fallback=False
            )
            tetrahedron[pending] = result.tetrahedron
            barycentric[pending] = result.barycentric
        return tetrahedron, barycentric

    # -- field ------------------------------------------------------------
    def _sample(
        self, points: np.ndarray, hint: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unit direction, outward progress, containment, and the cell.

        Both fields are read from the same barycentric weights. Splitting them
        would double the cost of every integration step, and locating a point
        is the whole cost of a step. The cell is returned so the caller can
        hand it back as the hint for the next step.
        """

        tetrahedron, barycentric = self._locate(points, hint)
        velocity = np.zeros_like(points)
        progress = np.full(len(points), np.inf)
        ok = tetrahedron >= 0
        if ok.any():
            corners = self.semantics.tetrahedra[tetrahedron[ok]]
            weights = barycentric[ok]
            raw = np.einsum("ni,nij->nj", weights, self.directions[corners])
            length = np.linalg.norm(raw, axis=1, keepdims=True)
            velocity[ok] = np.divide(
                raw, length, out=np.zeros_like(raw), where=length > 1e-9
            )
            progress[ok] = np.einsum(
                "ni,ni->n", weights, self.semantics.harmonic_r[corners]
            )
        return velocity, progress, ok, tetrahedron

    def _step(
        self, points: np.ndarray, sign: float, size: float,
        hint: np.ndarray | None = None,
    ) -> np.ndarray:
        """One classical RK4 step along ``sign`` times the unit field.

        A stage that lands outside the mesh returns a zero direction, which
        would quietly corrupt the combination. Those rows fall back to the
        Euler step the first stage already justifies, rather than averaging in
        a value that is not data.
        """

        k1 = sign * self._sample(points, hint)[0]
        k2, _, ok2, _ = self._sample(points + 0.5 * size * k1, hint)
        k3, _, ok3, _ = self._sample(points + 0.5 * size * sign * k2, hint)
        k4, _, ok4, _ = self._sample(points + size * sign * k3, hint)
        advance = (size / 6.0) * (
            k1 + 2.0 * sign * k2 + 2.0 * sign * k3 + sign * k4
        )
        degraded = ~(ok2 & ok3 & ok4)
        if degraded.any():
            advance[degraded] = size * k1[degraded]
        return advance

    def _cross_surface(
        self, start: np.ndarray, finish: np.ndarray, neighbours: int = 48
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment against the inner surface: which triangle, and where.

        Moller-Trumbore on the segment rather than a bisection on a signed
        distance, because it returns the triangle and the position inside it
        directly - which is exactly the (u, v) half of the address.
        """

        count = len(start)
        face = np.full(count, -1, dtype=np.int64)
        weights = np.zeros((count, 3))
        fraction = np.full(count, np.nan)
        if count == 0:
            return face, weights, fraction

        midpoint = 0.5 * (start + finish)
        _, candidates = self._surface_tree.query(
            midpoint, k=min(neighbours, len(self.triangles)), workers=-1
        )
        candidates = np.atleast_2d(np.asarray(candidates, dtype=np.int64))

        direction = finish - start
        origin = self.triangles[candidates, 0]
        edge_a = self._edge_a[candidates]
        edge_b = self._edge_b[candidates]
        spread = np.broadcast_to(direction[:, None, :], edge_b.shape)
        pvec = np.cross(spread, edge_b)
        determinant = np.einsum("nkj,nkj->nk", edge_a, pvec)
        usable = np.abs(determinant) > 1e-16
        inverse = np.divide(
            1.0, determinant, out=np.zeros_like(determinant), where=usable
        )
        tvec = start[:, None, :] - origin
        u = np.einsum("nkj,nkj->nk", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge_a)
        v = np.einsum("nj,nkj->nk", direction, qvec) * inverse
        t = np.einsum("nkj,nkj->nk", edge_b, qvec) * inverse
        hit = (
            usable
            & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
            & (t >= -1e-9) & (t <= 1.0 + 1e-9)
        )
        # The first crossing along the segment is the landing; a fiber passing
        # near a fold could otherwise be attributed to a farther triangle.
        ordered = np.where(hit, t, np.inf)
        slot = ordered.argmin(axis=1)
        rows = np.arange(count)
        best = ordered[rows, slot]
        landed = np.isfinite(best)
        if landed.any():
            index = rows[landed]
            picked = slot[landed]
            face[index] = candidates[index, picked]
            uu = np.clip(u[index, picked], 0.0, 1.0)
            vv = np.clip(v[index, picked], 0.0, 1.0)
            total = np.maximum(uu + vv, 1.0)
            uu, vv = uu / total, vv / total
            weights[index] = np.stack((1.0 - uu - vv, uu, vv), axis=1)
            fraction[index] = best[landed]
        return face, weights, fraction

    def project_to_surface(
        self, points: np.ndarray, neighbours: int = 24, chunk: int = 32768
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest point of the inner surface, as a triangle and weights.

        The interpolated correspondence lands near the skin but not exactly on
        it, because interpolation between vertices cuts the corner of a curved
        surface. Projecting repairs that, and is also what turns a position
        into the (u, v) half of an address.
        """

        points = np.asarray(points, dtype=np.float64)
        if len(points) > chunk:
            parts = [
                self.project_to_surface(points[start : start + chunk],
                                        neighbours, chunk)
                for start in range(0, len(points), chunk)
            ]
            return tuple(
                np.concatenate([part[slot] for part in parts]) for slot in range(3)
            )
        _, candidates = self._surface_tree.query(
            points, k=min(neighbours, len(self.triangles)), workers=-1
        )
        candidates = np.atleast_2d(np.asarray(candidates, dtype=np.int64))

        corner = self.triangles[candidates]                 # (N, K, 3, 3)
        a, b, c = corner[:, :, 0], corner[:, :, 1], corner[:, :, 2]
        ab, ac = b - a, c - a
        ap = points[:, None, :] - a
        d1 = np.einsum("nkj,nkj->nk", ab, ap)
        d2 = np.einsum("nkj,nkj->nk", ac, ap)
        bp = points[:, None, :] - b
        d3 = np.einsum("nkj,nkj->nk", ab, bp)
        d4 = np.einsum("nkj,nkj->nk", ac, bp)
        cp = points[:, None, :] - c
        d5 = np.einsum("nkj,nkj->nk", ab, cp)
        d6 = np.einsum("nkj,nkj->nk", ac, cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2

        # Ericson's region test, written as a cascade of overrides: start from
        # the interior solution and let each boundary case replace it.
        total = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1.0)
        v = vb / total
        w = vc / total
        u = 1.0 - v - w

        def override(mask, uu, vv, ww):
            nonlocal u, v, w
            u = np.where(mask, uu, u)
            v = np.where(mask, vv, v)
            w = np.where(mask, ww, w)

        edge = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
        override((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - edge, edge, 0.0)
        edge = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
        override((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - edge, 0.0, edge)
        span = (d4 - d3) + (d5 - d6)
        edge = np.divide(d4 - d3, span, out=np.zeros_like(d4), where=span != 0)
        override((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), 0.0, 1.0 - edge, edge)
        override((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)
        override((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)
        override((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)

        nearest = a + v[..., None] * ab + w[..., None] * ac
        gap = np.linalg.norm(nearest - points[:, None, :], axis=-1)
        slot = gap.argmin(axis=1)
        rows = np.arange(len(points))
        face = candidates[rows, slot]
        weights = np.stack((u[rows, slot], v[rows, slot], w[rows, slot]), axis=1)
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return face, weights, gap[rows, slot]

    # -- tracing ----------------------------------------------------------
    def trace_in(
        self,
        points: np.ndarray,
        step: float = 0.004,
        max_steps: int = 500,
        near_surface: float = 0.08,
        chunk: int = 32768,
    ) -> Landing:
        """Follow fibers inward from ``points`` until they meet the skin.

        The surface test only runs once a fiber is close to the skin. It is
        the expensive part of a step, and a fiber out at high progress cannot
        reach the surface within one step of the size used here.

        Fibers are independent, so a long list is traced in chunks purely to
        bound peak memory. The cost per point falls steeply with batch size,
        so the chunk is large.
        """

        points = np.asarray(points, dtype=np.float64)
        if len(points) > chunk:
            parts = [
                self.trace_in(points[start : start + chunk], step, max_steps,
                              near_surface, chunk)
                for start in range(0, len(points), chunk)
            ]
            return Landing(*(
                np.concatenate([getattr(part, name) for part in parts])
                for name in ("face", "barycentric", "point", "arclength",
                             "label", "status")
            ))
        count = len(points)
        face = np.full(count, -1, dtype=np.int64)
        weights = np.zeros((count, 3))
        landing = np.full((count, 3), np.nan)
        arclength = np.zeros(count)
        status = np.full(count, OUT_OF_STEPS, dtype=np.int8)

        alive = np.arange(count)
        current = points.copy()
        velocity, progress, inside, cell = self._sample(current)
        status[alive[~inside]] = LEFT_DOMAIN
        alive, current = alive[inside], current[inside]
        velocity, progress, cell = velocity[inside], progress[inside], cell[inside]

        for _ in range(max_steps):
            if len(alive) == 0:
                break
            weak = np.linalg.norm(velocity, axis=1) < 1e-9
            if weak.any():
                status[alive[weak]] = WEAK_DIRECTION
                keep = ~weak
                alive, current = alive[keep], current[keep]
                progress, cell = progress[keep], cell[keep]
                if len(alive) == 0:
                    break

            advance = self._step(current, -1.0, step, cell)
            finish = current + advance
            distance = np.linalg.norm(advance, axis=1)
            velocity, ahead, inside, next_cell = self._sample(finish, cell)

            # Two reasons to ask whether this step crossed the skin: the fiber
            # was already close to it, or the step ended outside the mesh. The
            # second is the important one - the mesh stops *at* the skin, so a
            # step that leaves it has usually gone through, and treating that
            # as a failure would throw away a fiber that in fact landed.
            settled = np.zeros(len(alive), dtype=bool)
            check = np.nonzero((progress < near_surface) | ~inside)[0]
            if check.size:
                hit, hit_weights, fraction = self._cross_surface(
                    current[check], finish[check]
                )
                crossed = hit >= 0
                if crossed.any():
                    local = check[crossed]
                    rows = alive[local]
                    face[rows] = hit[crossed]
                    weights[rows] = hit_weights[crossed]
                    part = fraction[crossed]
                    landing[rows] = current[local] + part[:, None] * advance[local]
                    arclength[rows] += distance[local] * part
                    status[rows] = REACHED
                    settled[local] = True

            lost = ~inside & ~settled
            if lost.any():
                status[alive[lost]] = LEFT_DOMAIN
                settled = settled | lost

            keep = ~settled
            arclength[alive[keep]] += distance[keep]
            alive, current = alive[keep], finish[keep]
            progress, velocity, cell = ahead[keep], velocity[keep], next_cell[keep]

        label = np.full(count, -1, dtype=np.int16)
        ok = face >= 0
        label[ok] = self.face_labels[face[ok]]
        return Landing(face, weights, landing, arclength, label, status)

    def trace_out(
        self,
        face: np.ndarray,
        barycentric: np.ndarray,
        target: np.ndarray,
        step: float = 0.004,
        max_steps: int = 500,
        level_tolerance: float = 1.0e-3,
    ) -> np.ndarray:
        """From a surface address outward to a given progress ``target``.

        The inverse of ``trace_in``: it returns the canonical point whose fiber
        starts at ``(face, barycentric)`` and whose harmonic progress equals
        ``target``. The crossing of that level is found by bisection within the
        final step, so the progress of the returned point is accurate to far
        better than the step size.
        """

        face = np.asarray(face, dtype=np.int64)
        barycentric = np.asarray(barycentric, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        surface = np.einsum("ni,nij->nj", barycentric, self.triangles[face])
        out = np.full((len(face), 3), np.nan)

        settled = target <= 1e-12
        out[settled] = surface[settled]
        alive = np.nonzero(~settled)[0]
        if alive.size == 0:
            return out

        # Lift off the surface before integrating. The direction field is
        # defined on the closed volume, and a point sitting exactly on the
        # boundary can locate into a cell on either side of it.
        current = surface[alive]
        velocity, _, inside, cell = self._sample(current)
        lift = inside & (np.linalg.norm(velocity, axis=1) > 1e-9)
        current = np.where(lift[:, None], current + 1e-6 * velocity, current)

        for _ in range(max_steps):
            if len(alive) == 0:
                break
            velocity, before, inside, cell = self._sample(current, cell)
            dead = ~inside | (np.linalg.norm(velocity, axis=1) < 1e-9)
            if dead.any():
                keep = ~dead
                alive, current = alive[keep], current[keep]
                before, cell = before[keep], cell[keep]
                if len(alive) == 0:
                    break

            advance = self._step(current, 1.0, step, cell)
            finish = current + advance
            after = self._sample(finish, cell)[1]
            want = target[alive]

            reached = np.isfinite(after) & (after >= want) & (before <= want + 1e-12)
            if reached.any():
                out[alive[reached]] = self._bisect(
                    current[reached], advance[reached], want[reached], cell[reached]
                )

            # A target at the envelope is not crossed, it is arrived at: the
            # mesh ends there, so the fiber stops by leaving. Those are
            # resolved by finding where the step left, not by a level search.
            departed = ~np.isfinite(after) & ~reached
            edge = departed & (want >= before - level_tolerance)
            if edge.any():
                out[alive[edge]] = self._last_inside(
                    current[edge], advance[edge], cell[edge]
                )

            done = reached | departed
            keep = ~done
            alive, current, cell = alive[keep], finish[keep], cell[keep]
        return out

    def trace_path(
        self,
        face: np.ndarray,
        barycentric: np.ndarray,
        samples: int = 400,
        step: float = 0.002,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trace outward from surface sites, keeping the whole curve.

        ``trace_out`` answers "where is this fiber at progress r"; this answers
        "where does this fiber go", which is what you need to ask what the
        fiber passes through. Samples past the end of a fiber are NaN, so the
        caller can tell a short fiber from a long one.

        Returns the curve, the arclength travelled to each sample, and the
        outward progress at it - all in canonical space, all indexed
        ``(site, sample)``.
        """

        face = np.asarray(face, dtype=np.int64)
        barycentric = np.asarray(barycentric, dtype=np.float64)
        count = len(face)
        curves = np.full((count, samples, 3), np.nan)
        arclength = np.full((count, samples), np.nan)
        progress = np.full((count, samples), np.nan)

        current = np.einsum("ni,nij->nj", barycentric, self.triangles[face])
        curves[:, 0] = current
        arclength[:, 0] = 0.0
        progress[:, 0] = 0.0

        # Lift off the boundary before integrating, for the same reason
        # ``trace_out`` does: a point exactly on the surface can locate into a
        # cell on either side of it.
        velocity, _, inside, cell = self._sample(current)
        lift = inside & (np.linalg.norm(velocity, axis=1) > 1e-9)
        current = np.where(lift[:, None], current + 1e-6 * velocity, current)

        alive = np.arange(count)
        travelled = np.zeros(count)
        for index in range(1, samples):
            if len(alive) == 0:
                break
            velocity, _, inside, cell = self._sample(current, cell)
            dead = ~inside | (np.linalg.norm(velocity, axis=1) < 1e-9)
            if dead.any():
                keep = ~dead
                alive, current, cell = alive[keep], current[keep], cell[keep]
                if len(alive) == 0:
                    break

            advance = self._step(current, 1.0, step, cell)
            current = current + advance
            _, ahead, still, cell = self._sample(current, cell)
            travelled[alive] += np.linalg.norm(advance, axis=1)
            curves[alive, index] = current
            arclength[alive, index] = travelled[alive]
            progress[alive, index] = np.where(still, ahead, np.nan)
            if not still.all():
                # The last point is kept: it is where the fiber left the
                # domain, which for an outward trace is the envelope.
                alive, current, cell = alive[still], current[still], cell[still]
        return curves, arclength, progress

    def _bisect(
        self, base: np.ndarray, delta: np.ndarray, want: np.ndarray,
        hint: np.ndarray, rounds: int = 40,
    ) -> np.ndarray:
        """Where along ``base + t * delta`` the progress reaches ``want``."""

        low = np.zeros(len(base))
        high = np.ones(len(base))
        for _ in range(rounds):
            middle = 0.5 * (low + high)
            value = self._sample(base + middle[:, None] * delta, hint)[1]
            below = np.isfinite(value) & (value < want)
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        return base + (0.5 * (low + high))[:, None] * delta

    def _last_inside(
        self, base: np.ndarray, delta: np.ndarray, hint: np.ndarray,
        rounds: int = 30,
    ) -> np.ndarray:
        """Where along ``base + t * delta`` the step left the mesh."""

        low = np.zeros(len(base))
        high = np.ones(len(base))
        for _ in range(rounds):
            middle = 0.5 * (low + high)
            here = self._sample(base + middle[:, None] * delta, hint)[2]
            low = np.where(here, middle, low)
            high = np.where(here, high, middle)
        return base + low[:, None] * delta


@dataclass
class CanonicalCorrespondence:
    """Where every canonical vertex's fiber starts on the skin.

    Built once, read by all 27 shoes. The stored quantity is the landing
    *point*, not the triangle index, because a point interpolates between
    vertices and an index does not. The triangle is recovered at query time by
    projection, which also absorbs the small interpolation error.
    """

    origin: np.ndarray        # (V, 3)
    face: np.ndarray          # (V,)
    barycentric: np.ndarray   # (V, 3)
    arclength: np.ndarray     # (V,)
    label: np.ndarray         # (V,)
    status: np.ndarray        # (V,)

    @classmethod
    def build(cls, tracer: FiberTracer, **kwargs) -> "CanonicalCorrespondence":
        semantics = tracer.semantics
        landing = tracer.trace_in(semantics.vertices, **kwargs)
        origin = landing.point.copy()
        # A vertex that already lies on the skin is its own origin. Tracing
        # from one would step off the boundary before it could land.
        surface = np.zeros(len(semantics.vertices), dtype=bool)
        surface[semantics.inner_vertex_indices] = True
        on_skin = surface & (semantics.harmonic_r <= 1e-12)
        if on_skin.any():
            origin[on_skin] = semantics.vertices[on_skin]
            landing.arclength[on_skin] = 0.0
            landing.status[on_skin] = REACHED
        return cls(
            origin, landing.face, landing.barycentric,
            landing.arclength, landing.label, landing.status,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            origin=self.origin.astype(np.float32),
            face=self.face,
            barycentric=self.barycentric.astype(np.float32),
            arclength=self.arclength.astype(np.float32),
            label=self.label,
            status=self.status,
        )

    @classmethod
    def load(cls, path: Path) -> "CanonicalCorrespondence":
        data = np.load(path)
        return cls(
            np.asarray(data["origin"], dtype=np.float64),
            np.asarray(data["face"], dtype=np.int64),
            np.asarray(data["barycentric"], dtype=np.float64),
            np.asarray(data["arclength"], dtype=np.float64),
            np.asarray(data["label"], dtype=np.int16),
            np.asarray(data["status"], dtype=np.int8),
        )


#: Query outcomes. Only ``ADDRESSED`` carries a usable ``(u, v, r)``.
ADDRESSED, INSIDE_ANATOMY, BEYOND_DOMAIN = 0, 1, 2
NOT_ANATOMICAL, UNRESOLVED, UNVERIFIED = 3, 4, 5

OUTCOME_NAMES = {
    ADDRESSED: "addressed",
    INSIDE_ANATOMY: "inside the anatomy",
    BEYOND_DOMAIN: "beyond the domain",
    NOT_ANATOMICAL: "fiber ends on a modelling cut",
    UNRESOLVED: "fiber could not be followed",
    UNVERIFIED: "address did not return to its point",
}


@dataclass
class Address:
    """Where each queried point sits on, and out from, the anatomy."""

    face: np.ndarray          # (N,) inner triangle; (u, v) is this plus weights
    barycentric: np.ndarray   # (N, 3) position inside that triangle
    r: np.ndarray             # (N,) outward progress, 0 on the skin, 1 outside
    rho: np.ndarray           # (N,) the quadratic semantic scalar
    label: np.ndarray         # (N,) which anatomical region the fiber lands on
    arclength: np.ndarray     # (N,) distance out along the fiber, normalized
    canonical: np.ndarray     # (N, 3) the query point in canonical space
    outcome: np.ndarray       # (N,) one of the codes above
    #: Distance in mm between the query point and where its own address puts
    #: it back. NaN unless ``verify`` was asked for.
    residual: np.ndarray

    @property
    def addressed(self) -> np.ndarray:
        return self.outcome == ADDRESSED

    def surface_point(self, tracer: "FiberTracer") -> np.ndarray:
        """The physical point on the canonical foot that each address names."""

        out = np.full((len(self.face), 3), np.nan)
        ok = self.face >= 0
        if ok.any():
            out[ok] = np.einsum(
                "ni,nij->nj", self.barycentric[ok], tracer.triangles[self.face[ok]]
            )
        return out


class AddressBook:
    """Addresses for one shoe: a point in its space, to and from anatomy.

    The two directions are separate journeys through the same two maps::

        query   shoe point --flow back--> canonical --fiber in--> (u, v, r)
        place   (u, v, r) --fiber out--> canonical --flow forward--> shoe point

    ``query`` has a fast path and an exact one. The fast path reads the
    precomputed correspondence and interpolates it; the exact path integrates
    the fiber for this particular point. They are the same quantity computed
    two ways, so the difference between them is a measurement of the
    interpolation error rather than an unknown.
    """

    def __init__(
        self,
        semantics: CanonicalSemantics,
        tracer: FiberTracer,
        correspondence: CanonicalCorrespondence,
        to_canonical,
        to_shoe,
    ) -> None:
        self.semantics = semantics
        self.tracer = tracer
        self.correspondence = correspondence
        self.to_canonical = to_canonical
        self.to_shoe = to_shoe
        self.cut_label = (
            semantics.boundary_label_names.index("knee_truncation")
            if "knee_truncation" in semantics.boundary_label_names else -1
        )

    def query(
        self,
        points: np.ndarray,
        exact: bool = True,
        verify: bool = False,
        tolerance_mm: float = 0.1,
        surface_tolerance_mm: float = 0.01,
        **trace,
    ) -> Address:
        """Address every point of ``points``, given in this shoe's frame.

        ``verify`` sends each address back through ``place`` and measures how
        far it lands from the point it came from. The direction field has a
        few singular cells inherited from the original fiber work, and a fiber
        threading one of them can arrive somewhere else entirely; that is not
        visible in the address itself but it is obvious in the round trip. An
        address that misses by more than ``tolerance_mm`` is reported as
        unverified rather than returned as though it were sound.
        """

        points = np.asarray(points, dtype=np.float64)
        canonical = np.asarray(self.to_canonical(points), dtype=np.float64)
        count = len(canonical)

        located = self.semantics.locate_fast(canonical, device=self.tracer.device)
        r = self.semantics.harmonic(located)
        rho = (
            self.semantics.semantic_rho(located)
            if self.semantics.semantic else np.full(count, np.nan)
        )
        inside = self.semantics.inside_inner_boundary_fast(
            canonical, device=self.tracer.device
        )

        face = np.full(count, -1, dtype=np.int64)
        weights = np.zeros((count, 3))
        arclength = np.full(count, np.nan)
        label = np.full(count, -1, dtype=np.int16)

        # The tetrahedra fill the shell between the anatomy and the envelope,
        # so landing in one *is* the test for being in the addressable region.
        # A point that no cell contains is either inside the anatomy or past
        # the envelope, and only then does the parity test decide which - it is
        # a coin flip for a point sitting exactly on the skin, where the cell
        # test is exact.
        outcome = np.full(count, BEYOND_DOMAIN, dtype=np.int8)
        buried = ~located.found & inside
        outcome[buried] = INSIDE_ANATOMY

        # A point inside the anatomy has no address - there is no fiber through
        # it - but it is not featureless either, and in practice it is a shoe
        # surface interpenetrating the fitted foot by a fraction of a
        # millimetre. Record the nearest place on the foot and how far in it
        # sits, as a negative arclength, so the next stage has the measurement
        # rather than a hole. The outcome still says this is not an address.
        grazing = np.zeros(count, dtype=bool)
        if buried.any():
            rows = np.nonzero(buried)[0]
            near, near_weights, depth = self.tracer.project_to_surface(canonical[rows])
            face[rows] = near
            weights[rows] = near_weights
            arclength[rows] = -depth
            label[rows] = self.tracer.face_labels[near]
            # Barely inside is on the surface, and the surface is addressable.
            touching = depth * 262.5 <= surface_tolerance_mm
            grazing[rows[touching]] = True
            # Both scalars are zero on the skin by construction, and an
            # addressed point should carry no missing field.
            r[rows[touching]] = 0.0
            rho[rows[touching]] = 0.0
            arclength[rows[touching]] = 0.0
            outcome[rows[touching]] = ADDRESSED

        live = np.nonzero(located.found)[0]
        if live.size:
            if exact:
                resolved = self._by_tracing(canonical[live], **trace)
            else:
                resolved = self._by_interpolation(
                    canonical[live], located, live, **trace
                )
            face[live], weights[live], arclength[live], label[live], failed = resolved
            outcome[live] = np.where(failed, UNRESOLVED, ADDRESSED)
            cut = (label[live] == self.cut_label) & ~failed
            if cut.any():
                outcome[live[cut]] = NOT_ANATOMICAL
        cut = grazing & (label == self.cut_label)
        if cut.any():
            outcome[cut] = NOT_ANATOMICAL

        residual = np.full(count, np.nan)
        address = Address(
            face, weights, r, rho, label, arclength, canonical, outcome, residual
        )
        if verify:
            check = np.nonzero(address.addressed)[0]
            if check.size:
                back = self.place(
                    face[check], weights[check], r[check], **trace
                )
                residual[check] = np.linalg.norm(
                    back - points[check], axis=1
                ) * 262.5
                missed = ~(residual[check] <= tolerance_mm)
                outcome[check[missed]] = UNVERIFIED
        return address

    def _by_tracing(self, canonical: np.ndarray, **trace):
        landing = self.tracer.trace_in(canonical, **trace)
        return (
            landing.face, landing.barycentric, landing.arclength,
            landing.label, ~landing.found,
        )

    def _by_interpolation(self, canonical, located, live, **trace):
        """Read the correspondence table at the containing cell, and project.

        A cell whose corners did not all resolve carries no trustworthy value,
        so those points fall through to a real trace rather than being handed
        an interpolation of a failure.
        """

        corners = self.semantics.tetrahedra[located.tetrahedron[live]]
        barycentric = located.barycentric[live]
        usable = (self.correspondence.status[corners] == REACHED).all(axis=1)

        face = np.full(len(canonical), -1, dtype=np.int64)
        weights = np.zeros((len(canonical), 3))
        arclength = np.full(len(canonical), np.nan)
        label = np.full(len(canonical), -1, dtype=np.int16)
        failed = ~usable

        if usable.any():
            rows = np.nonzero(usable)[0]
            origin = np.einsum(
                "ni,nij->nj", barycentric[rows],
                self.correspondence.origin[corners[rows]],
            )
            face[rows], weights[rows], _ = self.tracer.project_to_surface(origin)
            arclength[rows] = np.einsum(
                "ni,ni->n", barycentric[rows],
                self.correspondence.arclength[corners[rows]],
            )
            label[rows] = self.tracer.face_labels[face[rows]]
        if failed.any():
            rows = np.nonzero(failed)[0]
            landing = self.tracer.trace_in(canonical[rows], **trace)
            face[rows] = landing.face
            weights[rows] = landing.barycentric
            arclength[rows] = landing.arclength
            label[rows] = landing.label
            failed[rows] = ~landing.found
        return face, weights, arclength, label, failed

    def place(
        self, face: np.ndarray, barycentric: np.ndarray, r: np.ndarray, **trace
    ) -> np.ndarray:
        """The inverse: where an address sits in this shoe's space."""

        canonical = self.tracer.trace_out(face, barycentric, r, **trace)
        out = np.full_like(canonical, np.nan)
        ok = np.isfinite(canonical).all(axis=1)
        if ok.any():
            out[ok] = self.to_shoe(canonical[ok])
        return out
