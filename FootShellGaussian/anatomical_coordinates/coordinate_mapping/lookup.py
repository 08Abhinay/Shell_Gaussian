"""Canonical semantic lookup, shared by every shoe.

This is where the flow formulation pays off structurally. the older map lookup
builds a tetrahedron spatial index *per instance*, because each instance has its
own deformed volume. Under a flow map the instance geometry is carried by the
map itself, so the only tetrahedral mesh that exists is the canonical one - it
is identical for all 27 shoes and its index is built once.

A query becomes:

    x  --flow backwards-->  q  --canonical index-->  (tetrahedron, barycentric)
                                                 --> harmonic r, semantic rho

Nothing per-shoe is precomputed beyond the velocity field. The canonical
harmonic field, the quadratic semantic scalar and the boundary labels are all
read from the existing the canonical volume build / the older address stage artifacts unchanged; none of that
work is redone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LocateResult:
    tetrahedron: np.ndarray   # (N,) int64, -1 where no cell contains the point
    barycentric: np.ndarray   # (N, 4)
    found: np.ndarray         # (N,) bool


class CanonicalSemantics:
    """The canonical tetrahedral volume plus its scalar fields."""

    def __init__(
        self,
        volume_npz: Path,
        semantic_npz: Path | None = None,
        resolution: int = 64,
        tolerance: float = 1.0e-9,
    ) -> None:
        volume = np.load(volume_npz)
        self.vertices = np.asarray(volume["volume_vertices"], dtype=np.float64)
        self.tetrahedra = np.asarray(volume["tetrahedra"], dtype=np.int64)
        self.harmonic_r = np.asarray(volume["harmonic_r"], dtype=np.float64)
        self.boundary_faces = np.asarray(volume["boundary_faces"], dtype=np.int64)
        self.boundary_labels = np.asarray(volume["boundary_labels"], dtype=np.int16)
        self.boundary_label_names = [str(x) for x in volume["boundary_label_names"]]
        self.inner_faces = np.asarray(
            volume["computational_inner_faces"], dtype=np.int64
        )
        self.inner_vertex_indices = np.asarray(
            volume["computational_inner_vertex_indices"], dtype=np.int64
        )
        # Which anatomical region each inner triangle belongs to. The inner
        # boundary is not all skin: it also carries the flat knee truncation,
        # which is a modelling cut rather than anatomy, so an address landing
        # there has to be distinguishable from one landing on the foot.
        self.inner_face_labels = np.asarray(
            volume["computational_inner_face_labels"], dtype=np.int16
        )
        self.tolerance = float(tolerance)

        corners = self.vertices[self.tetrahedra]            # (T, 4, 3)
        self.origin = corners[:, 0]                          # (T, 3)
        edges = corners[:, 1:] - corners[:, 0:1]             # (T, 3, 3)
        # Column-wise edge matrix; its inverse turns a displacement into the
        # last three barycentric weights.
        self.inverse = np.linalg.inv(edges.transpose(0, 2, 1))

        self._build_grid(resolution)
        self.semantic = None
        if semantic_npz is not None and Path(semantic_npz).is_file():
            self._load_semantic(semantic_npz)

    # -- spatial index ----------------------------------------------------
    def _build_grid(self, resolution: int) -> None:
        corners = self.vertices[self.tetrahedra]
        lower = corners.min(axis=1)
        upper = corners.max(axis=1)
        self.grid_lower = self.vertices.min(axis=0)
        self.grid_upper = self.vertices.max(axis=0)
        span = np.maximum(self.grid_upper - self.grid_lower, 1e-12)
        self.resolution = int(resolution)
        self.cell = span / self.resolution

        low = np.clip(
            ((lower - self.grid_lower) / self.cell).astype(np.int64), 0, resolution - 1
        )
        high = np.clip(
            ((upper - self.grid_lower) / self.cell).astype(np.int64), 0, resolution - 1
        )
        counts = np.prod(high - low + 1, axis=1)
        total = int(counts.sum())
        tet_ids = np.empty(total, dtype=np.int64)
        cell_ids = np.empty(total, dtype=np.int64)
        offset = 0
        # Chunked so the fully expanded (tet, cell) pair list never materializes
        # for the whole mesh at once.
        for start in range(0, len(self.tetrahedra), 4096):
            stop = min(start + 4096, len(self.tetrahedra))
            for index in range(start, stop):
                a, b = low[index], high[index]
                xs = np.arange(a[0], b[0] + 1)
                ys = np.arange(a[1], b[1] + 1)
                zs = np.arange(a[2], b[2] + 1)
                grid = (
                    xs[:, None, None] * self.resolution * self.resolution
                    + ys[None, :, None] * self.resolution
                    + zs[None, None, :]
                ).reshape(-1)
                count = grid.size
                tet_ids[offset : offset + count] = index
                cell_ids[offset : offset + count] = grid
                offset += count
        order = np.argsort(cell_ids, kind="stable")
        self.cell_tets = tet_ids[order]
        sorted_cells = cell_ids[order]
        self.cell_start = np.searchsorted(
            sorted_cells, np.arange(self.resolution**3 + 1)
        )

    def _cell_index(self, points: np.ndarray) -> np.ndarray:
        grid = np.floor((points - self.grid_lower) / self.cell).astype(np.int64)
        outside = np.any((grid < 0) | (grid >= self.resolution), axis=1)
        grid = np.clip(grid, 0, self.resolution - 1)
        flat = (
            grid[:, 0] * self.resolution * self.resolution
            + grid[:, 1] * self.resolution
            + grid[:, 2]
        )
        flat[outside] = -1
        return flat

    def locate_fast(
        self, points: np.ndarray, neighbours: int = 64, device=None,
        fallback: bool = True,
    ) -> LocateResult:
        """Same answer as ``locate``, computed on the GPU.

        The exact version walks the grid one query point at a time, which is
        fine for a few thousand samples and hopeless for the 442k needed to
        audit 27 shoes. This takes the K nearest tetrahedra by centroid, tests
        all K barycentrically at once, and falls back to the exact grid walk
        for anything still unresolved - so it is a speed change, not an
        accuracy change. The fallback count is reported.

        ``fallback=False`` skips that repair. Fiber tracing calls this once per
        integration step and expects points to leave the domain - that is how a
        trace terminates - so paying the exact walk for them would cost far
        more than it could ever recover.
        """

        import torch
        from scipy.spatial import cKDTree

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = np.asarray(points, dtype=np.float64)
        if not hasattr(self, "_centroid_tree"):
            centroids = self.vertices[self.tetrahedra].mean(axis=1)
            self._centroid_tree = cKDTree(centroids)
        _, candidates = self._centroid_tree.query(points, k=neighbours, workers=-1)
        candidates = np.asarray(candidates, dtype=np.int64)

        pts = torch.as_tensor(points, dtype=torch.float64, device=device)
        cand = torch.as_tensor(candidates, device=device)
        origin = torch.as_tensor(self.origin, dtype=torch.float64, device=device)
        inverse = torch.as_tensor(self.inverse, dtype=torch.float64, device=device)

        tetrahedron = torch.full((len(points),), -1, dtype=torch.long, device=device)
        barycentric = torch.zeros((len(points), 4), dtype=torch.float64, device=device)
        chunk = 4096
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            block = cand[start:stop]                          # (B, K)
            relative = pts[start:stop, None, :] - origin[block]
            weights = torch.einsum("bkij,bkj->bki", inverse[block], relative)
            first = 1.0 - weights.sum(dim=-1, keepdim=True)
            full = torch.cat((first, weights), dim=-1)         # (B, K, 4)
            inside = (full >= -self.tolerance).all(dim=-1)
            # Deterministic tie-break on a shared face: smallest tetrahedron id.
            ranked = torch.where(inside, block, torch.full_like(block, 2**31))
            best, slot = ranked.min(dim=1)
            hit = best < 2**31
            rows = torch.nonzero(hit, as_tuple=False).squeeze(1)
            if rows.numel():
                tetrahedron[start + rows] = best[rows]
                barycentric[start + rows] = full[rows, slot[rows]]

        tet = tetrahedron.cpu().numpy()
        bary = barycentric.cpu().numpy()
        missing = np.nonzero(tet < 0)[0]
        if missing.size and fallback:
            exact = self.locate(points[missing])
            tet[missing] = exact.tetrahedron
            bary[missing] = exact.barycentric
        self.last_fallback = int(missing.size)
        return LocateResult(tet, bary, tet >= 0)

    def inside_inner_boundary_fast(self, points: np.ndarray, device=None) -> np.ndarray:
        """GPU ray-parity. Same test as the NumPy version, batched."""

        import torch

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        triangles = self.vertices[self.inner_vertex_indices[self.inner_faces]]
        tri = torch.as_tensor(triangles, dtype=torch.float32, device=device)
        origin = tri[:, 0]
        edge_a = tri[:, 1] - tri[:, 0]
        edge_b = tri[:, 2] - tri[:, 0]
        direction = torch.tensor([0.7213, 0.4519, 0.5241], device=device)
        direction = direction / direction.norm()
        pvec = torch.cross(direction.expand_as(edge_b), edge_b, dim=-1)
        determinant = (edge_a * pvec).sum(-1)
        usable = determinant.abs() > 1e-12
        origin, edge_a, edge_b = origin[usable], edge_a[usable], edge_b[usable]
        pvec, determinant = pvec[usable], determinant[usable]
        inverse = 1.0 / determinant

        pts = torch.as_tensor(points, dtype=torch.float32, device=device)
        out = torch.zeros(len(points), dtype=torch.bool, device=device)
        chunk = 64
        for start in range(0, len(points), chunk):
            block = pts[start : start + chunk]
            tvec = block[:, None, :] - origin[None]
            u = (tvec * pvec[None]).sum(-1) * inverse
            qvec = torch.cross(tvec, edge_a[None].expand_as(tvec), dim=-1)
            v = (qvec @ direction) * inverse
            t = (qvec * edge_b[None]).sum(-1) * inverse
            hit = (u >= 0) & (v >= 0) & (u + v <= 1.0) & (t > 1e-9)
            out[start : start + chunk] = (hit.sum(dim=1) % 2) == 1
        return out.cpu().numpy()

    def locate(self, points: np.ndarray) -> LocateResult:
        """Exact barycentric containment, no nearest-cell snapping."""

        points = np.asarray(points, dtype=np.float64)
        cells = self._cell_index(points)
        tetrahedron = np.full(len(points), -1, dtype=np.int64)
        barycentric = np.zeros((len(points), 4), dtype=np.float64)
        for index in np.nonzero(cells >= 0)[0]:
            cell = cells[index]
            candidates = self.cell_tets[
                self.cell_start[cell] : self.cell_start[cell + 1]
            ]
            if candidates.size == 0:
                continue
            relative = points[index] - self.origin[candidates]
            weights = np.einsum(
                "tij,tj->ti", self.inverse[candidates], relative
            )
            first = 1.0 - weights.sum(axis=1)
            full = np.concatenate((first[:, None], weights), axis=1)
            inside = np.all(full >= -self.tolerance, axis=1)
            hit = np.nonzero(inside)[0]
            if hit.size:
                # Deterministic on a shared face: smallest tetrahedron id.
                pick = hit[np.argmin(candidates[hit])]
                tetrahedron[index] = candidates[pick]
                barycentric[index] = full[pick]
        return LocateResult(tetrahedron, barycentric, tetrahedron >= 0)

    # -- fields -----------------------------------------------------------
    def harmonic(self, result: LocateResult) -> np.ndarray:
        values = np.full(len(result.tetrahedron), np.nan)
        ok = result.found
        corners = self.tetrahedra[result.tetrahedron[ok]]
        values[ok] = np.einsum(
            "ni,ni->n", result.barycentric[ok], self.harmonic_r[corners]
        )
        return values

    def _load_semantic(self, path: Path) -> None:
        data = np.load(path)
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
        edges = np.asarray(data["unique_edges"], dtype=np.int64)
        vertex_count = len(self.vertices)
        self.vertex_coefficients = coefficients[:vertex_count]
        self.edge_coefficients = coefficients[vertex_count:]
        if len(self.edge_coefficients) != len(edges):
            raise ValueError("semantic coefficients do not match the edge list")
        lookup = {}
        for index, (a, b) in enumerate(edges):
            lookup[(int(a), int(b))] = index
            lookup[(int(b), int(a))] = index
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.tet_edge_coefficients = np.empty(
            (len(self.tetrahedra), 6), dtype=np.int64
        )
        for slot, (a, b) in enumerate(pairs):
            keys = self.tetrahedra[:, [a, b]]
            self.tet_edge_coefficients[:, slot] = [
                lookup[(int(u), int(v))] for u, v in keys
            ]
        self.semantic = True
        self._edge_pairs = pairs

    def inside_inner_boundary(
        self, points: np.ndarray, chunk: int = 256
    ) -> np.ndarray:
        """Ray-parity containment against the closed computational boundary.

        The canonical tetrahedra fill the region *between* the anatomy and the
        outer envelope, so a point that no tetrahedron contains is either
        inside the anatomy or beyond the envelope. Those two mean opposite
        things - material the foot already occupies, versus material outside
        the modelled domain - so they have to be told apart.

        Parity is valid here because this boundary is closed by construction;
        the canonical volume is built from it. Implemented directly rather than
        through ``trimesh.contains`` because that needs ``rtree``, which is not
        in this environment and is not worth installing into a shared one.
        """

        triangles = self.vertices[self.inner_vertex_indices[self.inner_faces]]
        origin = triangles[:, 0]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        # Fixed ray direction, deliberately not axis aligned so it is unlikely
        # to graze a face or pass exactly through a shared edge.
        direction = np.array([0.7213, 0.4519, 0.5241])
        direction = direction / np.linalg.norm(direction)
        pvec = np.cross(direction, edge_b)
        determinant = np.einsum("ij,ij->i", edge_a, pvec)
        usable = np.abs(determinant) > 1e-14
        origin, edge_a, edge_b = origin[usable], edge_a[usable], edge_b[usable]
        pvec, determinant = pvec[usable], determinant[usable]
        inverse = 1.0 / determinant

        inside = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), chunk):
            block = points[start : start + chunk]
            tvec = block[:, None, :] - origin[None, :, :]
            u = np.einsum("pij,ij->pi", tvec, pvec) * inverse
            qvec = np.cross(tvec, edge_a[None, :, :])
            v = qvec @ direction * inverse
            t = np.einsum("pij,ij->pi", qvec, edge_b) * inverse
            hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-12)
            inside[start : start + chunk] = (hit.sum(axis=1) % 2) == 1
        return inside

    def semantic_rho(self, result: LocateResult) -> np.ndarray:
        """The quadratic semantic scalar used by the older address stage.

        rho = sum_a c_a lam_a^2 + 2 sum_{a<b} c_ab lam_a lam_b
        """

        if not self.semantic:
            raise RuntimeError("semantic coefficients were not loaded")
        values = np.full(len(result.tetrahedron), np.nan)
        ok = result.found
        tets = result.tetrahedron[ok]
        lam = result.barycentric[ok]
        corner_c = self.vertex_coefficients[self.tetrahedra[tets]]
        total = np.einsum("ni,ni->n", corner_c, lam * lam)
        edge_c = self.edge_coefficients[self.tet_edge_coefficients[tets]]
        for slot, (a, b) in enumerate(self._edge_pairs):
            total = total + 2.0 * edge_c[:, slot] * lam[:, a] * lam[:, b]
        values[ok] = total
        return values
