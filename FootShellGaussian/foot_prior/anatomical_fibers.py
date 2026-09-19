"""Experimental 11-D semantic scalar, continuous directions, and fiber queries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time
import hashlib
import json
from pathlib import Path
from enum import IntEnum

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.optimize import brentq
from scipy.sparse.linalg import splu

from .anatomical_volume import (
    BOUNDARY_KNEE_TRUNCATION, BOUNDARY_OUTER_ENVELOPE,
    CanonicalAnatomicalVolume, _tetrahedron_gradients,
)
from .anatomy import array_digest
from .instance_volume_mapping import CanonicalVolumeLocator


EDGES = np.asarray(list(combinations(range(4), 2)), dtype=np.int64)
FACE_CORNERS = np.asarray(list(combinations(range(4), 3)), dtype=np.int64)
SAMPLES = np.vstack((np.eye(4), np.eye(4)[EDGES].mean(axis=1),
                     np.eye(4)[FACE_CORNERS].mean(axis=1), np.full((1, 4), .25)))
SOLVER_OPTIONS = dict(maxiter=2000, maxfun=4000, maxls=40, maxcor=10,
                      ftol=1e-15, gtol=1e-10)
BARY_TOL = 1e-10
ZERO_TOL = 1e-10
WEAK_TOL = 1e-8
RESIDUAL_TOL = 1e-8
POLISH_MAX_ITERATIONS = 2000


def semantic_configuration() -> dict:
    return dict(basis="quadratic_bernstein", inner_boundary="closed_inner_zero",
                outer_boundary="one", coefficient_bounds=[0., 1.],
                optimizer="diagonally_scaled_lbfgsb", solver_options=SOLVER_OPTIONS.copy(),
                convergence_polish=dict(method="diagonal_projected_gradient",
                    step="inverse_scaled_gershgorin_bound", maxiter=POLISH_MAX_ITERATIONS),
                barycentric_tolerance=BARY_TOL, zero_gradient_tolerance=ZERO_TOL,
                weak_gradient_tolerance=WEAK_TOL, projected_residual_tolerance=RESIDUAL_TOL)


@dataclass(frozen=True)
class CanonicalSemanticField:
    coefficients: np.ndarray
    unique_edges: np.ndarray
    cell_coefficients: np.ndarray
    barycentric_gradients: np.ndarray
    zero_indices: np.ndarray
    one_indices: np.ndarray
    diagonal: float
    source_geometry_digest: str
    solver: dict


@dataclass(frozen=True)
class _ScalarRefinement:
    """Auxiliary FE grid only; parent IDs refer to the untouched canonical grid."""

    volume_vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    boundary_labels: np.ndarray
    harmonic_r: np.ndarray
    parent_tetrahedron_indices: np.ndarray
    refined_parent_indices: np.ndarray


def _shared_face_owners(cells):
    faces = np.sort(cells[:, FACE_CORNERS].reshape(-1, 3), axis=1)
    _, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("nonmanifold scalar grid")
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])][counts == 2]
    return order[starts[:, None] + np.arange(2)]


def _refine_selected_cells(canonical, selected):
    selected = np.asarray(selected)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= len(canonical.tetrahedra))):
        raise ValueError("invalid refinement parent IDs")
    selected = np.unique(selected)
    original = canonical.tetrahedra
    centers = canonical.volume_vertices[original[selected]].mean(axis=1)
    center_ids = len(canonical.volume_vertices) + np.arange(len(selected))
    children = np.repeat(original[selected, None, :], 4, axis=1)
    for corner in range(4):
        children[:, corner, corner] = center_ids
    cells = np.concatenate((original.copy(), children[:, 1:].reshape(-1, 4)))
    cells[selected] = children[:, 0]
    # Replacing one vertex by its parent centroid preserves orientation and
    # gives exactly one quarter of the parent's volume. No face is subdivided.
    return _ScalarRefinement(
        np.vstack((canonical.volume_vertices, centers)), cells,
        canonical.boundary_faces, canonical.boundary_labels,
        np.r_[canonical.harmonic_r, canonical.harmonic_r[original[selected]].mean(axis=1)],
        np.r_[np.arange(len(original)), np.repeat(selected, 3)], selected)


def refine_scalar_neighborhoods(canonical, seed_tetrahedron_indices):
    """One face-adjacency ring, followed by conforming centroid 1-to-4 splits."""
    seeds = np.asarray(seed_tetrahedron_indices)
    if (seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer)
            or not len(seeds) or np.any(seeds < 0) or np.any(seeds >= len(canonical.tetrahedra))):
        raise ValueError("refinement requires valid nonempty canonical seed IDs")
    pairs = _shared_face_owners(canonical.tetrahedra) // 4
    neighbors = pairs[np.isin(pairs, seeds).any(axis=1)].ravel()
    return _refine_selected_cells(canonical, np.unique(np.r_[seeds, neighbors]))


def _layout(canonical):
    vertices, cells = canonical.volume_vertices, canonical.tetrahedra
    edges, inverse = np.unique(np.sort(cells[:, EDGES].reshape(-1, 2), axis=1),
                               axis=0, return_inverse=True)
    indices = np.column_stack((cells, len(vertices) + inverse.reshape(-1, 6)))
    edge_lookup = {tuple(e): i + len(vertices) for i, e in enumerate(edges)}
    boundaries = []
    for outer in (False, True):
        faces = canonical.boundary_faces[
            (canonical.boundary_labels == BOUNDARY_OUTER_ENVELOPE) == outer]
        boundary_edges = np.unique(np.sort(
            faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1), axis=0)
        boundaries.append(np.unique(np.concatenate((np.unique(faces),
            np.asarray([edge_lookup[tuple(e)] for e in boundary_edges], dtype=np.int64)))))
    if not all(len(b) for b in boundaries) or np.intersect1d(*boundaries).size:
        raise ValueError("semantic boundaries must be nonempty and disjoint")
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("invalid canonical extent")
    gradients = _tetrahedron_gradients(vertices, cells)
    return edges, indices, boundaries[0], boundaries[1], diagonal, gradients


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_semantic_field(canonical, directory):
    """Read back coefficients only after verifying their exact canonical source."""
    directory = Path(directory)
    payload = json.loads((directory / 'semantic_field.json').read_text())
    if (payload.get('schema_version') != 1
            or payload.get('stage') != 'canonical_semantic_scalar_field'
            or payload.get('configuration') != semantic_configuration()):
        raise ValueError('semantic field source/configuration mismatch')
    geometry_digest = array_digest(canonical.volume_vertices, canonical.tetrahedra)
    if payload.get('source_geometry_digest') != geometry_digest:
        refinement = payload.get('refinement', {})
        if refinement.get('canonical_source_geometry_digest') != geometry_digest:
            raise ValueError('semantic field source/configuration mismatch')
        canonical = _refine_selected_cells(canonical, refinement['refined_parent_indices'])
        if payload['source_geometry_digest'] != array_digest(canonical.volume_vertices, canonical.tetrahedra):
            raise ValueError('semantic refinement geometry digest mismatch')
    for source in payload['inputs']:
        if _file_sha256(source['path']) != source['sha256']:
            raise ValueError('semantic field input file digest mismatch')
    edges, ids, zero, one, diagonal, gradients = _layout(canonical)
    with np.load(directory / 'semantic_field.npz', allow_pickle=False) as archive:
        coefficients = archive['coefficients']
        if (coefficients.shape != (len(canonical.volume_vertices) + len(edges),)
                or not np.isfinite(coefficients).all()
                or not np.array_equal(archive['unique_edges'], edges)
                or array_digest(coefficients, edges) != payload['coefficient_digest']):
            raise ValueError('semantic field coefficient arrays/digest mismatch')
    return CanonicalSemanticField(coefficients, edges, ids, gradients, zero, one,
        diagonal, payload['source_geometry_digest'], payload['solver'])


def _basis(weights):
    return np.concatenate((weights ** 2,
        2 * weights[..., EDGES[:, 0]] * weights[..., EDGES[:, 1]]), axis=-1)


def _basis_gradients(weights, gradients):
    return np.concatenate((2 * weights[..., :, None] * gradients,
        2 * (weights[..., EDGES[:, 0], None] * gradients[..., EDGES[:, 1], :]
           + weights[..., EDGES[:, 1], None] * gradients[..., EDGES[:, 0], :])), axis=-2)


def evaluate_semantic_field(field, tetrahedron_indices, barycentric_weights):
    """Evaluate scalar and one-sided gradient in canonical physical coordinates."""
    ids = np.asarray(tetrahedron_indices)
    w = np.asarray(barycentric_weights, dtype=np.float64)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or w.shape != (len(ids), 4) or not np.isfinite(w).all()
            or np.any(ids < 0) or np.any(ids >= len(field.cell_coefficients))
            or np.any(w < -BARY_TOL) or np.any(w > 1 + BARY_TOL)
            or np.any(abs(w.sum(axis=1) - 1) > BARY_TOL)):
        raise ValueError("invalid tetrahedral coordinates")
    w = np.clip(w, 0, 1)
    w /= w.sum(axis=1, keepdims=True)
    c = field.coefficients[field.cell_coefficients[ids]]
    return (np.einsum('ni,ni->n', c, _basis(w)),
            np.einsum('ni,nij->nj', c, _basis_gradients(w, field.barycentric_gradients[ids])))


def _stiffness(canonical, indices, gradients, diagonal):
    cells = canonical.tetrahedra
    v = canonical.volume_vertices[cells] / diagonal
    volume = np.linalg.det(np.transpose(v[:, 1:] - v[:, :1], (0, 2, 1))) / 6
    if np.any(volume <= 0) or not np.isfinite(volume).all():
        raise ValueError("canonical cells must be positively oriented; no reorientation allowed")
    a, b = (5 + 3 * np.sqrt(5)) / 20, (5 - np.sqrt(5)) / 20
    quadrature = np.full((4, 4), b)
    np.fill_diagonal(quadrature, a)
    local = np.zeros((len(cells), 10, 10))
    for w in quadrature:
        grad = _basis_gradients(w, gradients * diagonal)
        local += .25 * volume[:, None, None] * np.einsum('nij,nkj->nik', grad, grad)
    return sparse.coo_matrix((local.ravel(),
        (np.repeat(indices, 10, axis=1).ravel(), np.tile(indices, (1, 10)).ravel())),
        shape=(int(indices.max()) + 1,) * 2).tocsr()


def solve_canonical_semantic_field(canonical_volume: CanonicalAnatomicalVolume):
    """Solve only the bounded Dirichlet energy, leaving all input arrays intact."""
    start = time.monotonic()
    c = canonical_volume
    edges, ids, zero, one, diagonal, gradients = _layout(c)
    stiffness = _stiffness(c, ids, gradients, diagonal)
    values = np.clip(np.concatenate((c.harmonic_r, c.harmonic_r[edges].mean(axis=1))), 0, 1)
    values[zero], values[one] = 0., 1.
    free = np.setdiff1d(np.arange(len(values)), np.concatenate((zero, one)))
    matrix = stiffness[free][:, free]
    fixed = values.copy()
    fixed[free] = 0
    linear = (stiffness @ fixed)[free]
    d = matrix.diagonal()
    if np.any(d <= 0):
        raise ValueError("nonpositive free stiffness diagonal")
    scale = np.sqrt(d)
    energy_offset = float(.5 * fixed @ (stiffness @ fixed))
    print(f"assembly: {time.monotonic()-start:.2f}s; coefficients={len(values)}, free={len(free)}", flush=True)

    def objective(z):
        x = z / scale
        mx = matrix @ x
        # Include fixed-boundary energy: dropping this constant changes the
        # relative objective stopping criterion even though gradients agree.
        return float(.5 * x @ mx + x @ linear + energy_offset), (mx + linear) / scale

    iterations = 0
    def callback(z):
        nonlocal iterations
        iterations += 1
        if iterations % 50 == 0:
            print(f"scalar iteration {iterations}: objective={objective(z)[0]:.12g}", flush=True)

    if len(free):
        result = minimize(objective, values[free] * scale, jac=True, method="L-BFGS-B",
                          bounds=np.column_stack((np.zeros(len(free)), scale)),
                          options=SOLVER_OPTIONS.copy(), callback=callback)
        values[free] = result.x / scale
        message, success, nfev = str(result.message), bool(result.success), int(result.nfev)
    else:
        message, success, nfev = "all coefficients prescribed", True, 0
    # Energy stagnation can precede the independent KKT tolerance. A bounded
    # projected-gradient polish uses the SAME convex objective and a globally
    # conservative step, rather than relaxing acceptance or restarting L-BFGS.
    x = values[free].copy()
    step = 1 / float(np.max((abs(matrix) @ (1 / scale)) / scale, initial=1.))
    residual_before_polish = None
    for polish_iterations in range(POLISH_MAX_ITERATIONS + 1):
        g = matrix @ x + linear
        residual = float(np.max(abs(x - np.clip(x - g / d, 0, 1)), initial=0))
        if residual_before_polish is None:
            residual_before_polish = residual
        if not np.isfinite(residual) or residual <= RESIDUAL_TOL or polish_iterations == POLISH_MAX_ITERATIONS:
            break
        if polish_iterations and polish_iterations % 100 == 0:
            print(f"convergence polish {polish_iterations}: residual={residual:.6g}", flush=True)
        x = np.clip(x - step * g / d, 0, 1)
    values[free] = x
    finite = bool(np.isfinite(values).all() and np.isfinite(residual))
    solver = dict(optimizer_success=success, message=message, iterations=iterations,
                  evaluations=nfev, projected_residual=residual if finite else None,
                  residual_before_polish=residual_before_polish if finite else None,
                  polish_iterations=polish_iterations,
                  converged=bool(finite and residual <= RESIDUAL_TOL),
                  energy=float(.5 * values @ (stiffness @ values)) if finite else None)
    print(f"solve complete: {time.monotonic()-start:.2f}s; {solver}", flush=True)
    return CanonicalSemanticField(values, edges, ids, gradients, zero, one, diagonal,
        array_digest(c.volume_vertices, c.tetrahedra), solver)


def _boundary_simplices(canonical):
    faces = canonical.boundary_faces
    return (set(int(x) for x in np.unique(faces)),
            set(map(tuple, np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1))),
            set(map(tuple, np.sort(faces, axis=1))))


def _location(cell, weights, boundary):
    support = tuple(sorted(int(x) for x in cell[weights > BARY_TOL]))
    n = len(support)
    if n == 1 and support[0] in boundary[0] or n in (2, 3) and support in boundary[n - 1]:
        return "domain_boundary"
    return "cell_interior" if n == 4 else "internal_interface"


def _stationary_points(cells, vertex_gradients, boundary):
    """Solve affine gradient zeros; enumerate supports only for singular systems."""
    records = []
    systems = np.concatenate((vertex_gradients.transpose(0, 2, 1),
                              np.ones((len(cells), 1, 4))), axis=1)
    rhs = np.asarray([0., 0., 0., 1.])
    singular = np.linalg.svd(systems, compute_uv=False)
    full = singular[:, -1] > 1e-12 * np.maximum(singular[:, 0], 1.)
    ids = np.flatnonzero(full)
    roots = np.linalg.solve(systems[ids], np.broadcast_to(rhs, (len(ids), 4))[..., None])[..., 0]
    for k, w in zip(ids, roots):
        if np.all(w >= -BARY_TOL) and np.all(w <= 1 + BARY_TOL):
            location = _location(cells[k], w, boundary)
            if np.max(abs(systems[k] @ w - rhs)) > ZERO_TOL:
                location = "unresolved"
            records.append((int(k), location, w))
    for count, k in enumerate(np.flatnonzero(~full)):
        if count and count % 500 == 0:
            print(f"stationary audit: {count} rank-deficient cells checked", flush=True)
        for size in range(4, 0, -1):
            for support in combinations(range(4), size):
                a = systems[k][:, support]
                w, _, rank, _ = np.linalg.lstsq(a, rhs, rcond=1e-12)
                if np.max(abs(a @ w - rhs)) > ZERO_TOL:
                    continue
                if rank < size:
                    # Maximize the smallest supported weight, testing the relative
                    # interior of this simplex rather than an arbitrary root.
                    res = linprog(np.r_[np.zeros(size), -1.],
                        A_ub=np.column_stack((-np.eye(size), np.ones(size))),
                        b_ub=np.zeros(size), A_eq=np.column_stack((a, np.zeros(4))),
                        b_eq=rhs, bounds=[(0, 1)] * size + [(0, 1 / size)], method="highs")
                    if res.status == 2:
                        continue
                    if not res.success:
                        records.append((int(k), "unresolved", np.full(4, np.nan)))
                        continue
                    w = res.x[:size]
                if np.min(w) <= BARY_TOL:
                    continue
                full_w = np.zeros(4)
                full_w[list(support)] = w
                location = _location(cells[k], full_w, boundary)
                if np.max(abs(a @ w - rhs)) > ZERO_TOL:
                    location = "unresolved"
                records.append((int(k), location, full_w))
    return records


def audit_semantic_field(canonical_volume, field):
    """Return a JSON summary plus NPZ/VTK diagnostic arrays; never certify fibers."""
    c = canonical_volume
    if field.source_geometry_digest != array_digest(c.volume_vertices, c.tetrahedra):
        raise ValueError("semantic field canonical geometry digest mismatch")
    if not np.isfinite(field.coefficients).all():
        return dict(status="solver_failed", reason="nonfinite coefficients"), {}
    n = len(c.tetrahedra)
    values, norms = [], []
    boundary = _boundary_simplices(c)
    weak_inside = np.zeros(n, dtype=bool)
    for w in SAMPLES:
        val, grad = evaluate_semantic_field(field, np.arange(n), np.tile(w, (n, 1)))
        norm = np.linalg.norm(grad, axis=1) * field.diagonal
        values.append(val)
        norms.append(norm)
        for k in np.flatnonzero(norm < WEAK_TOL):
            if _location(c.tetrahedra[k], w, boundary) != "domain_boundary":
                weak_inside[k] = True
    values, norms = np.asarray(values).T, np.asarray(norms).T
    vertex_grad = np.stack([evaluate_semantic_field(field, np.arange(n), np.tile(w, (n, 1)))[1]
                           for w in np.eye(4)], axis=1) * field.diagonal
    near_flat = np.max(np.linalg.norm(vertex_grad, axis=2), axis=1) <= ZERO_TOL
    old_flat = np.all(c.harmonic_r[c.tetrahedra] == 0, axis=1)
    records = _stationary_points(c.tetrahedra, vertex_grad, boundary)
    critical = np.zeros(n, dtype=np.int8)
    for k, location, _ in records:
        critical[k] = max(critical[k], 1 if location == "domain_boundary" else 2)

    # Pair common faces and evaluate both traces at identical barycentric locations.
    owners = _shared_face_owners(c.tetrahedra)
    side_values, side_gradients = [], []
    for side in range(2):
        face_id = owners[:, side]
        w = np.eye(4)[FACE_CORNERS[face_id % 4]].mean(axis=1)
        val, grad = evaluate_semantic_field(field, face_id // 4, w)
        side_values.append(val)
        side_gradients.append(grad)
    continuity = float(np.max(abs(side_values[0] - side_values[1]), initial=0))
    jumps = np.linalg.norm(side_gradients[0] - side_gradients[1], axis=1) * field.diagonal
    cap_vertices = np.unique(c.boundary_faces[c.boundary_labels == BOUNDARY_KNEE_TRUNCATION])
    cap_adjacent = np.isin(c.tetrahedra, cap_vertices).any(axis=1)
    old_values = np.einsum('ni,si->ns', c.harmonic_r[c.tetrahedra], SAMPLES)
    difference = abs(values - old_values)
    boundary_ok = bool(np.all(field.coefficients[field.zero_indices] == 0)
                       and np.all(field.coefficients[field.one_indices] == 1))
    bounds_ok = bool(np.all((field.coefficients >= 0) & (field.coefficients <= 1)))
    checks = dict(boundary_exact=boundary_ok, coefficients_bounded=bounds_ok,
                  samples_finite=bool(np.isfinite(values).all() and np.isfinite(norms).all()),
                  shared_face_max_error=continuity,
                  near_flat_cells=int(near_flat.sum()), weak_interior_cells=int(weak_inside.sum()),
                  old_zero_cells=int(old_flat.sum()),
                  old_zero_cells_positive_centroid=int(np.sum(old_flat & (values[:, -1] > 0))),
                  stationary_counts={loc: sum(r[1] == loc for r in records) for loc in
                      ("cell_interior", "internal_interface", "domain_boundary", "unresolved")},
                  maximum_dimensionless_gradient_jump=float(np.max(jumps, initial=0)),
                  max_change_cap_adjacent=float(np.max(difference[cap_adjacent], initial=0)),
                  max_change_elsewhere=float(np.max(difference[~cap_adjacent], initial=0)))
    clean = (boundary_ok and bounds_ok and checks['samples_finite'] and continuity <= BARY_TOL
             and not near_flat.any() and not weak_inside.any() and not np.any(critical == 2))
    status = ("solver_failed" if not field.solver['converged'] else
              "scalar_candidate" if clean else "needs_field_revision")
    reasons = []
    if not field.solver['converged']:
        reasons.append('projected solver residual exceeds tolerance')
    if not boundary_ok or not bounds_ok or not checks['samples_finite'] or continuity > BARY_TOL:
        reasons.append('scalar boundary, bounds, finiteness, or continuity check failed')
    if near_flat.any() or weak_inside.any():
        reasons.append('flat or sampled weak-gradient regions inside shell')
    if np.any(critical == 2):
        reasons.append('stationary or unresolved regions inside shell')
    summary = dict(status=status, checks=checks, fibers_validated=False,
                   reason='; '.join(reasons) if reasons else
                   'scalar audit passed; direction field and fibers are not implemented')
    arrays = dict(old_zero_cells=old_flat, near_flat_cells=near_flat,
                  weak_interior_cells=weak_inside, critical_cell_class=critical,
                  sampled_min_gradient=norms.min(axis=1), semantic_centroid_r=values[:, -1],
                  stationary_cell_ids=np.asarray([r[0] for r in records], dtype=np.int64),
                  stationary_locations=np.asarray([r[1] for r in records], dtype='U32'),
                  stationary_barycentric=np.asarray([r[2] for r in records]).reshape(-1, 4))
    return summary, arrays


# Continuous directions and conservative experimental fiber queries.
class FiberStatus(IntEnum):
    VALID_ANATOMICAL = 0
    VALID_ARTIFICIAL_CAP = 1
    INVALID_VOLUME = 10
    ORIGINAL_SADDLE = 11
    DIRECTION_INCOMPATIBLE = 12
    DIRECTION_UNRESOLVED = 13
    WEAK_DIRECTION = 14
    BOUNDARY_AMBIGUITY = 15
    INTEGRATION_FAILED = 16
    ROUND_TRIP_FAILED = 17
    NONMONOTONE = 18
    HIT_INNER = 100
    HIT_OUTER = 101


def fiber_configuration():
    return dict(direction_smoothing_edge_fraction=.5, direction_residual_tolerance=1e-10,
        certification_depth=8, relative_tolerance=1e-7, absolute_tolerance=1e-10,
        minimum_step=1e-12, maximum_cell_step_fraction=.25, maximum_path_length=10.,
        maximum_accepted_steps=20000, maximum_rejected_steps=40000,
        barycentric_tolerance=1e-10, direction_tolerance=1e-10,
        rho_tolerance=1e-9, round_trip_relative=1e-6, method='batched_bs23',
        scalar='converged_unrefined_quadratic', acceptance='coverage_review_required')


@dataclass(frozen=True)
class FiberCoordinates:
    face_indices: np.ndarray
    barycentric_weights: np.ndarray
    semantic_r: np.ndarray


@dataclass(frozen=True)
class FiberQueryResult:
    coordinates: FiberCoordinates
    canonical_points: np.ndarray
    inner_origins: np.ndarray
    outer_endpoints: np.ndarray
    status_codes: np.ndarray
    volume_status_codes: np.ndarray
    backward_reasons: np.ndarray
    forward_reasons: np.ndarray
    round_trip_errors: np.ndarray
    integration_steps: np.ndarray
    label_weights: dict
    correspondence: dict

    @property
    def mappable_mask(self):
        return np.isin(self.status_codes, [FiberStatus.VALID_ANATOMICAL, FiberStatus.VALID_ARTIFICIAL_CAP])

    @property
    def footwear_support_mask(self):
        return self.status_codes == FiberStatus.VALID_ANATOMICAL


@dataclass(frozen=True)
class CanonicalFiberField:
    canonical: CanonicalAnatomicalVolume
    scalar: CanonicalSemanticField
    locator: CanonicalVolumeLocator
    directions: np.ndarray
    exclusions: np.ndarray
    vertices: np.ndarray
    inverse: np.ndarray
    gradients: np.ndarray
    neighbors: np.ndarray
    boundary_inner_ids: np.ndarray
    vertex_cells: tuple
    cell_diameters: np.ndarray
    origin: np.ndarray
    diagonal: float
    diagnostics: dict
    region_labels: dict


def _direction_system(canonical, scalar):
    cells = canonical.tetrahedra
    L = scalar.diagonal
    vertices = (canonical.volume_vertices - canonical.volume_vertices.min(0)) / L
    gradients = scalar.barycentric_gradients * L
    volume = canonical.tetrahedron_signed_volumes / L**3
    if np.any(volume <= 0):
        raise ValueError('direction solve requires positive unchanged tetrahedra')
    ids = np.arange(len(cells))
    g = np.stack([evaluate_semantic_field(scalar, ids, np.tile(w, (len(ids), 1)))[1]*L
                  for w in np.eye(4)], axis=1)
    local_coeff = scalar.coefficients[scalar.cell_coefficients]
    # Exact constant corner-and-incident-edge coefficients imply zero gradient.
    for a in range(4):
        incident = 4 + np.flatnonzero(np.any(EDGES == a, axis=1))
        structural = np.all(local_coeff[:, incident] == local_coeff[:, a, None], axis=1)
        g[structural, a] = 0.
    rows = np.repeat(cells, 4, axis=1).ravel()
    cols = np.tile(cells, (1, 4)).ravel()
    shape = (len(vertices), len(vertices))
    mass = sparse.coo_matrix(((volume[:, None, None]*(np.ones((4, 4))+np.eye(4))/20).ravel(),
                              (rows, cols)), shape=shape).tocsc()
    stiffness = sparse.coo_matrix(((volume[:, None, None]*np.einsum('nik,njk->nij', gradients, gradients)).ravel(),
                                   (rows, cols)), shape=shape).tocsc()
    edges = np.unique(np.sort(canonical.boundary_faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1), axis=0)
    h = float(np.median(np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)))
    rhs = np.zeros((len(vertices), 3))
    np.add.at(rhs, cells.ravel(), (volume[:, None, None]/20*(g+g.sum(1)[:, None])).reshape(-1, 3))
    return mass + (.5*h)**2*stiffness, rhs, g, h


def _certify_progress(matrix, physical_corners, cell, boundary, depth=0, weights=None):
    """Bernstein bounds on every relative simplex, allowing only boundary zeros."""
    if weights is None:
        weights = np.eye(4)
    values = weights @ matrix @ weights.T
    guard = 64*np.finfo(float).eps*max(float(np.max(abs(matrix))), 1e-30)
    # A strictly negative value at a corner/centroid proves incompatibility.
    samples = np.r_[np.diag(values), np.array([values.mean()])]
    if np.min(samples) < -guard:
        return int(FiberStatus.DIRECTION_INCOMPATIBLE)
    if np.all(values >= 0):
        good = True
        for size in range(1, 5):
            for support in combinations(range(4), size):
                w = weights[list(support)].mean(0)
                if _location(cell, w, boundary) == 'domain_boundary':
                    continue
                if np.max(values[np.ix_(support, support)]) <= guard:
                    good = False
                    break
            if not good:
                break
        if good:
            return 0
    if depth == 8:
        return int(FiberStatus.DIRECTION_UNRESOLVED)
    xyz = weights @ physical_corners
    lengths = np.sum((xyz[EDGES[:, 0]]-xyz[EDGES[:, 1]])**2, axis=1)
    a, b = EDGES[int(np.argmax(lengths))]
    midpoint = (weights[a]+weights[b])/2
    outcome = 0
    for corner in (a, b):
        child = weights.copy(); child[corner] = midpoint
        result = _certify_progress(matrix, physical_corners, cell, boundary, depth+1, child)
        if result == FiberStatus.DIRECTION_INCOMPATIBLE:
            return result
        outcome = max(outcome, result)
    return outcome


def _progress_exclusions(canonical, g, directions, seeds):
    cells = canonical.tetrahedra
    products = np.einsum('nai,nbi->nab', g, directions[cells])
    matrices = (products+products.transpose(0, 2, 1))/2
    excluded = np.zeros(len(cells), dtype=np.int16)
    excluded[seeds] = FiberStatus.ORIGINAL_SADDLE
    boundary = _boundary_simplices(canonical)
    start = last = time.monotonic()
    # Strictly positive coefficients certify cells and all their interfaces at once.
    guard = 64*np.finfo(float).eps*np.maximum(abs(matrices).max(axis=(1, 2)), 1e-30)
    uncertain = np.flatnonzero((matrices.min(axis=(1, 2)) <= guard) & (excluded == 0))
    for k in uncertain:
        excluded[k] = _certify_progress(matrices[k], canonical.volume_vertices[cells[k]], cells[k], boundary)
        if time.monotonic()-last > 30:
            print(f'direction certification: cell={k}; elapsed={time.monotonic()-start:.1f}s', flush=True)
            last = time.monotonic()
    return excluded


def _prepare_fiber_field(canonical, scalar, directions, excluded, diagnostics, region_labels):
    locator = CanonicalVolumeLocator.build(canonical)
    cells = canonical.tetrahedra
    neighbors = np.full((len(cells), 4), -1, dtype=np.int64)
    inner_ids = np.full_like(neighbors, -1)
    face_corners = np.asarray([[b for b in range(4) if b != a] for a in range(4)])
    faces = np.sort(cells[:, face_corners].reshape(-1, 3), axis=1)
    unique, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('nonmanifold volume faces')
    order = np.argsort(inverse, kind='stable'); starts = np.r_[0, np.cumsum(counts[:-1])]
    pairs = order[starts[counts == 2, None]+np.arange(2)]
    neighbors.ravel()[pairs[:, 0]] = pairs[:, 1]//4
    neighbors.ravel()[pairs[:, 1]] = pairs[:, 0]//4
    inner = {tuple(sorted(canonical.computational_inner_vertex_indices[f])):i
             for i, f in enumerate(canonical.computational_inner_faces)}
    for u in np.flatnonzero(counts == 1):
        slot = order[starts[u]]
        inner_ids.ravel()[slot] = inner.get(tuple(unique[u]), -1)
    vertex_cells = [[] for _ in canonical.volume_vertices]
    for k, t in enumerate(cells):
        for a in t:
            vertex_cells[a].append(k)
    origin = canonical.volume_vertices.min(0); L = scalar.diagonal
    vertices = (canonical.volume_vertices-origin)/L
    xyz = vertices[cells]
    diameter = np.linalg.norm(xyz[:, EDGES[:, 0]]-xyz[:, EDGES[:, 1]], axis=2).max(1)
    return CanonicalFiberField(canonical, scalar, locator, directions, excluded, vertices,
        locator.instance_inverse_matrices*L, scalar.barycentric_gradients*L,
        neighbors, inner_ids, tuple(np.asarray(x, dtype=np.int64) for x in vertex_cells),
        diameter, origin, L, diagnostics, region_labels or {})


def build_anatomical_fiber_field(canonical, scalar, stationary_cell_ids, region_labels=None):
    if scalar.source_geometry_digest != array_digest(canonical.volume_vertices, canonical.tetrahedra):
        raise ValueError('fiber scalar must use the unrefined canonical geometry')
    if not scalar.solver.get('converged'):
        raise ValueError('fiber scalar has not converged')
    if (np.any(scalar.coefficients < 0) or np.any(scalar.coefficients > 1)
            or np.any(scalar.coefficients[scalar.zero_indices] != 0)
            or np.any(scalar.coefficients[scalar.one_indices] != 1)):
        raise ValueError('invalid scalar coefficients/boundaries')
    seeds = np.unique(np.asarray(stationary_cell_ids, dtype=np.int64))
    if np.any(seeds < 0) or np.any(seeds >= len(canonical.tetrahedra)):
        raise ValueError('invalid stationary cell IDs')
    start = time.monotonic()
    matrix, rhs, gradients, h = _direction_system(canonical, scalar)
    directions = splu(matrix).solve(rhs)
    residual = float(np.linalg.norm(matrix@directions-rhs)/max(np.linalg.norm(rhs), 1e-30))
    if not np.isfinite(directions).all() or residual > 1e-10:
        raise ValueError('direction solve failed residual validation')
    excluded = _progress_exclusions(canonical, gradients, directions, seeds)
    diagnostics = dict(relative_residual=residual, normalized_boundary_edge_median=h,
        original_saddle_cells=seeds.tolist(), exclusion_counts={FiberStatus(int(k)).name:int(n)
            for k, n in zip(*np.unique(excluded[excluded != 0], return_counts=True))},
        accepted_cell_count=int(np.sum(excluded == 0)), status='coverage_review_required')
    print(f'direction field: {time.monotonic()-start:.2f}s; {diagnostics}', flush=True)
    return _prepare_fiber_field(canonical, scalar, directions, excluded, diagnostics, region_labels)


def load_anatomical_fiber_field(canonical, scalar, directory, region_labels=None):
    directory = Path(directory)
    record = json.loads((directory/'fiber_field.json').read_text())
    if (record.get('schema_version') != 1 or record.get('configuration') != fiber_configuration()
            or record.get('source_geometry_digest') != scalar.source_geometry_digest
            or record.get('scalar_digest') != array_digest(scalar.coefficients, scalar.unique_edges)):
        raise ValueError('fiber field source/configuration mismatch')
    for source in record['inputs']:
        if _file_sha256(source['path']) != source['sha256']:
            raise ValueError('fiber input digest mismatch')
    with np.load(directory/'fiber_field.npz', allow_pickle=False) as z:
        directions, excluded = z['directions'], z['exclusions']
    if (directions.shape != canonical.volume_vertices.shape or not np.isfinite(directions).all()
            or excluded.shape != (len(canonical.tetrahedra),)
            or array_digest(directions, excluded) != record['field_digest']):
        raise ValueError('fiber array digest/shape mismatch')
    matrix, rhs, g, _ = _direction_system(canonical, scalar)
    if np.linalg.norm(matrix@directions-rhs)/max(np.linalg.norm(rhs), 1e-30) > 1e-10:
        raise ValueError('saved direction residual invalid')
    certified = _progress_exclusions(canonical, g, directions, record['diagnostics']['original_saddle_cells'])
    if not np.array_equal(certified, excluded):
        raise ValueError('saved exclusion masks failed revalidation')
    return _prepare_fiber_field(canonical, scalar, directions, excluded, record['diagnostics'], region_labels)


def _fiber_weights(field, ids, points):
    d = points-field.vertices[field.canonical.tetrahedra[ids, 0]]
    last = np.einsum('nij,nj->ni', field.inverse[ids], d)
    return np.c_[1-last.sum(1), last]


def _fiber_values(field, ids, points):
    w = _fiber_weights(field, ids, points)
    c = field.scalar.coefficients[field.scalar.cell_coefficients[ids]]
    return np.einsum('ni,ni->n', c, _basis(w))


def _unit_direction(field, ids, points, sign):
    w = _fiber_weights(field, ids, points)
    v = np.einsum('ni,nij->nj', w, field.directions[field.canonical.tetrahedra[ids]])
    norm = np.linalg.norm(v, axis=1)
    return sign*v/np.maximum(norm[:, None], 1e-300), norm


def _rk23_step(field, ids, points, steps, sign):
    h = steps[:, None]
    k1, n1 = _unit_direction(field, ids, points, sign)
    k2, n2 = _unit_direction(field, ids, points+h*k1/2, sign)
    k3, n3 = _unit_direction(field, ids, points+3*h*k2/4, sign)
    end = points+h*(2*k1/9+k2/3+4*k3/9)
    k4, n4 = _unit_direction(field, ids, end, sign)
    low = points+h*(7*k1/24+k2/4+k3/3+k4/8)
    # y(t)=c0+c1*t+c2*t²+c3*t³, 0<=t<=1.
    dense = np.stack((points, h*k1, 3*(end-points)-h*(2*k1+k4),
                      2*(points-end)+h*(k1+k4)), axis=1)
    return end, end-low, dense, np.minimum.reduce([n1, n2, n3, n4])


def _polynomial_point(dense, t):
    return dense[0]+t*(dense[1]+t*(dense[2]+t*dense[3]))


def _first_face_event(field, cell, dense):
    coeff = np.empty((4, 4))
    coeff[0] = _fiber_weights(field, np.array([cell]), dense[:1])[0]
    coeff[1:, 1:] = dense[1:] @ field.inverse[cell].T
    coeff[1:, 0] = -coeff[1:, 1:].sum(1)
    # Convex-hull property of cubic Bernstein coefficients avoids most root solves.
    bern = np.stack((coeff[0], coeff[0]+coeff[1]/3,
                     coeff[0]+2*coeff[1]/3+coeff[2]/3, coeff.sum(0)))
    time_hit, face = 1., -1
    for a in np.flatnonzero(bern.min(0) < -1e-12):
        poly = coeff[:, a]
        for root in np.roots(np.trim_zeros(poly[::-1], 'f')):
            if abs(root.imag) > 1e-9:
                continue
            t = float(root.real)
            slope = poly[1]+2*t*poly[2]+3*t*t*poly[3]
            if 1e-12 < t <= time_hit+1e-12 and slope < -1e-12:
                time_hit, face = min(t, 1.), int(a)
    return time_hit, face


def _resolve_start(field, point, cell, sign):
    """Choose the incident cell entered by the direction, without perturbing q."""
    w = _fiber_weights(field, np.array([cell]), point[None])[0]
    if w.min() > 1e-10:
        return cell, 0, -1
    support = field.canonical.tetrahedra[cell][w > 1e-10]
    if not len(support):
        return cell, int(FiberStatus.BOUNDARY_AMBIGUITY), -1
    incident = field.vertex_cells[support[0]]
    for vertex in support[1:]:
        incident = np.intersect1d(incident, field.vertex_cells[vertex])
    points = np.tile(point, (len(incident), 1))
    weights = _fiber_weights(field, incident, points)
    direction, norm = _unit_direction(field, incident, points, sign)
    derivative = np.einsum('nij,nj->ni', field.gradients[incident], direction)
    near = weights <= 1e-10
    inward = (np.all((derivative > 1e-10) | ~near, axis=1)
              & (weights.min(1) >= -1e-9) & (norm > 1e-10))
    choices = incident[inward]
    if len(choices) == 1:
        return int(choices[0]), 0, -1
    if len(choices) > 1:
        return cell, int(FiberStatus.BOUNDARY_AMBIGUITY), -1
    hits = []
    for j, k in enumerate(incident):
        for a in np.flatnonzero(near[j] & (derivative[j] < -1e-10) & (field.neighbors[k] < 0)):
            face = int(field.boundary_inner_ids[k, a])
            hits.append((int(FiberStatus.HIT_INNER if face >= 0 else FiberStatus.HIT_OUTER), face))
    hits = sorted(set(hits))
    if len(hits) == 1:
        return cell, *hits[0]
    return cell, int(FiberStatus.BOUNDARY_AMBIGUITY), -1


def _trace_fibers(field, points, cell_ids, sign, target_r=None, tolerance_scale=1., paths=None):
    """Batched independent adaptive traces; all points and steps are normalized."""
    n = len(points); ids = cell_ids.copy(); q = points.copy()
    reasons = np.zeros(n, dtype=np.int16); faces = np.full(n, -1, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int64); rejected = steps.copy(); length = np.zeros(n)
    h = .05*field.cell_diameters[np.maximum(ids, 0)]
    targets = np.full((n, 3), np.nan)
    active = ids >= 0; reasons[~active] = FiberStatus.INVALID_VOLUME
    if paths is not None:
        paths.extend([[point.copy()] for point in q])
    resolve = active.copy(); last_print = time.monotonic()
    while np.any(active):
        for j in np.flatnonzero(active & resolve):
            ids[j], reason, face = _resolve_start(field, q[j], ids[j], sign)
            resolve[j] = False
            if reason:
                reasons[j], faces[j], active[j] = reason, face, False
        take = np.flatnonzero(active)
        if not len(take):
            break
        mask = field.exclusions[ids[take]]
        bad = take[mask != 0]
        reasons[bad] = mask[mask != 0]; active[bad] = False
        take = np.flatnonzero(active)
        if not len(take):
            continue
        limited = ((steps[take] >= 20000) | (rejected[take] >= 40000)
                   | (length[take] >= 10) | (h[take] < 1e-12))
        bad = take[limited]; reasons[bad] = FiberStatus.INTEGRATION_FAILED; active[bad] = False
        take = take[~limited]
        if not len(take):
            continue
        h[take] = np.minimum(h[take], .25*field.cell_diameters[ids[take]])
        end, error, dense, norm = _rk23_step(field, ids[take], q[take], h[take], sign)
        scale = tolerance_scale*(1e-10+1e-7*np.maximum(abs(q[take]), abs(end)))
        err = np.max(abs(error)/scale, axis=1)
        good = np.isfinite(end).all(1) & np.isfinite(err) & (err <= 1) & (norm > 1e-10)
        for offset, j in enumerate(take):
            if not good[offset]:
                h[j] *= .5; rejected[j] += 1
                if norm[offset] <= 1e-10 and h[j] < 1e-12:
                    reasons[j] = FiberStatus.WEAK_DIRECTION; active[j] = False
                continue
            fraction, face = _first_face_event(field, ids[j], dense[offset])
            next_q = _polynomial_point(dense[offset], fraction)
            old_r, new_r = _fiber_values(field, np.array([ids[j], ids[j]]), np.stack((q[j], next_q)))
            if sign*(new_r-old_r) < -1e-9:
                reasons[j] = FiberStatus.NONMONOTONE; active[j] = False; continue
            if target_r is not None and not np.isfinite(targets[j]).all():
                target = target_r[j]
                if abs(old_r-target) <= 1e-12:
                    targets[j] = q[j]
                elif old_r <= target <= new_r:
                    root = brentq(lambda t: _fiber_values(field, np.array([ids[j]]),
                        _polynomial_point(dense[offset], t)[None])[0]-target, 0, fraction, xtol=1e-12)
                    targets[j] = _polynomial_point(dense[offset], root)
            q[j] = next_q; steps[j] += 1; length[j] += fraction*h[j]
            if paths is not None:
                paths[j].append(next_q.copy())
            h[j] *= np.clip(.9*max(err[offset], 1e-12)**(-1/3), .2, 5.)
            if face >= 0:
                neighbor = field.neighbors[ids[j], face]
                if neighbor < 0:
                    faces[j] = field.boundary_inner_ids[ids[j], face]
                    reasons[j] = FiberStatus.HIT_INNER if faces[j] >= 0 else FiberStatus.HIT_OUTER
                    active[j] = False
                else:
                    ids[j] = neighbor; resolve[j] = True
        if time.monotonic()-last_print >= 30:
            print(f'trace sign={sign}: complete={int((~active).sum())}/{n}; steps={int(steps.sum())}', flush=True)
            last_print = time.monotonic()
    return q, reasons, faces, targets, steps


def _surface_weights(field, face_ids, points):
    tri = field.vertices[field.canonical.computational_inner_vertex_indices][field.canonical.computational_inner_faces[face_ids]]
    a, b, d = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], points-tri[:, 0]
    aa, ab, bb = (np.einsum('ni,ni->n', x, y) for x, y in [(a, a), (a, b), (b, b)])
    da, db = np.einsum('ni,ni->n', d, a), np.einsum('ni,ni->n', d, b)
    u = (da*bb-db*ab)/(aa*bb-ab*ab); v = (db*aa-da*ab)/(aa*bb-ab*ab)
    return np.c_[1-u-v, u, v]


def _origin_labels(field, faces, weights, valid):
    n = len(faces); attached = np.full((n, 3), -1, dtype=np.int64)
    bary = np.full((n, 3, 3), np.nan)
    ids = np.flatnonzero(valid)
    if len(ids):
        vertices = field.canonical.computational_inner_faces[faces[ids]]
        attached[ids] = field.canonical.computational_to_canonical_face_indices[vertices]
        bary[ids] = field.canonical.computational_to_canonical_barycentric[vertices]
    labels = {}
    for name, record in field.region_labels.items():
        values = np.full((n, len(record['names'])), np.nan)
        if len(ids):
            face_labels = np.asarray(record['face_labels'])
            good = ids[np.all(attached[ids] >= 0, axis=1)]
            for k in range(len(record['names'])):
                values[good, k] = np.sum(weights[good]*(face_labels[attached[good]] == k), axis=1)
        labels[name] = values
    return labels, dict(anatomical_face_indices=attached, anatomical_barycentric=bary,
                        attachment_weights=weights.copy())


def canonical_to_semantic(field, points, *, tolerance_scale=1., located=None):
    points = np.asarray(points, dtype=float)
    mapping = field.locator.locate(points) if located is None else located
    n = len(points); ids = mapping.coordinates.tetrahedron_indices
    status = np.full(n, int(FiberStatus.INVALID_VOLUME), dtype=np.int16)
    face = np.full(n, -1, dtype=np.int64); weights = np.full((n, 3), np.nan)
    rho = np.full(n, np.nan); origins = np.full((n, 3), np.nan); ends = origins.copy(); reconstructed = origins.copy()
    back = status.copy(); forward = status.copy(); errors = np.full(n, np.nan); steps = np.zeros(n, dtype=np.int64)
    good = np.flatnonzero(mapping.mappable_mask)
    if len(good):
        rho[good] = evaluate_semantic_field(field.scalar, ids[good], mapping.coordinates.barycentric_weights[good])[0]
        q = (points[good]-field.origin)/field.diagonal
        inner, why, f, _, count = _trace_fibers(field, q, ids[good], -1, tolerance_scale=tolerance_scale)
        back[good] = why; status[good] = why; steps[good] += count
        reached = good[why == FiberStatus.HIT_INNER]
        local = np.flatnonzero(why == FiberStatus.HIT_INNER)
        if len(reached):
            face[reached] = f[local]; weights[reached] = _surface_weights(field, f[local], inner[local])
            origins[reached] = inner[local]*field.diagonal+field.origin
            location = field.locator.locate(origins[reached])
            outer, why2, _, target, count2 = _trace_fibers(field, inner[local],
                location.coordinates.tetrahedron_indices, 1, rho[reached], tolerance_scale)
            forward[reached] = why2; steps[reached] += count2
            ends[reached] = outer*field.diagonal+field.origin
            reconstructed[reached] = target*field.diagonal+field.origin
            errors[reached] = np.linalg.norm(reconstructed[reached]-points[reached], axis=1)
            status[reached] = why2
            ok = (why2 == FiberStatus.HIT_OUTER) & np.isfinite(errors[reached]) & (errors[reached] <= 1e-6*field.diagonal)
            status[reached[(why2 == FiberStatus.HIT_OUTER) & ~ok]] = FiberStatus.ROUND_TRIP_FAILED
            accepted = reached[ok]
            cap = field.canonical.computational_inner_face_labels[face[accepted]] == BOUNDARY_KNEE_TRUNCATION
            status[accepted] = np.where(cap, FiberStatus.VALID_ARTIFICIAL_CAP, FiberStatus.VALID_ANATOMICAL)
    # Expected boundaries reached in the wrong direction are not successful fibers.
    status[np.isin(status, [FiberStatus.HIT_INNER, FiberStatus.HIT_OUTER])] = FiberStatus.BOUNDARY_AMBIGUITY
    valid = np.isin(status, [FiberStatus.VALID_ANATOMICAL, FiberStatus.VALID_ARTIFICIAL_CAP])
    labels, correspondence = _origin_labels(field, face, weights, valid)
    face[~valid] = -1; weights[~valid] = np.nan
    return FiberQueryResult(FiberCoordinates(face, weights, rho), reconstructed, origins, ends,
        status, mapping.status_codes.copy(), back, forward, errors, steps, labels, correspondence)


def semantic_to_canonical(field, face_indices, barycentric_weights, semantic_r, *, tolerance_scale=1.):
    faces = np.asarray(face_indices); weights = np.asarray(barycentric_weights, dtype=float)
    rho = np.asarray(semantic_r, dtype=float)
    if (faces.ndim != 1 or not np.issubdtype(faces.dtype, np.integer) or weights.shape != (len(faces), 3)
            or rho.shape != (len(faces),) or np.any(faces < 0)
            or np.any(faces >= len(field.canonical.computational_inner_faces))
            or not np.isfinite(weights).all() or not np.isfinite(rho).all()
            or np.any(weights < -1e-10) or np.any(abs(weights.sum(1)-1) > 1e-10)
            or np.any(rho < 0) or np.any(rho > 1)):
        raise ValueError('invalid semantic coordinates')
    tri = field.vertices[field.canonical.computational_inner_vertex_indices][field.canonical.computational_inner_faces[faces]]
    start = np.einsum('ni,nij->nj', weights, tri)
    location = field.locator.locate(start*field.diagonal+field.origin)
    _, why, _, target, _ = _trace_fibers(field, start, location.coordinates.tetrahedron_indices, 1, rho, tolerance_scale)
    target[why != FiberStatus.HIT_OUTER] = np.nan
    return field.locator.locate(target*field.diagonal+field.origin)


def instance_to_semantic(field, instance_map, points, *, input_frame='normalized_shoe', tolerance_scale=1.):
    location = instance_map.instance_to_canonical(points, input_frame=input_frame)
    return canonical_to_semantic(field, location.canonical_points, tolerance_scale=tolerance_scale, located=location)


def semantic_to_instance(field, instance_map, face_indices, barycentric_weights, semantic_r, *, output_frame='normalized_shoe'):
    result = semantic_to_canonical(field, face_indices, barycentric_weights, semantic_r)
    points = np.full(result.canonical_points.shape, np.nan)
    mask = result.mappable_mask
    points[mask] = instance_map.canonical_to_instance(result.coordinates.tetrahedron_indices[mask],
        result.coordinates.barycentric_weights[mask], output_frame=output_frame)
    return points


def sample_fiber_surface(vertices, faces, area_vertices, count, selected_faces=None):
    """Nested deterministic equal-area queries; targeted populations stay separate."""
    from scipy.stats import qmc
    triangles = area_vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]),axis=1)/2
    population = np.arange(len(faces)) if selected_faces is None else np.unique(selected_faces)
    population = population[area[population] > 0]
    total = float(area[population].sum())
    if not total > 0:
        raise ValueError('shoe sampling population has no positive area')
    sequence = qmc.Halton(3, scramble=False); sequence.fast_forward(1)
    u = sequence.random(count)
    face_ids = population[np.searchsorted(np.cumsum(area[population]),u[:,0]*total,side='right')]
    root = np.sqrt(u[:,1]); weights = np.c_[1-root,root*(1-u[:,2]),root*u[:,2]]
    points = np.einsum('ni,nij->nj',weights,vertices[faces[face_ids]])
    return face_ids, weights, points, total


def fiber_category_codes(result):
    """Disjoint observed categories, retaining the unchanged 11-C invalid codes."""
    codes = result.status_codes.astype(np.int16).copy()
    bad = codes == FiberStatus.INVALID_VOLUME
    codes[bad] = 1000+result.volume_status_codes[bad]
    return codes


def fiber_category_name(code):
    from .instance_volume_mapping import VolumePointStatus
    return ('11c_'+VolumePointStatus(int(code)-1000).name.lower() if code >= 1000
            else FiberStatus(int(code)).name.lower())


def summarize_fiber_coverage(codes, area):
    values, counts = np.unique(codes,return_counts=True)
    valid_volume = int(np.sum(codes < 1000)); n=len(codes)
    return {fiber_category_name(k):dict(count=int(count),estimated_area=float(area*count/n),
        total_area_fraction=float(count/n),
        valid_volume_area_fraction=float(count/valid_volume) if k < 1000 and valid_volume else None)
        for k,count in zip(values,counts)}


def selected_fiber_paths(field, points):
    location=field.locator.locate(points); curves=[]; reasons=[]
    for j in np.flatnonzero(location.mappable_mask):
        paths=[]; start=(points[j:j+1]-field.origin)/field.diagonal
        end, why, _, _, _=_trace_fibers(field,start,location.coordinates.tetrahedron_indices[j:j+1],-1,paths=paths)
        if why[0] == FiberStatus.HIT_INNER:
            loc=field.locator.locate(end*field.diagonal+field.origin); paths=[]
            _,why,_,_,_=_trace_fibers(field,end,loc.coordinates.tetrahedron_indices,1,paths=paths)
        curves.append(np.asarray(paths[0])*field.diagonal+field.origin); reasons.append(int(why[0]))
    return curves, reasons
