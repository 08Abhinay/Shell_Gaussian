"""Small mathematical checks for the experimental quadratic field."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from foot_prior.anatomical_fibers import (
    CanonicalSemanticField, EDGES, _basis, _basis_gradients, _boundary_simplices,
    _layout, _stationary_points, _stiffness, audit_semantic_field,
    evaluate_semantic_field, refine_scalar_neighborhoods, solve_canonical_semantic_field,
)
from foot_prior.anatomy import array_digest
from scripts.run_anatomical_fibers import _load_field, _write_artifacts


def mesh():
    # Two positively oriented tetrahedra, sharing face 123.
    return SimpleNamespace(volume_vertices=np.array([[0., 0., 0.], [1., 0., 0.],
        [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
        tetrahedra=np.array([[0, 1, 2, 3], [4, 1, 3, 2]]),
        boundary_faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3],
                                 [4, 1, 2], [4, 1, 3], [4, 2, 3]]),
        boundary_labels=np.array([1, 1, 1, 5, 5, 5]), harmonic_r=np.arange(5.) / 5)


def field_for(c):
    # This fixture's two Dirichlet boundaries touch. Test layout separately
    # with disjoint boundary faces; evaluation only needs the shared topology.
    edges, inv = np.unique(np.sort(c.tetrahedra[:, EDGES].reshape(-1, 2), axis=1),
                           axis=0, return_inverse=True)
    from foot_prior.anatomical_volume import _tetrahedron_gradients
    coef = np.r_[c.harmonic_r, c.harmonic_r[edges].mean(axis=1)]
    return CanonicalSemanticField(coef, edges, np.c_[c.tetrahedra, 5 + inv.reshape(-1, 6)],
        _tetrahedron_gradients(c.volume_vertices, c.tetrahedra), np.array([], dtype=int),
        np.array([], dtype=int), np.linalg.norm(np.ptp(c.volume_vertices, axis=0)),
        array_digest(c.volume_vertices, c.tetrahedra), dict(converged=True))


def shell_mesh():
    # Conforming cube shell constructed from the six surface quads, each
    # split consistently into triangular prisms, then tetrahedra.
    outer = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = np.vstack((outer * .4, outer))
    triangles = []
    for axis in range(3):
        for value in (-1, 1):
            ids = np.flatnonzero(outer[:, axis] == value)
            a, b, c, d = ids
            triangles.extend([(a, b, d), (a, d, c)])
    cells = []
    for triangle in triangles:
        a, b, c = sorted(triangle)
        cells.extend([(a, b, c, c+8), (a, b, b+8, c+8), (a, a+8, b+8, c+8)])
    from foot_prior.anatomical_volume import orient_tetrahedra_positive
    cells, _ = orient_tetrahedra_positive(vertices, np.asarray(cells))
    return SimpleNamespace(volume_vertices=vertices, tetrahedra=cells,
        boundary_faces=np.vstack((triangles, np.asarray(triangles)+8)),
        boundary_labels=np.r_[np.ones(12, dtype=int), np.full(12, 5)],
        harmonic_r=np.r_[np.zeros(8), np.ones(8)])


def test_basis_linear_midpoint_and_gradients():
    c = mesh()
    field = field_for(c)
    w = np.array([[.13, .27, .19, .41]])
    np.testing.assert_allclose(_basis(w).sum(axis=1), 1)
    val, grad = evaluate_semantic_field(field, np.array([0]), w)
    np.testing.assert_allclose(val, w @ c.harmonic_r[c.tetrahedra[0]])
    np.testing.assert_allclose(grad, [[.2, .4, .6]])
    rng = np.random.default_rng(2)
    field = replace(field, coefficients=rng.uniform(size=len(field.coefficients)))
    val, grad = evaluate_semantic_field(field, np.array([0]), w)
    for axis in range(3):
        dw = np.zeros((1, 4)); dw[0, axis+1] = 1e-6; dw[0, 0] = -1e-6
        plus = evaluate_semantic_field(field, np.array([0]), w+dw)[0]
        minus = evaluate_semantic_field(field, np.array([0]), w-dw)[0]
        np.testing.assert_allclose((plus-minus)/2e-6, grad[:, axis], atol=1e-9)
    midpoint = evaluate_semantic_field(field, np.array([0]), np.array([[.5, .5, 0, 0]]))[0]
    ids = field.cell_coefficients[0]
    np.testing.assert_allclose(midpoint, (field.coefficients[ids[0]] + field.coefficients[ids[1]]
                                        + 2*field.coefficients[ids[4]])/4)
    shared = evaluate_semantic_field(field, np.array([0, 1]),
                                    np.array([[0, .2, .3, .5], [0, .2, .5, .3]]))[0]
    np.testing.assert_allclose(shared[0], shared[1], atol=1e-14)


def test_layout_solver_and_energy_gradient():
    c = shell_mesh()
    edges, ids, zero, one, diagonal, gradients = _layout(c)
    # Surface diagonals are boundary; radial edges joining boundaries remain free.
    radial = np.flatnonzero(np.all(edges == [0, 8], axis=1))[0] + len(c.volume_vertices)
    assert radial not in zero and radial not in one
    k = _stiffness(c, ids, gradients, diagonal)
    x = np.linspace(.1, .9, k.shape[0])
    for i in (0, radial, len(x)-1):
        d = np.zeros_like(x); d[i] = 1e-6
        numerical = (.5*(x+d)@(k@(x+d)) - .5*(x-d)@(k@(x-d))) / 2e-6
        np.testing.assert_allclose(numerical, (k@x)[i], atol=1e-9)
    before = c.volume_vertices.copy()
    field = solve_canonical_semantic_field(c)
    assert field.solver['converged']
    assert np.all(field.coefficients[zero] == 0)
    assert np.all(field.coefficients[one] == 1)
    assert np.all((field.coefficients >= 0) & (field.coefficients <= 1))
    np.testing.assert_array_equal(before, c.volume_vertices)


@pytest.mark.parametrize('kind', ['isolated', 'plane', 'interface', 'boundary', 'outside'])
def test_stationary_detection(kind):
    c = mesh()
    q = c.volume_vertices[c.tetrahedra[:1]]
    if kind == 'plane':
        g = np.zeros_like(q); g[..., 0] = q[..., 0] - .2
    else:
        target = dict(isolated=[.2, .2, .2], interface=[1/3]*3,
                      boundary=[.2, .2, 0], outside=[2., 2., 2.])[kind]
        g = q - target
    records = _stationary_points(c.tetrahedra[:1], g, _boundary_simplices(c))
    locations = {r[1] for r in records}
    if kind == 'outside':
        assert not records
    else:
        expected = dict(isolated='cell_interior', plane='cell_interior',
                        interface='internal_interface', boundary='domain_boundary')[kind]
        assert expected in locations


def test_artifacts_and_source_rejection(tmp_path):
    c = shell_mesh()
    field = solve_canonical_semantic_field(c)
    report, arrays = audit_semantic_field(c, field)
    _write_artifacts(tmp_path, c, field, report, arrays, [])
    loaded = _load_field(c, tmp_path)
    np.testing.assert_array_equal(loaded.coefficients, field.coefficients)
    vtk = (tmp_path/'semantic_field.vtk').read_text()
    assert 'semantic_r' in vtk and 'critical_cell_class' in vtk
    with pytest.raises(FileExistsError):
        _write_artifacts(tmp_path, c, field, report, arrays, [])
    metadata = json.loads((tmp_path/'semantic_field.json').read_text())
    metadata['source_geometry_digest'] = 'modified'
    (tmp_path/'semantic_field.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='source'):
        _load_field(c, tmp_path)


def test_scalar_refinement_is_conforming_and_preserves_geometry():
    c = shell_mesh()
    old_vertices = c.volume_vertices.copy()
    old_cells = c.tetrahedra.copy()
    refined = refine_scalar_neighborhoods(c, np.array([0]))
    assert len(refined.refined_parent_indices) > 1
    np.testing.assert_array_equal(c.volume_vertices, old_vertices)
    np.testing.assert_array_equal(c.tetrahedra, old_cells)
    np.testing.assert_array_equal(refined.boundary_faces, c.boundary_faces)
    np.testing.assert_array_equal(refined.boundary_labels, c.boundary_labels)
    np.testing.assert_array_equal(refined.volume_vertices[:len(old_vertices)], old_vertices)
    def volumes(v, t):
        corners = v[t]
        return np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)) / 6
    volume = volumes(refined.volume_vertices, refined.tetrahedra)
    assert np.all(volume > 0)
    summed = np.bincount(refined.parent_tetrahedron_indices, weights=volume, minlength=len(old_cells))
    np.testing.assert_allclose(summed, volumes(old_vertices, old_cells), atol=1e-15)
    from foot_prior.anatomical_fibers import FACE_CORNERS
    faces, counts = np.unique(np.sort(refined.tetrahedra[:, FACE_CORNERS].reshape(-1, 3), axis=1),
                              axis=0, return_counts=True)
    assert np.all(counts <= 2)
    assert set(map(tuple, faces[counts == 1])) == set(map(tuple, np.sort(c.boundary_faces, axis=1)))


def test_refined_artifact_reconstructs_from_parent_ids(tmp_path):
    c = shell_mesh()
    refined = refine_scalar_neighborhoods(c, np.array([0]))
    field = solve_canonical_semantic_field(refined)
    report, arrays = audit_semantic_field(refined, field)
    report['refinement'] = dict(canonical_source_geometry_digest=array_digest(c.volume_vertices, c.tetrahedra),
                                refined_parent_indices=refined.refined_parent_indices.tolist())
    _write_artifacts(tmp_path, refined, field, report, arrays, [])
    loaded = _load_field(c, tmp_path)
    np.testing.assert_array_equal(field.coefficients, loaded.coefficients)
    np.testing.assert_array_equal(field.cell_coefficients, loaded.cell_coefficients)


def fiber_box():
    from itertools import product
    from scipy.spatial import Delaunay
    from foot_prior.anatomical_volume import _tetrahedron_gradients
    from foot_prior.anatomical_fibers import _prepare_fiber_field
    v = np.asarray(list(product([0., 1.], repeat=3)))
    t = Delaunay(v).simplices.copy()
    volume = np.linalg.det((v[t[:, 1:]]-v[t[:, :1]]).transpose(0, 2, 1))/6
    for k in np.flatnonzero(volume < 0):
        t[k, [0, 1]] = t[k, [1, 0]]
    faces, count = np.unique(np.sort(t[:, [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]].reshape(-1, 3), axis=1),
                             axis=0, return_counts=True)
    boundary = faces[count == 1]
    inner = boundary[np.all(v[boundary, 0] == 0, axis=1)]
    labels = np.where(np.all(v[boundary, 0] == 0, axis=1), 1, 5)
    edges, inverse = np.unique(np.sort(t[:, EDGES].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    coefficients = np.r_[v[:, 0], v[edges, 0].mean(1)]
    c = SimpleNamespace(volume_vertices=v, tetrahedra=t, tetrahedron_signed_volumes=abs(volume),
        boundary_faces=boundary, boundary_labels=labels, harmonic_r=v[:, 0],
        computational_inner_vertex_indices=np.arange(4), computational_inner_faces=inner,
        computational_inner_face_labels=np.ones(len(inner), dtype=int),
        computational_to_canonical_face_indices=np.zeros(4,dtype=int),
        computational_to_canonical_barycentric=np.tile([1.,0,0],(4,1)))
    scalar = CanonicalSemanticField(coefficients, edges, np.c_[t, len(v)+inverse.reshape(-1,6)],
        _tetrahedron_gradients(v,t), np.flatnonzero(coefficients==0), np.flatnonzero(coefficients==1),
        np.sqrt(3), array_digest(v,t), {'converged':True})
    f = _prepare_fiber_field(c,scalar,np.tile([1.,0,0],(8,1)),np.zeros(len(t),dtype=np.int16),{}, {})
    return f


def test_direction_fem_constant_gradient():
    from scipy.sparse.linalg import splu
    from foot_prior.anatomical_fibers import _direction_system
    f = fiber_box()
    matrix, rhs, g, h = _direction_system(f.canonical, f.scalar)
    values = splu(matrix).solve(rhs)
    np.testing.assert_allclose(values, np.tile([np.sqrt(3),0,0],(8,1)), atol=1e-13)
    assert np.linalg.norm(matrix@values-rhs) < 1e-12 and h > 0


def test_fiber_round_trip_and_faces():
    from foot_prior.anatomical_fibers import canonical_to_semantic, semantic_to_canonical, FiberStatus
    f=fiber_box(); q=np.array([[.23,.31,.47],[.77,.41,.63],[0.,.32,.61]])
    result=canonical_to_semantic(f,q)
    np.testing.assert_array_equal(result.status_codes,FiberStatus.VALID_ANATOMICAL)
    np.testing.assert_allclose(result.inner_origins[:,0],0,atol=1e-10)
    np.testing.assert_allclose(result.outer_endpoints[:,0],1,atol=1e-10)
    np.testing.assert_allclose(result.canonical_points,q,atol=1e-9)
    restored=semantic_to_canonical(f,result.coordinates.face_indices,result.coordinates.barycentric_weights,
                                   result.coordinates.semantic_r)
    np.testing.assert_allclose(restored.canonical_points,q,atol=1e-9)


def test_fiber_never_crosses_exclusion():
    from foot_prior.anatomical_fibers import canonical_to_semantic,FiberStatus
    f=fiber_box(); mask=np.full(len(f.canonical.tetrahedra),int(FiberStatus.ORIGINAL_SADDLE),dtype=np.int16)
    result=canonical_to_semantic(replace(f,exclusions=mask),np.array([[.2,.31,.47]]))
    assert result.status_codes[0] == FiberStatus.ORIGINAL_SADDLE
    assert not result.mappable_mask[0]


def test_progress_certificate_rejects_saddle_and_negative():
    from foot_prior.anatomical_fibers import _certify_progress,FiberStatus
    v=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    boundary=(set(),set(),set())
    assert _certify_progress(np.ones((4,4)),v,np.arange(4),boundary)==0
    assert _certify_progress(-np.eye(4),v,np.arange(4),boundary)==FiberStatus.DIRECTION_INCOMPATIBLE
    assert _certify_progress(np.zeros((4,4)),v,np.arange(4),boundary)==FiberStatus.DIRECTION_UNRESOLVED


def test_rk23_matches_scipy():
    from scipy.integrate import solve_ivp
    from foot_prior.anatomical_fibers import _rk23_step,_unit_direction
    f=fiber_box(); f=replace(f,directions=np.c_[np.ones(8),.1*f.vertices[:,0],np.zeros(8)])
    q=np.array([[.18,.21,.24]])/f.diagonal
    ids=f.locator.locate(q*f.diagonal).coordinates.tetrahedron_indices
    end,_,_,_=_rk23_step(f,ids,q,np.array([1e-4]),1)
    expected=solve_ivp(lambda t,y:_unit_direction(f,ids,y[None],1)[0][0],(0,1e-4),q[0],
                       method='RK23',rtol=1e-12,atol=1e-14).y[:,-1]
    np.testing.assert_allclose(end[0],expected,atol=1e-12)
