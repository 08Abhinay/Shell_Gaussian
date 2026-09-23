"""Addresses on a shell whose fibers are known exactly.

The geometry is a unit box tetrahedralized into a grid, with the ``y = 0``
face standing in for the skin and a direction field pointing straight up. Then
the fiber through any point is the vertical line through it, so the address of
``(x, y, z)`` is the surface point ``(x, 0, z)`` at outward progress ``y``, and
the fiber length is ``y``. Every assertion below compares against that, not
against another run of the same code.
"""

from __future__ import annotations

import numpy as np
import pytest

from anatomical_coordinates.coordinate_mapping.address import (
    ADDRESSED,
    REACHED,
    AddressBook,
    CanonicalCorrespondence,
    FiberTracer,
)
from anatomical_coordinates.coordinate_mapping.lookup import CanonicalSemantics

#: Kuhn decomposition of a cube; corner bits are (x, y, z).
_CUBE = ((0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4),
         (0, 3, 2, 7), (0, 6, 4, 7), (0, 2, 6, 7))


def _slab(path, cells: int = 4):
    """A tetrahedralized unit box, written in the canonical volume's format."""

    axis = np.linspace(0.0, 1.0, cells + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    vertices = grid.reshape(-1, 3)
    stride = np.array([(cells + 1) ** 2, cells + 1, 1])

    def corner(i, j, k, bits):
        return int(
            (i + (bits >> 2 & 1)) * stride[0]
            + (j + (bits >> 1 & 1)) * stride[1]
            + (k + (bits & 1)) * stride[2]
        )

    tetrahedra = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for cell in _CUBE:
                    tetrahedra.append([corner(i, j, k, bits) for bits in cell])
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    # Orientation is not guaranteed by the decomposition; make every cell
    # positive so the barycentric inverse is well conditioned.
    corners = vertices[tetrahedra]
    volume = np.einsum(
        "ni,ni->n",
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        corners[:, 3] - corners[:, 0],
    )
    flip = volume < 0
    tetrahedra[flip] = tetrahedra[flip][:, [0, 2, 1, 3]]

    floor = np.nonzero(vertices[:, 1] == 0.0)[0]
    position = {int(v): n for n, v in enumerate(floor)}
    faces = []
    for i in range(cells):
        for k in range(cells):
            a = position[corner(i, 0, k, 0b000)]
            b = position[corner(i, 0, k, 0b100)]
            c = position[corner(i, 0, k, 0b101)]
            d = position[corner(i, 0, k, 0b001)]
            faces.extend(([a, b, c], [a, c, d]))
    faces = np.asarray(faces, dtype=np.int64)

    np.savez(
        path,
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        harmonic_r=vertices[:, 1].copy(),
        boundary_faces=faces,
        boundary_labels=np.zeros(len(faces), dtype=np.int16),
        boundary_label_names=np.array(["skin", "knee_truncation"]),
        computational_inner_faces=faces,
        computational_inner_vertex_indices=floor,
        computational_inner_face_labels=np.zeros(len(faces), dtype=np.int16),
    )
    return path


@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    path = _slab(tmp_path_factory.mktemp("shell") / "volume.npz")
    semantics = CanonicalSemantics(path, resolution=8)
    directions = np.zeros((len(semantics.vertices), 3))
    directions[:, 1] = 1.0                      # every fiber runs straight up
    return semantics, FiberTracer(semantics, directions, device="cpu")


def _interior(count: int = 64, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    points = generator.uniform(0.12, 0.88, size=(count, 3))
    return points


def test_adjacency_is_symmetric(shell):
    _, tracer = shell
    cells, slots = np.nonzero(tracer.neighbours >= 0)
    other = tracer.neighbours[cells, slots]
    assert np.isin(cells, tracer.neighbours[other]).all()


def test_hinted_locate_matches_the_index(shell):
    semantics, tracer = shell
    points = _interior(200, seed=5)
    reference = semantics.locate_fast(points, device="cpu", fallback=False)
    hinted, _ = tracer._locate(points, hint=reference.tetrahedron)
    assert (hinted == reference.tetrahedron).all()


def test_fiber_lands_directly_below_its_start(shell):
    _, tracer = shell
    points = _interior()
    landing = tracer.trace_in(points, step=0.01, max_steps=400)
    assert landing.found.all()
    assert np.allclose(landing.point[:, [0, 2]], points[:, [0, 2]], atol=2e-3)
    assert np.allclose(landing.point[:, 1], 0.0, atol=1e-9)
    # The fiber is vertical, so its length is exactly the starting height.
    assert np.allclose(landing.arclength, points[:, 1], atol=2e-3)


def test_landing_weights_reproduce_the_landing_point(shell):
    _, tracer = shell
    landing = tracer.trace_in(_interior(), step=0.01, max_steps=400)
    corners = tracer.triangles[landing.face]
    rebuilt = np.einsum("ni,nij->nj", landing.barycentric, corners)
    assert np.allclose(rebuilt, landing.point, atol=1e-9)
    assert np.allclose(landing.barycentric.sum(axis=1), 1.0)
    assert (landing.barycentric >= -1e-12).all()


def test_tracing_out_reaches_the_requested_progress(shell):
    _, tracer = shell
    points = _interior()
    landing = tracer.trace_in(points, step=0.01, max_steps=400)
    target = points[:, 1]
    back = tracer.trace_out(
        landing.face, landing.barycentric, target, step=0.01, max_steps=400
    )
    assert np.isfinite(back).all()
    # Vertical fibers make this exact: the point at progress t is at height t.
    assert np.allclose(back[:, 1], target, atol=1e-6)
    assert np.allclose(back, points, atol=3e-3)


def test_projection_finds_the_triangle_a_point_lies_on(shell):
    _, tracer = shell
    generator = np.random.default_rng(3)
    chosen = generator.integers(0, len(tracer.triangles), 50)
    weights = generator.dirichlet(np.ones(3), size=50)
    points = np.einsum("ni,nij->nj", weights, tracer.triangles[chosen])
    face, found, gap = tracer.project_to_surface(points)
    assert np.allclose(gap, 0.0, atol=1e-9)
    rebuilt = np.einsum("ni,nij->nj", found, tracer.triangles[face])
    assert np.allclose(rebuilt, points, atol=1e-9)


def test_correspondence_table_agrees_with_tracing(shell):
    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    assert (table.status == REACHED).all()
    # Every vertex's fiber starts directly below it.
    assert np.allclose(table.origin[:, [0, 2]], semantics.vertices[:, [0, 2]], atol=2e-3)
    assert np.allclose(table.origin[:, 1], 0.0, atol=1e-9)
    assert np.allclose(table.arclength, semantics.vertices[:, 1], atol=2e-3)


def test_address_book_round_trips_through_a_shoe(shell):
    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    shift = np.array([0.0, 0.0, 0.05])
    book = AddressBook(
        semantics, tracer, table,
        to_canonical=lambda p: p - shift,     # a stand-in for the flow map
        to_shoe=lambda p: p + shift,
    )
    points = _interior(48, seed=9) + shift
    address = book.query(points, exact=True, step=0.01, max_steps=400)
    assert (address.outcome == ADDRESSED).all()
    # r is the harmonic field, which on this shell is height above the skin.
    assert np.allclose(address.r, points[:, 1], atol=1e-6)

    back = book.place(
        address.face, address.barycentric, address.r, step=0.01, max_steps=400
    )
    assert np.allclose(back, points, atol=3e-3)


def test_verification_reports_the_round_trip_distance(shell):
    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    book = AddressBook(
        semantics, tracer, table, lambda p: p, lambda p: p
    )
    points = _interior(32, seed=11)
    address = book.query(
        points, exact=True, verify=True, tolerance_mm=1.0,
        step=0.01, max_steps=400,
    )
    assert np.isfinite(address.residual[address.addressed]).all()
    assert (address.residual[address.addressed] >= 0.0).all()


def test_interpolated_and_exact_agree_on_a_linear_field(shell):
    """With straight fibers the correspondence is linear, so they must match."""

    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    book = AddressBook(semantics, tracer, table, lambda p: p, lambda p: p)
    points = _interior(64, seed=13)
    fast = book.query(points, exact=False, step=0.01, max_steps=400)
    slow = book.query(points, exact=True, step=0.01, max_steps=400)
    assert np.allclose(
        fast.surface_point(tracer), slow.surface_point(tracer), atol=5e-3
    )


def test_a_point_on_the_skin_is_addressed_at_zero(shell):
    """The skin itself has to be addressable, or r = 0 names nothing."""

    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    book = AddressBook(semantics, tracer, table, lambda p: p, lambda p: p)

    generator = np.random.default_rng(17)
    chosen = generator.integers(0, len(tracer.triangles), 32)
    weights = generator.dirichlet(np.ones(3), size=32)
    on_skin = np.einsum("ni,nij->nj", weights, tracer.triangles[chosen])
    assert np.allclose(on_skin[:, 1], 0.0)

    address = book.query(on_skin, exact=True, step=0.01, max_steps=400)
    assert (address.outcome == ADDRESSED).all()
    assert np.allclose(address.r, 0.0, atol=1e-9)
    assert np.allclose(address.arclength, 0.0, atol=1e-6)
    # The address of a surface point is that point.
    assert np.allclose(address.surface_point(tracer), on_skin, atol=1e-6)


def test_places_at_the_skin_come_back_to_the_skin(shell):
    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    book = AddressBook(semantics, tracer, table, lambda p: p, lambda p: p)
    generator = np.random.default_rng(23)
    chosen = generator.integers(0, len(tracer.triangles), 32)
    weights = generator.dirichlet(np.ones(3), size=32)
    placed = book.place(chosen, weights, np.zeros(32), step=0.01, max_steps=400)
    expected = np.einsum("ni,nij->nj", weights, tracer.triangles[chosen])
    assert np.allclose(placed, expected, atol=1e-9)


def test_an_addressed_point_carries_no_missing_field(shell):
    semantics, tracer = shell
    table = CanonicalCorrespondence.build(tracer, step=0.01, max_steps=400)
    book = AddressBook(semantics, tracer, table, lambda p: p, lambda p: p)
    points = np.concatenate((
        _interior(32, seed=29),
        np.einsum(
            "ni,nij->nj",
            np.full((8, 3), 1.0 / 3.0),
            tracer.triangles[np.arange(8)],
        ),
    ))
    address = book.query(points, exact=True, step=0.01, max_steps=400)
    ok = address.addressed
    assert ok.any()
    assert np.isfinite(address.r[ok]).all()
    assert np.isfinite(address.arclength[ok]).all()
    assert (address.face[ok] >= 0).all()
    assert np.isfinite(address.barycentric[ok]).all()
