"""HARD GATE 2: sign convention, axis order, interpolation and gradients.

Everything here runs on synthetic geometry whose answer is known analytically,
so a failure points at the field code rather than at a shoe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anatomical_coordinates.cavity_field import (
    ShoeCavityField,
    build_cavity_field,
    inside_domain,
    sample_field,
    sample_footbed_height,
)
from foot_prior.mesh import TriangleMesh


def _quad(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return corners, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _box_shoe(
    length: float = 1.0,
    half_width: float = 0.2,
    height: float = 0.3,
    floor_y: float = 0.0,
) -> tuple[TriangleMesh, TriangleMesh, np.ndarray, np.ndarray]:
    """An open-topped rectangular trough: floor is footbed, walls are obstacles.

    Shoe +Y points down, so the ceiling sits at ``floor_y - height`` and the
    interior is ``floor_y - height < y < floor_y``, ``|z| < half_width``.
    """

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    def add(corners):
        offset = len(vertices)
        for corner in corners:
            vertices.append(np.asarray(corner, dtype=np.float64))
        faces.append(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + offset)

    top = floor_y - height
    # floor (footbed) first so its face indices are 0 and 1
    add([(0, floor_y, -half_width), (length, floor_y, -half_width),
         (length, floor_y, half_width), (0, floor_y, half_width)])
    footbed_faces = np.asarray([0, 1], dtype=np.int64)
    # ceiling
    add([(0, top, -half_width), (length, top, -half_width),
         (length, top, half_width), (0, top, half_width)])
    # two side walls
    add([(0, floor_y, -half_width), (length, floor_y, -half_width),
         (length, top, -half_width), (0, top, -half_width)])
    add([(0, floor_y, half_width), (length, floor_y, half_width),
         (length, top, half_width), (0, top, half_width)])

    all_vertices = np.stack(vertices)
    all_faces = np.concatenate(faces, axis=0)
    shoe = TriangleMesh(all_vertices, all_faces)
    footbed = TriangleMesh(
        all_vertices[all_faces[footbed_faces].reshape(-1)],
        np.arange(6, dtype=np.int64).reshape(2, 3),
    )
    centerline = np.stack(
        (np.linspace(0.0, length, 16), np.zeros(16)), axis=1
    )
    return shoe, footbed, footbed_faces, centerline


@pytest.fixture(scope="module")
def trough():
    if not torch.cuda.is_available():
        pytest.skip("field baking targets CUDA")
    shoe, footbed, footbed_faces, centerline = _box_shoe()
    foot_bounds = np.asarray([[0.05, -0.32, -0.24], [0.95, -0.02, 0.24]])
    field = build_cavity_field(
        shoe,
        footbed,
        footbed_faces,
        centerline,
        foot_bounds,
        target_spacing=0.004,
        margin=0.03,
    )
    return field


def test_sign_convention_inside_is_positive(trough):
    """A point in the middle of the trough must report positive clearance."""

    points = torch.tensor([[[0.5, -0.15, 0.0]]], device="cuda")
    value = sample_field(trough.clearance, points, trough.lower, trough.upper)
    weight = sample_field(trough.valid, points, trough.lower, trough.upper)
    assert float(weight) > 0.99
    assert float(value) > 0.0, "interior point must be inside (positive)"


def test_clearance_matches_analytic_distances(trough):
    """min(distance to ceiling, distance to nearer wall), within a cell."""

    height, half_width, floor_y = 0.3, 0.2, 0.0
    ceiling = floor_y - height
    samples = [
        (0.5, -0.15, 0.00),
        (0.5, -0.05, 0.10),
        (0.3, -0.25, -0.05),
        (0.7, -0.10, 0.18),
        (0.2, -0.02, -0.19),
    ]
    points = torch.tensor([samples], dtype=torch.float32, device="cuda")
    value = sample_field(
        trough.clearance, points, trough.lower, trough.upper
    )[0, :, 0]
    expected = []
    for x, y, z in samples:
        expected.append(min(y - ceiling, half_width - abs(z)))
    expected_tensor = torch.tensor(expected, device="cuda")
    spacing = float(trough.spacing.max())
    error = (value - expected_tensor).abs().max()
    assert float(error) < spacing, (
        f"clearance error {float(error):.5f} exceeds one cell {spacing:.5f}; "
        f"got {value.tolist()} expected {expected}"
    )


def test_outside_wall_is_negative(trough):
    """Beyond a side wall the clearance must go negative."""

    points = torch.tensor([[[0.5, -0.15, 0.24]]], device="cuda")
    value = float(sample_field(trough.clearance, points, trough.lower, trough.upper))
    assert value < 0.0, f"point outside the wall reported {value:.5f}"


def test_above_ceiling_is_negative(trough):
    """Above the ceiling (smaller Y, since +Y is down) must go negative."""

    points = torch.tensor([[[0.5, -0.33, 0.0]]], device="cuda")
    value = float(sample_field(trough.clearance, points, trough.lower, trough.upper))
    assert value < 0.0, f"point above the ceiling reported {value:.5f}"


def test_axis_order_is_not_transposed(trough):
    """An asymmetric probe catches an X/Z swap in the grid_sample ordering."""

    # Inside in (x, z) = (0.5, 0.18) but outside if X and Z were swapped,
    # because x = 0.18 / z = 0.5 lies well beyond the half-width wall.
    inside = torch.tensor([[[0.5, -0.15, 0.18]]], device="cuda")
    swapped = torch.tensor([[[0.18, -0.15, 0.5]]], device="cuda")
    inside_value = float(
        sample_field(trough.clearance, inside, trough.lower, trough.upper)
    )
    swapped_value = float(
        sample_field(trough.clearance, swapped, trough.lower, trough.upper)
    )
    assert inside_value > 0.0
    assert swapped_value < inside_value


def test_gradient_points_toward_containment(trough):
    """The containment gradient must push an escaping vertex back inside."""

    point = torch.tensor(
        [[[0.5, -0.15, 0.23]]], device="cuda", requires_grad=True
    )
    value = sample_field(trough.clearance, point, trough.lower, trough.upper)
    penalty = torch.nn.functional.softplus(-value / 0.01).sum()
    penalty.backward()
    gradient = point.grad[0, 0]
    assert torch.isfinite(gradient).all()
    # Descending the penalty means stepping along -gradient; that step has to
    # reduce z, i.e. move back toward the centerline.
    assert float(-gradient[2]) < 0.0, (
        f"descent direction {(-gradient).tolist()} does not pull -Z inward"
    )
    ceiling_point = torch.tensor(
        [[[0.5, -0.32, 0.0]]], device="cuda", requires_grad=True
    )
    value = sample_field(
        trough.clearance, ceiling_point, trough.lower, trough.upper
    )
    torch.nn.functional.softplus(-value / 0.01).sum().backward()
    # +Y is down: escaping through the ceiling must be pushed back to larger Y.
    assert float(-ceiling_point.grad[0, 0, 1]) > 0.0


def test_open_space_is_marked_invalid():
    """A trough with no ceiling must report open space, not a fake boundary."""

    vertices: list = []
    faces: list = []

    def add(corners):
        offset = len(vertices)
        for corner in corners:
            vertices.append(corner)
        faces.append(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + offset)

    add([(0, 0.0, -0.2), (1, 0.0, -0.2), (1, 0.0, 0.2), (0, 0.0, 0.2)])
    add([(0, 0.0, -0.2), (1, 0.0, -0.2), (1, -0.3, -0.2), (0, -0.3, -0.2)])
    add([(0, 0.0, 0.2), (1, 0.0, 0.2), (1, -0.3, 0.2), (0, -0.3, 0.2)])
    shoe = TriangleMesh(np.asarray(vertices, dtype=np.float64),
                        np.concatenate(faces, axis=0))
    footbed_faces = np.asarray([0, 1], dtype=np.int64)
    footbed = TriangleMesh(
        shoe.vertices[shoe.faces[footbed_faces].reshape(-1)],
        np.arange(6, dtype=np.int64).reshape(2, 3),
    )
    centerline = np.stack((np.linspace(0.0, 1.0, 16), np.zeros(16)), axis=1)
    field = build_cavity_field(
        shoe,
        footbed,
        footbed_faces,
        centerline,
        np.asarray([[0.05, -0.25, -0.15], [0.95, -0.02, 0.15]]),
        target_spacing=0.004,
        margin=0.03,
    )
    # Directly over the open top there is no ceiling, so only the side walls
    # constrain; the combined field stays finite but must never claim the
    # vertical direction is bounded.
    points = torch.tensor([[[0.5, -0.15, 0.0]]], device="cuda")
    value = float(sample_field(field.clearance, points, field.lower, field.upper))
    assert value == pytest.approx(0.2, abs=0.01), (
        "with no ceiling the clearance must come from the walls alone"
    )


def test_escaped_vertex_is_pulled_back_into_the_domain(trough):
    """Outside the baked box, border padding alone gives no restoring force."""

    far = torch.tensor([[[0.5, -0.15, 0.9]]], device="cuda", requires_grad=True)
    plain = sample_field(trough.clearance, far, trough.lower, trough.upper)
    extended = sample_field(
        trough.clearance, far, trough.lower, trough.upper, extend_outside=True
    )
    assert float(extended) < float(plain), "extension must not be a no-op"
    assert float(extended) < 0.0, "a vertex far outside must read as outside"
    torch.nn.functional.softplus(-extended / 0.01).sum().backward()
    assert float(-far.grad[0, 0, 2]) < 0.0, "descent must pull +Z back inward"


def test_out_of_domain_is_detected(trough):
    points = torch.tensor(
        [[[0.5, -0.15, 0.0], [5.0, -0.15, 0.0]]], device="cuda"
    )
    mask = inside_domain(points, trough.lower, trough.upper)
    assert bool(mask[0, 0]) and not bool(mask[0, 1])


def test_footbed_height_sampling(trough):
    points = torch.tensor([[[0.5, 0.0], [0.25, 0.1]]], device="cuda")
    height, weight = sample_footbed_height(trough, points)
    assert torch.all(weight > 0.9)
    assert torch.allclose(height, torch.zeros_like(height), atol=2e-3)


def test_batched_sampling_matches_single(trough):
    points = torch.tensor(
        [[[0.5, -0.15, 0.0], [0.3, -0.10, 0.05]]], device="cuda"
    )
    single = sample_field(trough.clearance, points, trough.lower, trough.upper)
    batched = sample_field(
        trough.clearance, points.repeat(4, 1, 1), trough.lower, trough.upper
    )
    for item in range(4):
        assert torch.allclose(batched[item], single[0], atol=1e-6)


def test_nearest_surface_wins_with_two_ceilings():
    """A single-layer box cannot distinguish nearest from farthest hits.

    This trough has two ceilings stacked above the floor. The clearance must be
    measured to the *lower* one (the first surface the upward ray meets).
    """

    vertices: list = []
    faces: list = []

    def add(corners):
        offset = len(vertices)
        vertices.extend(corners)
        faces.append(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + offset)

    half_width = 0.2
    add([(0, 0.0, -half_width), (1, 0.0, -half_width),
         (1, 0.0, half_width), (0, 0.0, half_width)])          # footbed
    add([(0, -0.15, -half_width), (1, -0.15, -half_width),
         (1, -0.15, half_width), (0, -0.15, half_width)])      # near ceiling
    add([(0, -0.30, -half_width), (1, -0.30, -half_width),
         (1, -0.30, half_width), (0, -0.30, half_width)])      # far ceiling
    for side in (-half_width, half_width):
        add([(0, 0.0, side), (1, 0.0, side), (1, -0.30, side), (0, -0.30, side)])

    shoe = TriangleMesh(np.asarray(vertices, dtype=np.float64),
                        np.concatenate(faces, axis=0))
    footbed_faces = np.asarray([0, 1], dtype=np.int64)
    footbed = TriangleMesh(
        shoe.vertices[shoe.faces[footbed_faces].reshape(-1)],
        np.arange(6, dtype=np.int64).reshape(2, 3),
    )
    centerline = np.stack((np.linspace(0.0, 1.0, 16), np.zeros(16)), axis=1)
    field = build_cavity_field(
        shoe, footbed, footbed_faces, centerline,
        np.asarray([[0.05, -0.28, -0.15], [0.95, -0.02, 0.15]]),
        target_spacing=0.004, margin=0.02,
    )
    # y = -0.05 is 0.10 below the near ceiling and 0.25 below the far one.
    points = torch.tensor([[[0.5, -0.05, 0.0]]], device="cuda")
    value = float(sample_field(field.clearance, points, field.lower, field.upper))
    assert value == pytest.approx(0.10, abs=0.01), (
        f"expected clearance to the nearest ceiling (0.10), got {value:.4f}"
    )


def test_distance_channel_sees_the_longitudinal_wall():
    """The directional rays never probe X; the distance channel must.

    A closed front wall is invisible to the ceiling and side-wall rays, so a
    point pressed against it reports healthy directional clearance. This is the
    exact blind spot that let the fitted foot slide out through the toe box.
    """

    vertices: list = []
    faces: list = []

    def add(corners):
        offset = len(vertices)
        vertices.extend(corners)
        faces.append(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + offset)

    half_width, length, top = 0.2, 1.0, -0.3
    add([(0, 0.0, -half_width), (length, 0.0, -half_width),
         (length, 0.0, half_width), (0, 0.0, half_width)])           # footbed
    add([(0, top, -half_width), (length, top, -half_width),
         (length, top, half_width), (0, top, half_width)])           # ceiling
    for side in (-half_width, half_width):
        add([(0, 0.0, side), (length, 0.0, side), (length, top, side), (0, top, side)])
    add([(length, 0.0, -half_width), (length, 0.0, half_width),
         (length, top, half_width), (length, top, -half_width)])     # front wall

    shoe = TriangleMesh(np.asarray(vertices, dtype=np.float64),
                        np.concatenate(faces, axis=0))
    footbed_faces = np.asarray([0, 1], dtype=np.int64)
    footbed = TriangleMesh(
        shoe.vertices[shoe.faces[footbed_faces].reshape(-1)],
        np.arange(6, dtype=np.int64).reshape(2, 3),
    )
    centerline = np.stack((np.linspace(0.0, length, 16), np.zeros(16)), axis=1)
    field = build_cavity_field(
        shoe, footbed, footbed_faces, centerline,
        np.asarray([[0.05, -0.28, -0.15], [0.98, -0.02, 0.15]]),
        target_spacing=0.004, margin=0.02,
    )
    # 0.01 short of the front wall, but 0.15 from the ceiling and 0.2 from
    # either side: the directional field must be fooled, the distance must not.
    point = torch.tensor([[[0.99, -0.15, 0.0]]], device="cuda")
    directional = float(
        sample_field(field.clearance, point, field.lower, field.upper)
    )
    distance = float(
        sample_field(field.distance, point, field.lower, field.upper)
    )
    assert directional > 0.10, (
        "the directional field is expected to miss a longitudinal wall"
    )
    assert distance == pytest.approx(0.01, abs=0.005), (
        f"distance channel should see the front wall, got {distance:.4f}"
    )
    point = point.clone().requires_grad_(True)
    value = sample_field(
        field.distance, point, field.lower, field.upper, extend_outside=True
    )
    torch.nn.functional.softplus(-value / 0.005).sum().backward()
    assert float(-point.grad[0, 0, 0]) < 0.0, "descent must push -X off the wall"
