"""The longitudinal sign fix and the geometry-keyed collar exemption.

Two defects let the fitted foot reverse out through the heel counter at no
cost, and they compound:

  1. ``outside`` was derived from the +/-Y and +/-Z ray families alone. Behind
     the heel counter neither family finds a boundary, so the true distance was
     signed *positive* and grew with depth: the field told the optimizer that
     leaving the shoe backwards took it deeper inside.
  2. The containment exemption was an axis-aligned box in joint space that
     covered the whole rear-upper quadrant, heel included, and dropped the sign
     there.

Everything below is synthetic geometry with an analytic answer, so a failure
points at the field or the mask rather than at a shoe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anatomical_coordinates.losses import containment_loss
from anatomical_coordinates.cavity_field import (
    build_cavity_field,
    sample_field,
    sample_open_above,
)
from foot_prior.mesh import TriangleMesh


def _build(quads: list, footbed_count: int = 1):
    """Assemble a mesh from quads; the first ``footbed_count`` are the footbed."""

    vertices: list = []
    faces: list = []
    for corners in quads:
        offset = len(vertices)
        vertices.extend(corners)
        faces.append(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + offset)
    shoe = TriangleMesh(
        np.asarray(vertices, dtype=np.float64), np.concatenate(faces, axis=0)
    )
    footbed_faces = np.arange(2 * footbed_count, dtype=np.int64)
    footbed = TriangleMesh(
        shoe.vertices[shoe.faces[footbed_faces].reshape(-1)],
        np.arange(6 * footbed_count, dtype=np.int64).reshape(-1, 3),
    )
    return shoe, footbed, footbed_faces


def _closed_box(half_width: float = 0.2, top: float = -0.3, length: float = 1.0):
    """A shoe closed at both ends: heel counter at x=0, toe box at x=length."""

    return _build([
        # footbed floor, first so its face indices are 0 and 1
        [(0, 0.0, -half_width), (length, 0.0, -half_width),
         (length, 0.0, half_width), (0, 0.0, half_width)],
        # ceiling
        [(0, top, -half_width), (length, top, -half_width),
         (length, top, half_width), (0, top, half_width)],
        # side walls
        [(0, 0.0, -half_width), (length, 0.0, -half_width),
         (length, top, -half_width), (0, top, -half_width)],
        [(0, 0.0, half_width), (length, 0.0, half_width),
         (length, top, half_width), (0, top, half_width)],
        # heel counter (x = 0) and toe box (x = length)
        [(0, 0.0, -half_width), (0, 0.0, half_width),
         (0, top, half_width), (0, top, -half_width)],
        [(length, 0.0, -half_width), (length, 0.0, half_width),
         (length, top, half_width), (length, top, -half_width)],
    ])


@pytest.fixture(scope="module")
def closed():
    if not torch.cuda.is_available():
        pytest.skip("field baking targets CUDA")
    shoe, footbed, footbed_faces = _closed_box()
    centerline = np.stack((np.linspace(0.0, 1.0, 16), np.zeros(16)), axis=1)
    return build_cavity_field(
        shoe, footbed, footbed_faces, centerline,
        # grown past both ends so the probes stay inside the domain, and wide
        # enough in Z that the side walls are actually voxelized - the distance
        # channel can only see geometry inside the baked box.
        np.asarray([[-0.06, -0.28, -0.18], [1.06, -0.02, 0.18]]),
        target_spacing=0.004, margin=0.03,
    )


def _distance(field, point):
    tensor = torch.tensor([[point]], dtype=torch.float32, device="cuda")
    return float(
        sample_field(field.distance, tensor, field.lower, field.upper)
    )


def test_behind_the_heel_counter_is_signed_outside(closed):
    """The defect this fix exists for: 0.05 behind the counter must be -0.05.

    Before the longitudinal ray family, no ray reached this sample, so it was
    signed positive and the value *increased* the further out it went.
    """

    value = _distance(closed, (-0.05, -0.15, 0.0))
    assert value < 0.0, f"behind the heel counter must be outside, got {value:.4f}"
    assert value == pytest.approx(-0.05, abs=0.01)


def test_outside_sign_deepens_with_depth_behind_the_heel(closed):
    """Monotone in depth, so there is a restoring force rather than a cliff."""

    near = _distance(closed, (-0.02, -0.15, 0.0))
    far = _distance(closed, (-0.08, -0.15, 0.0))
    assert far < near < 0.0, f"expected {far:.4f} < {near:.4f} < 0"


def test_beyond_the_toe_box_is_signed_outside(closed):
    value = _distance(closed, (1.05, -0.15, 0.0))
    assert value < 0.0, f"beyond the toe box must be outside, got {value:.4f}"
    assert value == pytest.approx(-0.05, abs=0.01)


def test_interior_points_keep_the_correct_positive_sign(closed):
    """The fix must not flip anything that was already right."""

    for point, expected in (
        ((0.5, -0.15, 0.0), 0.15),   # dead centre, nearest surface is a ceiling
        # just above the footbed: the footbed is not an obstacle, so the
        # nearest obstacle is a side wall at 0.20, not the floor
        ((0.5, -0.05, 0.0), 0.20),
        ((0.1, -0.15, 0.0), 0.10),   # inside, near the heel counter
        ((0.9, -0.15, 0.0), 0.10),   # inside, near the toe box
        ((0.5, -0.15, 0.15), 0.05),  # inside, near a side wall
    ):
        value = _distance(closed, point)
        assert value > 0.0, f"{point} is inside but was signed {value:.4f}"
        assert value == pytest.approx(expected, abs=0.01), (
            f"{point}: expected {expected}, got {value:.4f}"
        )


def test_restoring_gradient_points_back_into_the_shoe(closed):
    """Descending the barrier behind the counter must move the sample +X."""

    point = torch.tensor(
        [[(-0.05, -0.15, 0.0)]], dtype=torch.float32, device="cuda"
    ).requires_grad_(True)
    value = sample_field(
        closed.distance, point, closed.lower, closed.upper, extend_outside=True
    )
    torch.nn.functional.softplus(-value / 0.005).sum().backward()
    assert float(-point.grad[0, 0, 0]) > 0.0, (
        "descent must push the sample forward, back inside the shoe"
    )


# --------------------------------------------------------------------------
# The collar exemption


@pytest.fixture(scope="module")
def collared():
    """Closed at both ends, with a real hole in the upper over the rear half."""

    if not torch.cuda.is_available():
        pytest.skip("field baking targets CUDA")
    half_width, top, length, gap = 0.2, -0.3, 1.0, 0.4
    shoe, footbed, footbed_faces = _build([
        [(0, 0.0, -half_width), (length, 0.0, -half_width),
         (length, 0.0, half_width), (0, 0.0, half_width)],           # footbed
        # ceiling over the FRONT only: x in [gap, length]. x < gap is open.
        [(gap, top, -half_width), (length, top, -half_width),
         (length, top, half_width), (gap, top, half_width)],
        [(0, 0.0, -half_width), (length, 0.0, -half_width),
         (length, top, -half_width), (0, top, -half_width)],
        [(0, 0.0, half_width), (length, 0.0, half_width),
         (length, top, half_width), (0, top, half_width)],
        [(0, 0.0, -half_width), (0, 0.0, half_width),
         (0, top, half_width), (0, top, -half_width)],               # counter
        [(length, 0.0, -half_width), (length, 0.0, half_width),
         (length, top, half_width), (length, top, -half_width)],     # toe box
    ])
    centerline = np.stack((np.linspace(0.0, length, 16), np.zeros(16)), axis=1)
    return build_cavity_field(
        shoe, footbed, footbed_faces, centerline,
        np.asarray([[-0.06, -0.40, -0.18], [1.06, -0.02, 0.18]]),
        target_spacing=0.004, margin=0.03,
    )


def _exemption(field, point):
    tensor = torch.tensor([[point]], dtype=torch.float32, device="cuda")
    return float(sample_open_above(field, tensor[:, :, (0, 2)]))


def test_leg_through_the_opening_is_exempt(collared):
    """A sample in the collar throat sits in a footbed column with no ceiling."""

    assert _exemption(collared, (0.20, -0.35, 0.0)) > 0.99
    assert _exemption(collared, (0.20, -0.15, 0.0)) > 0.99


def test_heel_behind_the_counter_is_never_exempt(collared):
    """The whole point of the change: no footbed behind the counter, no pass.

    The old joint-space box exempted this region outright, which is where
    64-86 % of heel containment violations were sitting.
    """

    for depth in (0.01, 0.02, 0.05, 0.08):
        weight = _exemption(collared, (-depth, -0.15, 0.0))
        assert weight < 0.01, (
            f"{depth:.2f} behind the counter was exempt at {weight:.3f}"
        )
    # and above the counter's top edge, level with the collar opening
    assert _exemption(collared, (-0.05, -0.35, 0.0)) < 0.01


def test_covered_upper_is_never_exempt(collared):
    """Under the closed part of the upper there is a ceiling, so no exemption."""

    assert _exemption(collared, (0.80, -0.15, 0.0)) < 0.01
    assert _exemption(collared, (1.05, -0.15, 0.0)) < 0.01


def test_exempt_mask_keeps_the_heel_violation_signed(collared):
    """End to end through the real loss: the mask must not launder the heel.

    ``containment_loss`` blends the exempt samples onto the unsigned distance.
    The collar sample may come out non-negative; the heel sample behind the
    counter may not, however the mask is sampled.
    """

    samples = torch.tensor(
        [[(0.20, -0.35, 0.0), (-0.05, -0.15, 0.0), (0.5, -0.15, 0.0)]],
        dtype=torch.float32, device="cuda",
    )
    mask = sample_open_above(collared, samples[:, :, (0, 2)])
    _, clearance = containment_loss(
        samples, collared, margin_mm=1.0, softness_mm=0.5, scale_mm=1.0,
        exempt_mask=mask,
    )
    collar, heel, interior = (float(value) for value in clearance[0])
    assert collar >= 0.0, f"the leg through the opening was charged: {collar:.4f}"
    assert heel < 0.0, f"the heel behind the counter was laundered: {heel:.4f}"
    assert interior > 0.0
