"""Material measurement against geometry whose answer is known by hand.

The fibers are straight vertical lines and the "footwear" is a stack of flat
plates at chosen heights, so every crossing distance is a number written into
the test rather than one produced by the code being tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from anatomical_coordinates.coordinate_mapping.material import (
    MILLIMETRES,
    base_field,
    crossings,
    subdivide_large,
)


def _plate(height: float, half: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A square of two triangles, flat at ``height``, centred on the origin."""

    corners = np.array([
        [-half, height, -half], [half, height, -half],
        [half, height, half], [-half, height, half],
    ])
    return corners, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _stack(heights) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, offset = [], [], 0
    for height in heights:
        corner, face = _plate(height)
        vertices.append(corner)
        faces.append(face + offset)
        offset += len(corner)
    return np.concatenate(vertices), np.concatenate(faces)


def _fibers(count: int, top: float, samples: int = 64, seed: int = 0):
    """Straight vertical fibers, with arclength equal to height."""

    generator = np.random.default_rng(seed)
    base = generator.uniform(-1.0, 1.0, size=(count, 2))
    height = np.linspace(0.0, top, samples)
    curves = np.empty((count, samples, 3))
    curves[:, :, 0] = base[:, 0:1]
    curves[:, :, 2] = base[:, 1:2]
    curves[:, :, 1] = height[None, :]
    return curves, np.broadcast_to(height, (count, samples)).copy()


def test_subdivision_preserves_the_surface():
    vertices, faces = _plate(0.0, half=4.0)

    def area(v, f):
        t = v[f]
        return 0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1
        ).sum()

    before = area(vertices, faces)
    fine_v, fine_f = subdivide_large(vertices, faces, max_radius=0.5)
    assert np.isclose(area(fine_v, fine_f), before)
    corners = fine_v[fine_f]
    centre = corners.mean(axis=1)
    radius = np.linalg.norm(corners - centre[:, None, :], axis=-1).max(axis=1)
    assert radius.max() <= 0.5 + 1e-12
    assert len(fine_f) > len(faces)


def test_subdivision_leaves_small_triangles_alone():
    vertices, faces = _plate(0.0, half=0.1)
    same_v, same_f = subdivide_large(vertices, faces, max_radius=1.0)
    assert len(same_f) == len(faces)
    assert np.array_equal(same_f, faces)


def test_a_fiber_through_two_plates_reports_both():
    vertices, faces = _stack([0.3, 0.5])
    curves, length = _fibers(24, top=1.0)
    site, at_r, at_mm, report = crossings(
        curves, length, vertices, faces, max_triangle_radius=0.5
    )
    assert report["saturated_segments"] == 0
    for index in range(24):
        mine = at_r[site == index]
        assert len(mine) == 2
        assert np.allclose(np.sort(mine), [0.3, 0.5], atol=2e-2)
    assert np.allclose(at_mm, at_r * MILLIMETRES, rtol=1e-9)


def test_a_fiber_that_misses_reports_nothing():
    vertices, faces = _stack([0.4])
    vertices = vertices + np.array([50.0, 0.0, 0.0])   # move the plate away
    curves, length = _fibers(8, top=1.0)
    site, at_r, _, _ = crossings(
        curves, length, vertices, faces, max_triangle_radius=0.5
    )
    assert site.size == 0


def test_three_plates_give_three_crossings():
    """Absence and multiplicity are structure, so the middle is kept."""

    vertices, faces = _stack([0.2, 0.45, 0.8])
    curves, length = _fibers(12, top=1.0)
    site, at_r, _, _ = crossings(
        curves, length, vertices, faces, max_triangle_radius=0.5
    )
    for index in range(12):
        mine = np.sort(at_r[site == index])
        assert len(mine) == 3
        assert np.allclose(mine, [0.2, 0.45, 0.8], atol=2e-2)


def test_a_coarse_plate_is_still_found():
    """The failure this subdivision exists to prevent.

    One very wide triangle has its centroid far from where the fiber crosses
    it. Without subdivision a nearest-centroid search can miss it entirely.
    """

    vertices, faces = _plate(0.5, half=40.0)
    curves, length = _fibers(16, top=1.0)
    site, at_r, _, _ = crossings(
        curves, length, vertices, faces, max_triangle_radius=0.5
    )
    assert len(np.unique(site)) == 16
    assert np.allclose(at_r, 0.5, atol=2e-2)


def test_base_field_is_negative_only_inside_material():
    delta_in = np.array([0.2, 0.2, 0.2, 0.2])
    delta_out = np.array([0.5, 0.5, 0.5, 0.5])
    covered = np.array([-1.0, -1.0, -1.0, 1.0])   # the last one is uncovered
    r = np.array([0.1, 0.35, 0.7, 0.35])
    value = base_field(covered, delta_in, delta_out, r)
    assert value[0] > 0      # short of the material
    assert value[1] < 0      # inside it
    assert value[2] > 0      # beyond it
    assert value[3] > 0      # nothing here at all, whatever r says


def test_base_field_boundary_is_zero_at_both_faces():
    value = base_field(
        np.array([-1.0, -1.0]), np.array([0.2, 0.2]),
        np.array([0.5, 0.5]), np.array([0.2, 0.5]),
    )
    assert np.allclose(value, 0.0)
