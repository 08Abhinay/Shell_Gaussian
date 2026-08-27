"""Tests for deterministic footbed identification and sampling."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from foot_prior.footbed import identify_footbed_surface, sample_footbed_y
from foot_prior.mesh import TriangleMesh, load_triangle_mesh


DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right"
)


def evaluation_root() -> Path:
    return Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT))


def sheet(
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    y_values: tuple[float, float, float, float],
    upward: bool = True,
) -> TriangleMesh:
    x0, x1 = x_bounds
    z0, z1 = z_bounds
    vertices = np.asarray(
        [
            [x0, y_values[0], z0],
            [x1, y_values[1], z0],
            [x1, y_values[2], z1],
            [x0, y_values[3], z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    if not upward:
        faces = faces[:, ::-1]
    return TriangleMesh(vertices, faces)


def combine(*meshes: TriangleMesh) -> TriangleMesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return TriangleMesh(np.concatenate(vertices), np.concatenate(faces))


def vertical_component() -> TriangleMesh:
    return TriangleMesh(
        np.asarray(
            [[-5.0, -1.0, -2.0], [5.0, -1.0, -2.0], [5.0, 1.0, -2.0], [-5.0, 1.0, -2.0]]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def test_selects_topmost_of_two_qualifying_sheets() -> None:
    top = sheet((-5.0, 5.0), (-2.0, 2.0), (0.0, 0.0, 0.0, 0.0))
    bottom = sheet((-5.0, 5.0), (-2.0, 2.0), (0.3, 0.3, 0.3, 0.3))
    result = identify_footbed_surface(combine(bottom, top))
    assert result.original_face_indices.tolist() == [2, 3]
    assert result.area_weighted_median_y == pytest.approx(0.0)
    assert result.length_coverage == pytest.approx(1.0)
    assert result.width_coverage == pytest.approx(1.0)


def test_rejects_narrow_vertical_and_downward_components() -> None:
    narrow = sheet((-2.0, 2.0), (-2.0, 2.0), (0.0, 0.0, 0.0, 0.0))
    downward = sheet((-5.0, 5.0), (-2.0, 2.0), (0.5, 0.5, 0.5, 0.5), upward=False)
    mesh = combine(narrow, vertical_component(), downward)
    with pytest.raises(ValueError, match="candidates=") as error:
        identify_footbed_surface(mesh)
    message = str(error.value)
    assert '"length_coverage"' in message
    assert '"upward_facing_area_fraction"' in message


def test_height_interpolation_inside_and_outside_projection() -> None:
    # The four corners lie on y = 0.1*x + 0.2*z + 0.3.
    surface = sheet((-1.0, 1.0), (-1.0, 1.0), (0.0, 0.2, 0.6, 0.4))
    footbed = identify_footbed_surface(surface)
    points = np.asarray([[0.0, 0.0], [0.5, -0.5], [2.0, 0.0]])
    heights, valid = sample_footbed_y(footbed, points)
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(heights[:2], [0.3, 0.25], atol=1e-12)
    assert np.isnan(heights[2])


def test_known_canvas_component_and_sampling() -> None:
    path = evaluation_root() / "canvas_shoe/reference_mesh.ply"
    if not path.is_file():
        pytest.skip(f"external canvas dataset is unavailable: {path}")
    footbed = identify_footbed_surface(load_triangle_mesh(path))
    assert len(footbed.diagnostics) == 35
    assert footbed.original_face_indices[0] == 8090
    assert footbed.original_face_indices[-1] == 8559
    assert footbed.mesh.vertices.shape == (207, 3)
    assert footbed.mesh.faces.shape == (350, 3)
    np.testing.assert_allclose(
        footbed.bounds,
        [[-0.10368063, 0.01983586, -0.03464793], [0.10016327, 0.02972209, 0.03431007]],
        atol=1e-8,
    )
    assert footbed.length_coverage == pytest.approx(0.949945, abs=1e-6)
    assert footbed.width_coverage == pytest.approx(0.864212, abs=1e-6)
    assert footbed.upward_facing_area_fraction == pytest.approx(1.0)
    assert footbed.area_weighted_median_y == pytest.approx(0.028700418, abs=1e-9)
    heights, valid = sample_footbed_y(footbed, np.asarray([[0.0, 0.0], [0.2, 0.0]]))
    np.testing.assert_array_equal(valid, [True, False])
    assert np.isfinite(heights[0])
    assert np.isnan(heights[1])
