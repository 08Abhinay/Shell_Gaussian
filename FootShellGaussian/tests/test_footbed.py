"""Tests for topology-independent footbed identification and sampling."""

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
NORMAL_SHOES = (
    "aj_12_basketball_sneakers",
    "birkenstock_arizona_sandal",
    "canvas_shoe",
    "crocs",
    "crocs_by_speedyart_studio",
    "crocs_shoe",
    "duinn_shoes_womens_hiking_sandal_sport",
    "nike_air_jordan",
    "pb129_shoe_low",
    "priest_karol_wojtyas_sports_shoes",
    "sandal_1",
    "sandals_0001",
    "shoes_mockup_asset_vans_skate_old_skool_shoes",
    "sneaker_vibe",
    "sneakers_seen",
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
            [
                [-5.0, -1.0, -2.0],
                [5.0, -1.0, -2.0],
                [5.0, 1.0, -2.0],
                [-5.0, 1.0, -2.0],
            ]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def connected_footbed_and_sidewall() -> TriangleMesh:
    return TriangleMesh(
        np.asarray(
            [
                [-5.0, 0.0, -2.0],
                [5.0, 0.0, -2.0],
                [5.0, 0.0, 2.0],
                [-5.0, 0.0, 2.0],
                [5.0, -1.0, 2.0],
                [-5.0, -1.0, 2.0],
            ]
        ),
        np.asarray(
            [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]], dtype=np.int64
        ),
    )


def ring_sheet() -> TriangleMesh:
    x_values = (-5.0, -1.0, 1.0, 5.0)
    z_values = (-2.0, -0.4, 0.4, 2.0)
    vertices = np.asarray([[x, 0.0, z] for x in x_values for z in z_values])
    faces: list[list[int]] = []
    for x_index in range(3):
        for z_index in range(3):
            if (x_index, z_index) == (1, 1):
                continue
            first = x_index * 4 + z_index
            second = (x_index + 1) * 4 + z_index
            faces.extend(
                [[first, second, second + 1], [first, second + 1, first + 1]]
            )
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64))


@pytest.mark.parametrize("reversed_winding", [False, True])
def test_selects_inner_sheet_independently_of_face_winding(
    reversed_winding: bool,
) -> None:
    upper = sheet((-5.0, 5.0), (-2.0, 2.0), (-2.0,) * 4)
    footbed = sheet(
        (-5.0, 5.0),
        (-2.0, 2.0),
        (0.0,) * 4,
        upward=not reversed_winding,
    )
    outsole = sheet(
        (-5.0, 5.0),
        (-2.0, 2.0),
        (0.5,) * 4,
        upward=not reversed_winding,
    )
    result = identify_footbed_surface(combine(upper, outsole, footbed))
    assert result.original_face_indices.tolist() == [4, 5]
    assert result.area_weighted_median_y == pytest.approx(0.0)
    assert result.length_coverage == pytest.approx(1.0)
    assert result.width_coverage == pytest.approx(1.0)


def test_extracts_footbed_faces_when_connected_to_sidewall() -> None:
    result = identify_footbed_surface(connected_footbed_and_sidewall())
    assert result.original_face_indices.tolist() == [0, 1]
    assert result.mesh.faces.shape == (2, 3)
    assert result.support_like_area_fraction == pytest.approx(1.0)


def test_rejects_narrow_and_vertical_surfaces() -> None:
    narrow = sheet((-2.0, 2.0), (-2.0, 2.0), (0.0,) * 4)
    mesh = combine(narrow, vertical_component())
    with pytest.raises(ValueError, match="layers=") as error:
        identify_footbed_surface(mesh)
    message = str(error.value)
    assert '"length_coverage"' in message
    assert '"footprint_fill_fraction"' in message


def test_height_interpolation_inside_and_outside_projection() -> None:
    # The four corners lie on y = 0.03*x + 0.04*z + 0.1.
    surface = sheet((-1.0, 1.0), (-1.0, 1.0), (0.03, 0.09, 0.17, 0.11))
    footbed = identify_footbed_surface(surface)
    points = np.asarray([[0.0, 0.0], [0.5, -0.5], [2.0, 0.0]])
    heights, valid = sample_footbed_y(footbed, points)
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(heights[:2], [0.1, 0.095], atol=1e-12)
    assert np.isnan(heights[2])


def test_preserves_real_hole_in_selected_surface() -> None:
    footbed = identify_footbed_surface(ring_sheet())
    heights, valid = sample_footbed_y(
        footbed, np.asarray([[-3.0, 0.0], [0.0, 0.0]])
    )
    np.testing.assert_array_equal(valid, [True, False])
    assert heights[0] == pytest.approx(0.0)
    assert np.isnan(heights[1])


def test_known_canvas_component_and_sampling() -> None:
    path = evaluation_root() / "canvas_shoe/reference_mesh.ply"
    if not path.is_file():
        pytest.skip(f"external canvas dataset is unavailable: {path}")
    footbed = identify_footbed_surface(load_triangle_mesh(path))
    assert footbed.original_face_indices[0] == 8090
    assert footbed.original_face_indices[-1] == 8559
    assert footbed.mesh.vertices.shape == (207, 3)
    assert footbed.mesh.faces.shape == (350, 3)
    np.testing.assert_allclose(
        footbed.bounds,
        [
            [-0.10368063, 0.01983586, -0.03464793],
            [0.10016327, 0.02972209, 0.03431007],
        ],
        atol=1e-8,
    )
    assert footbed.height_grid.shape == (256, 96)
    assert footbed.valid_mask.shape == footbed.height_grid.shape
    assert np.count_nonzero(footbed.valid_mask) == 15922
    assert footbed.length_coverage == pytest.approx(0.94921875)
    assert footbed.width_coverage == pytest.approx(0.864583333333331)
    assert footbed.upward_facing_area_fraction == pytest.approx(1.0)
    assert footbed.support_like_area_fraction == pytest.approx(1.0)
    assert footbed.area_weighted_median_y == pytest.approx(0.028700418, abs=1e-9)
    heights, valid = sample_footbed_y(
        footbed, np.asarray([[0.0, 0.0], [0.2, 0.0]])
    )
    np.testing.assert_array_equal(valid, [True, False])
    assert np.isfinite(heights[0])
    assert np.isnan(heights[1])


@pytest.mark.parametrize("shoe_name", NORMAL_SHOES)
def test_all_normal_shoes_select_opening_facing_support(shoe_name: str) -> None:
    path = evaluation_root() / shoe_name / "reference_mesh.ply"
    if not path.is_file():
        pytest.skip(f"external evaluation shoe is unavailable: {path}")
    first = identify_footbed_surface(load_triangle_mesh(path))
    second = identify_footbed_surface(load_triangle_mesh(path))
    np.testing.assert_array_equal(
        first.original_face_indices, second.original_face_indices
    )
    assert first.length_coverage >= 0.65
    assert first.width_coverage >= 0.40
    assert first.upward_facing_area_fraction >= 0.90
    assert first.support_like_area_fraction == pytest.approx(1.0)
    assert np.count_nonzero(first.valid_mask) > 0
