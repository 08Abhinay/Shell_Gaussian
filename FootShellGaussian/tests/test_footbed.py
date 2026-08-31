"""Tests for topology-independent footbed identification and sampling."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from foot_prior.footbed import identify_footbed_surface, sample_footbed_y
from foot_prior.mesh import TriangleMesh, load_triangle_mesh
from foot_prior.normalization import build_shoe_normalization


DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right"
)
DEFAULT_CORRECTED_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_correct_orientation"
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
EXPECTED_SUPPORT_FACE_COUNTS = {
    "aj_12_basketball_sneakers": 2252,
    "birkenstock_arizona_sandal": 8937,
    "canvas_shoe": 350,
    "crocs": 974,
    "crocs_by_speedyart_studio": 11017,
    "crocs_shoe": 80790,
    "duinn_shoes_womens_hiking_sandal_sport": 4812,
    "nike_air_jordan": 286,
    "pb129_shoe_low": 713,
    "priest_karol_wojtyas_sports_shoes": 2340,
    "sandal_1": 14023,
    "sandals_0001": 5909,
    "shoes_mockup_asset_vans_skate_old_skool_shoes": 602,
    "sneaker_vibe": 390,
    "sneakers_seen": 240,
}
CORRECTED_SUPPORT_FACE_DIGESTS = {
    "aj_12_basketball_sneakers": (
        "73c1a29ce794759a576a72be2ddccd9fc5c15195db7511d889cb32c3383d92bd"
    ),
    "birkenstock_arizona_sandal": (
        "0a621bfe9df5252527cfbd94cd87a7d0f291a3a78d94e32d203c73a8759dee76"
    ),
    "canvas_shoe": (
        "85b7229c82defdcf2d7a417a50b914b5ffdbb29c9233bfd3f332444af9e38803"
    ),
    "crocs": (
        "a68699182c39d494eef2112a047373b5e92071bdad6b448b4f8708e5be567e8c"
    ),
    "crocs_by_speedyart_studio": (
        "d4c4d2ed2f2052b93a07c3944d1629dcb84193b188400f9042eaa172d245fe00"
    ),
    "crocs_shoe": (
        "10ca8464700abb8fe2c95e5b96d579487b0cbf1b2812cef25c0e13ff13ea65ed"
    ),
    "duinn_shoes_womens_hiking_sandal_sport": (
        "7c7ea9c01aa55ba13c898fda70abcc6f2098ede8626759f68a4435de5efef5ff"
    ),
    "leather_boots": (
        "f1364c7a4761a2d9ab5e5a29a2a98adf02c910d83dfba50a84f731b9d4e18138"
    ),
    "nike_air_jordan": (
        "98d306f0deb807c355e9e446a3c434c324fae6c22fe10514558d6f3aba159c75"
    ),
    "pb129_shoe_low": (
        "e477afecae101e2af221ba8bdd473c3728a5bc86dd2beeca2e24cdb8accb5964"
    ),
    "priest_karol_wojtyas_sports_shoes": (
        "7934b15d57d946a5d881ca6b188edb8fc48294c8e4297ff01afc1e20d5c5ec72"
    ),
    "sandal_1": (
        "af1f55fcfdcbc0868be5e05387fb8e53929ad35ce945e8b5c50c80b6b04800b7"
    ),
    "sandals_0001": (
        "21f4e401c80b42a1d95dfd7dd3615af10aeff66b5151c1495378afec8782e696"
    ),
    "shoes_mockup_asset_vans_skate_old_skool_shoes": (
        "9f3f4052e14cb0e38689e2080fd2d10cf09b52fae5d3eeb3bfe5efed881cd024"
    ),
    "sneaker_vibe": (
        "d6437ea39711c75c2e7369fa1155b19e5973f4f82b30cd8bf47a8fedb1abbe45"
    ),
    "sneakers_seen": (
        "6a08d6bf87e987b8e2f83d135c839c90039d9c81e63f9e09daadf173b9c2bca8"
    ),
    "ww_ii_german_jack_boots": (
        "9ba2e26fc61cf7a9f9b7cef7ea09cb77637d9d3e2b1f1f3fa4c63be77156d430"
    ),
}


def evaluation_root() -> Path:
    return Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT))


def corrected_evaluation_root() -> Path:
    return Path(
        os.environ.get(
            "FOOTSHELL_CORRECTED_EVAL_ROOT", DEFAULT_CORRECTED_EVAL_ROOT
        )
    )


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


def u_shaped_upper(y: float, upward: bool = True) -> TriangleMesh:
    """Build a broad surface whose middle is supported only near the toe."""

    x_values = (-5.0, 3.0, 5.0)
    z_values = (-2.0, -1.0, 1.0, 2.0)
    vertices = np.asarray([[x, y, z] for x in x_values for z in z_values])
    faces: list[list[int]] = []
    for x_index in range(2):
        for z_index in range(3):
            if (x_index, z_index) == (0, 1):
                continue
            first = x_index * 4 + z_index
            second = (x_index + 1) * 4 + z_index
            faces.extend(
                [[first, second, second + 1], [first, second + 1, first + 1]]
            )
    face_array = np.asarray(faces, dtype=np.int64)
    if not upward:
        face_array = face_array[:, ::-1]
    return TriangleMesh(vertices, face_array)


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
    assert result.selection_method == "component_layers"


@pytest.mark.parametrize("reversed_winding", [False, True])
def test_rejects_broad_upper_without_central_support(
    reversed_winding: bool,
) -> None:
    upward = not reversed_winding
    height_marker = sheet((-1.0, 1.0), (-0.2, 0.2), (-1.0,) * 4, upward)
    upper = u_shaped_upper(0.0, upward)
    footbed = sheet((-5.0, 5.0), (-2.0, 2.0), (0.5,) * 4, upward)
    outsole = sheet((-5.0, 5.0), (-2.0, 2.0), (1.0,) * 4, upward)

    result = identify_footbed_surface(
        combine(height_marker, upper, footbed, outsole)
    )

    assert result.original_face_indices.tolist() == [12, 13]
    assert result.central_support_length_coverage == pytest.approx(1.0)
    false_upper = next(
        layer for layer in result.diagnostics if layer["source_face_count"] == 10
    )
    assert false_upper["length_coverage"] >= 0.65
    assert false_upper["width_coverage"] >= 0.40
    assert false_upper["footprint_fill_fraction"] >= 0.25
    assert false_upper["central_support_length_coverage"] < 0.65
    assert not false_upper["qualifies"]


def birkenstock_like_layers(*, reverse_all_faces: bool = False) -> TriangleMesh:
    """Build a split smooth support above a downward-facing outsole."""

    mesh = combine(
        sheet((-1.0, 1.0), (-0.2, 0.2), (-1.0,) * 4),
        u_shaped_upper(0.0),
        sheet((-3.0, 3.0), (-1.0, 1.0), (0.0,) * 4),
        sheet((-5.0, 5.0), (-2.0, 2.0), (0.5,) * 4, upward=False),
    )
    if not reverse_all_faces:
        return mesh
    return TriangleMesh(mesh.vertices, mesh.faces[:, ::-1])


def test_local_height_trace_recovers_split_smooth_support() -> None:
    result = identify_footbed_surface(birkenstock_like_layers())

    assert result.selection_method == "local_height_trace"
    assert result.fallback_reason is not None
    assert result.central_support_length_coverage >= 0.65
    assert result.upward_facing_area_fraction == pytest.approx(1.0)
    assert not np.isin([14, 15], result.original_face_indices).any()
    assert any(
        layer["selection_method"] == "component_layers"
        and layer["qualification_failures"] == ["central_support"]
        for layer in result.diagnostics
    )


def test_global_winding_reversal_does_not_spuriously_trigger_fallback() -> None:
    result = identify_footbed_surface(
        birkenstock_like_layers(reverse_all_faces=True)
    )
    assert result.selection_method == "component_layers"
    assert result.fallback_reason is None


def test_ambiguous_outsole_case_fails_instead_of_guessing() -> None:
    ambiguous = combine(
        sheet((-1.0, 1.0), (-0.2, 0.2), (-1.0,) * 4),
        u_shaped_upper(0.0),
        sheet((-5.0, 5.0), (-2.0, 2.0), (0.5,) * 4, upward=False),
    )
    with pytest.raises(ValueError, match="local-height tracing found no"):
        identify_footbed_surface(ambiguous)


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
    assert '"central_support_length_coverage"' in message
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
    assert footbed.central_support_length_coverage == pytest.approx(0.94921875)
    assert footbed.upward_facing_area_fraction == pytest.approx(1.0)
    assert footbed.support_like_area_fraction == pytest.approx(1.0)
    assert footbed.area_weighted_median_y == pytest.approx(0.028700418, abs=1e-9)
    heights, valid = sample_footbed_y(
        footbed, np.asarray([[0.0, 0.0], [0.2, 0.0]])
    )
    np.testing.assert_array_equal(valid, [True, False])
    assert np.isfinite(heights[0])
    assert np.isnan(heights[1])


def test_pb129_selects_complete_interior_footbed() -> None:
    path = evaluation_root() / "pb129_shoe_low/reference_mesh.ply"
    if not path.is_file():
        pytest.skip(f"external PB129 dataset is unavailable: {path}")
    footbed = identify_footbed_surface(load_triangle_mesh(path))
    assert footbed.original_face_indices[0] == 2001
    assert footbed.original_face_indices[-1] == 6995
    assert footbed.mesh.vertices.shape == (400, 3)
    assert footbed.mesh.faces.shape == (713, 3)
    np.testing.assert_allclose(
        footbed.bounds,
        [
            [-0.09381073, 0.00014681, -0.03388835],
            [0.08254347, 0.02671137, 0.03761325],
        ],
        atol=1e-8,
    )
    assert footbed.height_grid.shape == (256, 119)
    assert np.count_nonzero(footbed.valid_mask) == 14486
    assert footbed.central_support_length_coverage == pytest.approx(0.81640625)


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
    assert len(first.original_face_indices) == EXPECTED_SUPPORT_FACE_COUNTS[shoe_name]
    assert first.length_coverage >= 0.65
    assert first.width_coverage >= 0.40
    assert first.central_support_length_coverage >= 0.65
    assert first.upward_facing_area_fraction >= 0.90
    assert first.support_like_area_fraction == pytest.approx(1.0)
    assert np.count_nonzero(first.valid_mask) > 0


@pytest.mark.parametrize("shoe_name", tuple(CORRECTED_SUPPORT_FACE_DIGESTS))
def test_corrected_shoes_preserve_expected_support(shoe_name: str) -> None:
    path = corrected_evaluation_root() / shoe_name / "reference_mesh.ply"
    if not path.is_file():
        pytest.skip(f"correctly oriented evaluation shoe is unavailable: {path}")

    footbed = identify_footbed_surface(load_triangle_mesh(path))
    digest = hashlib.sha256(
        np.asarray(footbed.original_face_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    assert digest == CORRECTED_SUPPORT_FACE_DIGESTS[shoe_name]

    if shoe_name != "birkenstock_arizona_sandal":
        assert footbed.selection_method == "component_layers"
        return

    assert footbed.selection_method == "local_height_trace"
    assert footbed.fallback_reason is not None
    assert footbed.mesh.vertices.shape == (6392, 3)
    assert footbed.mesh.faces.shape == (8861, 3)
    assert footbed.original_face_indices[0] == 875
    assert footbed.original_face_indices[-1] == 77013
    np.testing.assert_allclose(
        footbed.bounds,
        [
            [-0.10193875432014465, 0.006601303815841675, -0.03334125876426697],
            [0.10366135835647583, 0.027411580085754395, 0.03679308295249939],
        ],
        atol=1e-8,
    )
    assert footbed.length_coverage == pytest.approx(0.9375)
    assert footbed.width_coverage == pytest.approx(0.8217821782178241)
    assert footbed.central_support_length_coverage == pytest.approx(0.93359375)
    assert footbed.upward_facing_area_fraction == pytest.approx(1.0)
    normalization = build_shoe_normalization(load_triangle_mesh(path), footbed)
    assert normalization.functional_length > 0.0


def test_birkenstock_before_and_after_heading_selects_interior_support() -> None:
    before_path = (
        evaluation_root()
        / "birkenstock_arizona_sandal"
        / "reference_mesh.ply"
    )
    after_path = (
        corrected_evaluation_root()
        / "birkenstock_arizona_sandal"
        / "reference_mesh.ply"
    )
    if not before_path.is_file() or not after_path.is_file():
        pytest.skip("Birkenstock heading-regression meshes are unavailable")

    before = identify_footbed_surface(load_triangle_mesh(before_path))
    after = identify_footbed_surface(load_triangle_mesh(after_path))
    overlap = np.intersect1d(
        before.original_face_indices, after.original_face_indices
    )

    assert before.upward_facing_area_fraction == pytest.approx(1.0)
    assert after.upward_facing_area_fraction == pytest.approx(1.0)
    assert before.central_support_length_coverage >= 0.65
    assert after.central_support_length_coverage >= 0.65
    assert len(overlap) / min(
        len(before.original_face_indices), len(after.original_face_indices)
    ) >= 0.70


@pytest.mark.parametrize("angle_degrees", [-2.983, -2.0, 2.0])
def test_birkenstock_trace_is_stable_under_small_heading_changes(
    angle_degrees: float,
) -> None:
    path = (
        corrected_evaluation_root()
        / "birkenstock_arizona_sandal"
        / "reference_mesh.ply"
    )
    if not path.is_file():
        pytest.skip(f"correctly oriented Birkenstock is unavailable: {path}")

    mesh = load_triangle_mesh(path)
    baseline = identify_footbed_surface(mesh)
    angle = np.deg2rad(angle_degrees)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rotated = TriangleMesh(mesh.vertices @ rotation.T, mesh.faces)
    result = identify_footbed_surface(rotated)

    assert result.selection_method in {"component_layers", "local_height_trace"}
    assert result.central_support_length_coverage >= 0.65
    assert result.upward_facing_area_fraction == pytest.approx(1.0)
    overlap = np.intersect1d(
        baseline.original_face_indices, result.original_face_indices
    )
    assert len(overlap) / min(
        len(baseline.original_face_indices), len(result.original_face_indices)
    ) >= 0.70
