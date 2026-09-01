"""Tests for canonical right-shoe validation and normalization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foot_prior.footbed import FootbedSurface, identify_footbed_surface
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh
from foot_prior.normalization import (
    EXPECTED_SHOE_COORDINATE_SYSTEM,
    HIGH_HEEL_SHOE_PROFILE,
    NORMAL_SHOE_PROFILE,
    SHOE_AXIS_SEMANTICS,
    SHOE_SIDE,
    build_shoe_normalization,
    validate_shoe_frame_metadata,
)


DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMAL_SHOES = (
    "aj_12_basketball_sneakers",
    "birkenstock_arizona_sandal",
    "canvas_shoe",
    "crocs",
    "crocs_by_speedyart_studio",
    "crocs_shoe",
    "duinn_shoes_womens_hiking_sandal_sport",
    "leather_boots",
    "nike_air_jordan",
    "pb129_shoe_low",
    "priest_karol_wojtyas_sports_shoes",
    "sandal_1",
    "sandals_0001",
    "shoes_mockup_asset_vans_skate_old_skool_shoes",
    "sneaker_vibe",
    "sneakers_seen",
    "ww_ii_german_jack_boots",
)
EXPECTED_SUPPORT_FACE_DIGESTS = {
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


def write_metadata(
    path: Path,
    coordinate_system: object,
    mirror_width: bool,
    shoe_profile: object = NORMAL_SHOE_PROFILE,
) -> None:
    path.write_text(
        json.dumps(
            {
                "shoe_profile": shoe_profile,
                "canonical_geometry": {"mirror_width": mirror_width},
                "reference_mesh": {"coordinate_system": coordinate_system},
            }
        ),
        encoding="utf-8",
    )


def box_mesh(
    x_bounds: tuple[float, float] = (-1.0, 8.0),
    y_bounds: tuple[float, float] = (-1.0, 3.0),
    z_bounds: tuple[float, float] = (-1.0, 20.0),
) -> TriangleMesh:
    vertices = np.asarray(
        [
            [x_bounds[0], y_bounds[0], z_bounds[0]],
            [x_bounds[1], y_bounds[0], z_bounds[0]],
            [x_bounds[1], y_bounds[1], z_bounds[0]],
            [x_bounds[0], y_bounds[1], z_bounds[0]],
            [x_bounds[0], y_bounds[0], z_bounds[1]],
            [x_bounds[1], y_bounds[0], z_bounds[1]],
            [x_bounds[1], y_bounds[1], z_bounds[1]],
            [x_bounds[0], y_bounds[1], z_bounds[1]],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces)


def high_heel_runner_mesh() -> TriangleMesh:
    vertices = np.asarray(
        [
            [-5.0, -2.0, -2.0],
            [5.0, 2.0, -2.0],
            [5.0, 2.0, 2.0],
            [-5.0, -2.0, 2.0],
            [-5.0, -1.5, -2.0],
            [5.0, 2.5, -2.0],
            [5.0, 2.5, 2.0],
            [-5.0, -1.5, 2.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces)


def grid_footbed(
    mask: np.ndarray,
    heights: float | np.ndarray = 2.0,
    x_coordinates: np.ndarray | None = None,
    z_coordinates: np.ndarray | None = None,
) -> FootbedSurface:
    footprint = np.asarray(mask, dtype=bool)
    x_values = (
        np.arange(footprint.shape[0], dtype=np.float64)
        if x_coordinates is None
        else np.asarray(x_coordinates, dtype=np.float64)
    )
    z_values = (
        np.arange(footprint.shape[1], dtype=np.float64)
        if z_coordinates is None
        else np.asarray(z_coordinates, dtype=np.float64)
    )
    if np.isscalar(heights):
        height_grid = np.full(footprint.shape, float(heights), dtype=np.float64)
    else:
        height_grid = np.asarray(heights, dtype=np.float64).copy()
    height_grid[~footprint] = np.nan
    surface_y = 0.0 if not np.any(footprint) else float(np.nanmedian(height_grid))
    surface = TriangleMesh(
        np.asarray(
            [
                [x_values[0], surface_y, z_values[0]],
                [x_values[-1], surface_y, z_values[0]],
                [x_values[-1], surface_y, z_values[-1]],
                [x_values[0], surface_y, z_values[-1]],
            ]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    return FootbedSurface(
        mesh=surface,
        original_face_indices=np.asarray([0, 1], dtype=np.int64),
        bounds=surface.bounds,
        extents=surface.extents,
        x_coordinates=x_values,
        z_coordinates=z_values,
        height_grid=height_grid,
        valid_mask=footprint,
        length_coverage=1.0,
        width_coverage=1.0,
        central_support_length_coverage=1.0,
        upward_facing_area_fraction=1.0,
        support_like_area_fraction=1.0,
        area_weighted_median_y=surface_y,
        projected_xz_area=float(np.count_nonzero(footprint)),
        diagnostics=(),
    )


@pytest.mark.parametrize("mirror_width", [False, True])
def test_accepts_canonical_metadata_regardless_of_source_mirroring(
    tmp_path: Path, mirror_width: bool
) -> None:
    path = tmp_path / "blender_canonicalization.json"
    write_metadata(path, EXPECTED_SHOE_COORDINATE_SYSTEM, mirror_width)
    assert validate_shoe_frame_metadata(path) == NORMAL_SHOE_PROFILE


@pytest.mark.parametrize("shoe_profile", [None, "heel", "NORMAL", 1])
def test_rejects_missing_or_unknown_shoe_profile(
    tmp_path: Path, shoe_profile: object
) -> None:
    path = tmp_path / "blender_canonicalization.json"
    write_metadata(
        path,
        EXPECTED_SHOE_COORDINATE_SYSTEM,
        False,
        shoe_profile,
    )
    with pytest.raises(ValueError, match="invalid or missing shoe_profile"):
        validate_shoe_frame_metadata(path)


def test_rejects_incompatible_coordinate_system(tmp_path: Path) -> None:
    path = tmp_path / "blender_canonicalization.json"
    write_metadata(path, "x_width_y_up_z_length", False)
    with pytest.raises(ValueError, match="unsupported shoe coordinate system"):
        validate_shoe_frame_metadata(path)


def test_rejects_missing_reference_mesh_metadata(tmp_path: Path) -> None:
    path = tmp_path / "blender_canonicalization.json"
    path.write_text(
        json.dumps({"shoe_profile": NORMAL_SHOE_PROFILE}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing reference_mesh"):
        validate_shoe_frame_metadata(path)


def test_rejects_malformed_or_missing_metadata(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shoe-frame metadata JSON"):
        validate_shoe_frame_metadata(malformed)
    with pytest.raises(FileNotFoundError):
        validate_shoe_frame_metadata(tmp_path / "missing.json")


def test_right_shoe_axis_contract_is_explicit() -> None:
    assert SHOE_SIDE == "right"
    assert SHOE_AXIS_SEMANTICS == {
        "+X": "heel_to_toe",
        "+Y": "down_toward_sole",
        "+Z": "shoe_width",
    }


def test_all_evaluation_metadata_uses_the_canonical_frame() -> None:
    root = evaluation_root()
    if not root.is_dir():
        pytest.skip(f"external evaluation dataset is unavailable: {root}")
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(scenes) == 19
    profiles = []
    for scene in scenes:
        profiles.append(
            validate_shoe_frame_metadata(
                scene / "blender_canonicalization.json"
            )
        )
    assert profiles.count(NORMAL_SHOE_PROFILE) == 17
    assert profiles.count(HIGH_HEEL_SHOE_PROFILE) == 2


def test_flat_support_trims_narrow_ends_and_maps_landmarks() -> None:
    mask = np.ones((8, 20), dtype=bool)
    mask[0] = False
    mask[0, 10] = True
    mask[7] = False
    mask[7, 10] = True
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    assert normalization.functional_column_range == (1, 6)
    assert normalization.functional_length == pytest.approx(5.0)
    np.testing.assert_allclose(normalization.origin, [1.0, 2.0, 9.5])
    landmarks = normalization.shoe_points_to_normalized(
        np.stack((normalization.heel_landmark, normalization.toe_landmark))
    )
    np.testing.assert_allclose(landmarks[0], [0.0, 0.0, 0.0], atol=1e-14)
    assert landmarks[1, 0] == pytest.approx(1.0)


def test_sloped_support_uses_rear_and_front_heights() -> None:
    mask = np.ones((8, 20), dtype=bool)
    x_values = np.arange(8, dtype=np.float64)
    heights = np.broadcast_to(x_values[:, None] * 0.5, mask.shape).copy()
    normalization = build_shoe_normalization(
        box_mesh(), grid_footbed(mask, heights)
    )

    assert normalization.heel_landmark[1] == pytest.approx(0.25)
    assert normalization.toe_landmark[1] == pytest.approx(3.25)
    assert normalization.toe_landmark[1] > normalization.heel_landmark[1]


def test_largest_footprint_rejects_disconnected_noise() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:9, 4:16] = True
    mask[0, 0:2] = True
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    assert normalization.footprint_component_count == 2
    assert normalization.selected_footprint_cell_count == 7 * 12
    assert normalization.functional_column_range == (2, 8)


def test_holes_are_not_added_to_selected_footprint() -> None:
    mask = np.ones((6, 20), dtype=bool)
    mask[2:4, 8:12] = False
    original = mask.copy()
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    np.testing.assert_array_equal(mask, original)
    assert normalization.selected_footprint_cell_count == np.count_nonzero(mask)
    assert normalization.selected_footprint_cell_count < mask.size


def test_component_tie_prefers_greater_x_span() -> None:
    mask = np.zeros((9, 20), dtype=bool)
    mask[0:2, 0:8] = True
    mask[4:8, 10:14] = True
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    assert normalization.footprint_component_count == 2
    assert normalization.functional_column_range == (4, 7)


def test_complete_component_tie_prefers_lower_flattened_index() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[0:4, 0:4] = True
    mask[5:9, 10:14] = True
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    assert normalization.footprint_component_count == 2
    assert normalization.functional_column_range == (0, 3)


def test_equal_functional_runs_choose_heelward_range() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[0:3] = True
    mask[7:10] = True
    mask[3:7, 10] = True
    normalization = build_shoe_normalization(box_mesh(), grid_footbed(mask))

    assert normalization.footprint_component_count == 1
    assert normalization.functional_column_range == (0, 2)


def test_invalid_and_zero_length_support_fail_clearly() -> None:
    empty = np.zeros((4, 12), dtype=bool)
    with pytest.raises(ValueError, match="no occupied"):
        build_shoe_normalization(box_mesh(), grid_footbed(empty))

    one_column = np.zeros((4, 12), dtype=bool)
    one_column[2] = True
    with pytest.raises(ValueError, match="at least two X coordinates"):
        build_shoe_normalization(box_mesh(), grid_footbed(one_column))

    inconsistent = grid_footbed(np.ones((4, 12), dtype=bool))
    inconsistent.height_grid[0, 0] = np.nan
    with pytest.raises(ValueError, match="valid_mask"):
        build_shoe_normalization(box_mesh(), inconsistent)


def test_matrix_point_and_mesh_round_trip_preserves_axes_and_topology() -> None:
    shoe = box_mesh()
    normalization = build_shoe_normalization(
        shoe, grid_footbed(np.ones((8, 20), dtype=bool))
    )
    np.testing.assert_allclose(
        normalization.normalized_to_shoe @ normalization.shoe_to_normalized,
        np.eye(4),
        atol=1e-14,
    )
    points = np.stack(
        (
            normalization.origin,
            normalization.origin + [1.0, 0.0, 0.0],
            normalization.origin + [0.0, 1.0, 0.0],
            normalization.origin + [0.0, 0.0, 1.0],
        )
    )
    normalized_points = normalization.shoe_points_to_normalized(points)
    np.testing.assert_allclose(
        normalization.normalized_points_to_shoe(normalized_points),
        points,
        atol=1e-14,
    )
    assert np.all(np.diag(normalization.shoe_to_normalized)[:3] > 0.0)

    normalized_mesh = normalization.shoe_mesh_to_normalized(shoe)
    restored_mesh = normalization.normalized_mesh_to_shoe(normalized_mesh)
    np.testing.assert_allclose(restored_mesh.vertices, shoe.vertices, atol=1e-14)
    np.testing.assert_array_equal(restored_mesh.faces, shoe.faces)


def test_known_canvas_functional_normalization() -> None:
    scene = evaluation_root() / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    shoe = load_triangle_mesh(scene / "reference_mesh.ply")
    footbed = identify_footbed_surface(shoe)
    normalization = build_shoe_normalization(shoe, footbed)

    assert normalization.functional_column_range == (5, 246)
    assert normalization.functional_length == pytest.approx(
        0.20201217103749514, abs=1e-14
    )
    assert normalization.outer_length_ratio == pytest.approx(0.94140625)
    np.testing.assert_allclose(
        normalization.origin,
        [-0.10268253507092595, 0.029422231794832464, -0.00041614383614311384],
        atol=1e-14,
    )


def test_all_normal_shoes_preserve_support_and_normalize_deterministically() -> None:
    root = evaluation_root()
    if not root.is_dir():
        pytest.skip(f"external evaluation dataset is unavailable: {root}")
    for shoe_name in NORMAL_SHOES:
        shoe = load_triangle_mesh(root / shoe_name / "reference_mesh.ply")
        footbed = identify_footbed_surface(shoe)
        original_faces = footbed.original_face_indices.copy()
        digest = hashlib.sha256(
            np.asarray(original_faces, dtype=np.int64).tobytes()
        ).hexdigest()
        assert digest == EXPECTED_SUPPORT_FACE_DIGESTS[shoe_name]

        first = build_shoe_normalization(shoe, footbed)
        second = build_shoe_normalization(shoe, footbed)
        np.testing.assert_array_equal(footbed.original_face_indices, original_faces)
        np.testing.assert_allclose(first.shoe_to_normalized, second.shoe_to_normalized)
        assert np.isfinite(first.shoe_to_normalized).all()
        assert first.functional_length > 0.0
        assert 0.65 <= first.outer_length_ratio <= 1.0
        np.testing.assert_allclose(
            first.normalized_to_shoe @ first.shoe_to_normalized,
            np.eye(4),
            atol=1e-12,
        )
        normalized_landmarks = first.shoe_points_to_normalized(
            np.stack((first.heel_landmark, first.toe_landmark))
        )
        np.testing.assert_allclose(
            normalized_landmarks[0], [0.0, 0.0, 0.0], atol=1e-12
        )
        assert normalized_landmarks[1, 0] == pytest.approx(1.0, abs=1e-12)
        normalized_shoe = first.shoe_mesh_to_normalized(shoe)
        np.testing.assert_array_equal(normalized_shoe.faces, shoe.faces)
        np.testing.assert_allclose(
            first.normalized_mesh_to_shoe(normalized_shoe).vertices,
            shoe.vertices,
            atol=1e-12,
        )


def test_canvas_preparation_runner_and_overwrite_contract(tmp_path: Path) -> None:
    scene = evaluation_root() / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    output = tmp_path / "preparation"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("preserve me\n", encoding="utf-8")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_shoe_preparation.py"),
        "--shoe-mesh",
        str(scene / "reference_mesh.ply"),
        "--canonicalization",
        str(scene / "blender_canonicalization.json"),
        "--output-dir",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert {path.name for path in output.iterdir()} == {
        "shoe_preparation.json",
        "footbed_surface.ply",
        "footbed_overlay.ply",
        "shoe_normalized.ply",
        "keep.txt",
    }

    payload = json.loads((output / "shoe_preparation.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["shoe_profile"] == NORMAL_SHOE_PROFILE
    assert Path(payload["inputs"]["shoe_mesh"]).is_absolute()
    assert Path(payload["inputs"]["canonicalization"]).is_absolute()
    assert payload["coordinate_contract"]["side"] == "right"
    assert payload["normalization"]["functional_length"] == pytest.approx(
        0.20201217103749514
    )
    shoe = load_triangle_mesh(scene / "reference_mesh.ply")
    normalized = load_triangle_mesh(output / "shoe_normalized.ply")
    np.testing.assert_array_equal(normalized.faces, shoe.faces)
    assert normalized.vertices.shape == shoe.vertices.shape
    assert load_triangle_mesh(output / "footbed_surface.ply").faces.shape == (350, 3)
    overlay = load_triangle_mesh(output / "footbed_overlay.ply")
    assert len(overlay.vertices) == len(shoe.vertices) + 207
    assert len(overlay.faces) == len(shoe.faces) + 350

    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "already exist" in refused.stderr
    replaced = subprocess.run(
        [*command, "--overwrite"], capture_output=True, text=True, check=False
    )
    assert replaced.returncode == 0, replaced.stderr
    assert unrelated.read_text(encoding="utf-8") == "preserve me\n"


def test_high_heel_runner_writes_support_review_without_normalization(
    tmp_path: Path,
) -> None:
    shoe_mesh = tmp_path / "high_heel.ply"
    save_triangle_mesh(shoe_mesh, high_heel_runner_mesh())
    canonicalization = tmp_path / "blender_canonicalization.json"
    write_metadata(
        canonicalization,
        EXPECTED_SHOE_COORDINATE_SYSTEM,
        False,
        HIGH_HEEL_SHOE_PROFILE,
    )
    output = tmp_path / "preparation"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_shoe_preparation.py"),
        "--shoe-mesh",
        str(shoe_mesh),
        "--canonicalization",
        str(canonicalization),
        "--output-dir",
        str(output),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "[support-detected]" in result.stdout
    assert {path.name for path in output.iterdir()} == {
        "shoe_preparation.json",
        "footbed_surface.ply",
        "footbed_overlay.ply",
    }
    payload = json.loads((output / "shoe_preparation.json").read_text())
    assert payload["shoe_profile"] == HIGH_HEEL_SHOE_PROFILE
    assert (
        payload["preparation_status"]
        == "support_detected_normalization_deferred"
    )
    assert payload["normalization"] is None
    assert payload["footbed_selection"]["selection_method"] == (
        "high_heel_upper_envelope"
    )
    assert payload["high_heel_support"]["heel_elevation"] > 0.0
    assert not (output / "shoe_normalized.ply").exists()

    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "already exist" in refused.stderr
    replaced = subprocess.run(
        [*command, "--overwrite"], capture_output=True, text=True, check=False
    )
    assert replaced.returncode == 0, replaced.stderr
