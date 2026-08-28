"""Tests for fixed-axis SUPR-to-shoe alignment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from foot_prior.alignment import (
    build_axis_scale_xz_transform,
    build_initial_alignment,
    make_supr_to_shoe_axis_remap,
    transform_points,
)
from foot_prior.footbed import identify_footbed_surface
from foot_prior.mesh import TriangleMesh, load_triangle_mesh
from foot_prior.supr_foot import load_neutral_supr_foot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right"
)


def tetrahedron(vertices: np.ndarray) -> TriangleMesh:
    return TriangleMesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )


def test_exact_axis_remap_of_basis_vectors() -> None:
    basis = np.eye(3, dtype=np.float64)
    remapped = transform_points(basis, make_supr_to_shoe_axis_remap())
    np.testing.assert_array_equal(
        remapped,
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_matrix_inverse_and_point_round_trip() -> None:
    foot = tetrahedron(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    shoe = tetrahedron(
        [[-5.0, -2.0, -3.0], [5.0, -2.0, -3.0], [-5.0, 2.0, -3.0], [-5.0, -2.0, 3.0]]
    )
    matrix, scale, translation = build_axis_scale_xz_transform(foot, shoe)
    inverse = np.linalg.inv(matrix)
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()
    assert scale == pytest.approx(4.25)
    assert translation[1] == 0.0
    np.testing.assert_allclose(inverse @ matrix, np.eye(4), atol=1e-12)
    points = np.asarray([[0.25, 0.5, 1.5], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        transform_points(transform_points(points, matrix), inverse), points, atol=1e-12
    )


def test_exact_length_ratio_and_xz_centering_on_canvas() -> None:
    scene = Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT)) / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    foot = load_neutral_supr_foot(SUPR_MODEL)
    shoe = load_triangle_mesh(scene / "reference_mesh.ply")
    matrix, scale, translation = build_axis_scale_xz_transform(foot, shoe, 0.85)
    aligned = transform_points(foot.vertices, matrix)
    bounds = np.stack((aligned.min(axis=0), aligned.max(axis=0)), axis=0)
    center = bounds.mean(axis=0)
    assert np.ptp(aligned[:, 0]) / shoe.extents[0] == pytest.approx(0.85, abs=1e-12)
    np.testing.assert_allclose(center[[0, 2]], shoe.center[[0, 2]], atol=1e-12)
    assert scale == pytest.approx(0.6706355053488126, abs=1e-12)
    assert translation[1] == 0.0
    assert np.ptp(aligned[:, 2]) < shoe.extents[2]


@pytest.mark.parametrize("ratio", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_length_ratio(ratio: float) -> None:
    mesh = tetrahedron(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    with pytest.raises(ValueError, match="length_ratio"):
        build_axis_scale_xz_transform(mesh, mesh, ratio)


def raw_plantar_sheet() -> TriangleMesh:
    return TriangleMesh(
        np.asarray(
            [[-0.5, 0.0, -2.0], [0.5, 0.0, -2.0], [0.5, 0.0, 2.0], [-0.5, 0.0, 2.0]]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def shoe_sheet(y_left: float, y_right: float) -> TriangleMesh:
    return TriangleMesh(
        np.asarray(
            [
                [-5.0, y_left, -2.0],
                [5.0, y_right, -2.0],
                [5.0, y_right, 2.0],
                [-5.0, y_left, 2.0],
            ]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def test_flat_footbed_first_contact() -> None:
    foot = raw_plantar_sheet()
    shoe = shoe_sheet(1.0, 1.0)
    alignment = build_initial_alignment(
        foot, shoe, identify_footbed_surface(shoe), length_ratio=0.85
    )
    assert alignment.translation[1] == pytest.approx(1.0)
    assert alignment.footbed_contact_coverage == pytest.approx(1.0)
    assert alignment.minimum_footbed_gap == pytest.approx(0.0, abs=1e-14)
    assert alignment.maximum_footbed_gap == pytest.approx(0.0, abs=1e-14)


def test_sloped_footbed_first_contact_has_no_negative_gaps() -> None:
    foot = raw_plantar_sheet()
    shoe = shoe_sheet(0.5, 1.5)
    alignment = build_initial_alignment(
        foot, shoe, identify_footbed_surface(shoe), length_ratio=0.85
    )
    assert alignment.minimum_footbed_gap == pytest.approx(0.0, abs=1e-14)
    assert alignment.maximum_footbed_gap > 0.0
    assert alignment.minimum_footbed_gap >= 0.0


def test_canvas_first_contact_coverage_gaps_and_round_trip() -> None:
    scene = Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT)) / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    foot = load_neutral_supr_foot(SUPR_MODEL)
    shoe = load_triangle_mesh(scene / "reference_mesh.ply")
    footbed = identify_footbed_surface(shoe)
    alignment = build_initial_alignment(foot, shoe, footbed)
    assert alignment.plantar_sample_count == 107
    assert alignment.covered_plantar_sample_count == 104
    assert alignment.footbed_contact_coverage == pytest.approx(104 / 107)
    assert alignment.footbed_contact_coverage >= 0.95
    assert alignment.translation[1] == pytest.approx(-0.8870308744255349, abs=1e-12)
    assert alignment.minimum_footbed_gap == pytest.approx(0.0, abs=1e-14)
    assert alignment.maximum_footbed_gap == pytest.approx(0.021287513665358193, abs=1e-12)
    assert alignment.minimum_footbed_gap >= 0.0
    np.testing.assert_allclose(
        alignment.shoe_to_foot @ alignment.foot_to_shoe, np.eye(4), atol=1e-12
    )
    samples = foot.vertices[[0, 96, 265]]
    np.testing.assert_allclose(
        alignment.shoe_points_to_foot(alignment.foot_points_to_shoe(samples)),
        samples,
        atol=1e-12,
    )


def test_canvas_runner_artifacts_and_overwrite_contract(tmp_path: Path) -> None:
    scene = Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT)) / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    output = tmp_path / "alignment"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("preserve me\n", encoding="utf-8")
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "FootShellGaussian/scripts/run_alignment.py"),
        "--shoe-mesh",
        str(scene / "reference_mesh.ply"),
        "--supr-model",
        str(SUPR_MODEL),
        "--output-dir",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    expected = {
        "alignment.json",
        "foot_aligned.ply",
        "footbed_surface.ply",
        "alignment_overlay.ply",
        "keep.txt",
    }
    assert {path.name for path in output.iterdir()} == expected
    payload = json.loads((output / "alignment.json").read_text())
    assert payload["schema_version"] == 1
    assert Path(payload["inputs"]["shoe_mesh"]).is_absolute()
    assert Path(payload["inputs"]["supr_model"]).is_absolute()
    assert payload["plantar_contact"]["coverage"] == pytest.approx(104 / 107)
    assert load_triangle_mesh(output / "foot_aligned.ply").vertices.shape == (266, 3)
    assert load_triangle_mesh(output / "footbed_surface.ply").faces.shape == (350, 3)
    overlay = load_triangle_mesh(output / "alignment_overlay.ply")
    assert overlay.vertices.shape == (15030, 3)
    assert overlay.faces.shape == (26883, 3)

    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "already exist" in refused.stderr
    replaced = subprocess.run(
        [*command, "--overwrite"], capture_output=True, text=True, check=False
    )
    assert replaced.returncode == 0, replaced.stderr
    assert unrelated.read_text() == "preserve me\n"
