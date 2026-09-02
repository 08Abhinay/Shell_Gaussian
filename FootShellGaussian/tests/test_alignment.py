"""Focused tests for preparation-driven rigid SUPR placement."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foot_prior.alignment import (
    build_initial_placement,
    make_supr_to_shoe_axis_remap,
    transform_points,
)
from foot_prior.mesh import TriangleMesh, load_triangle_mesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
PREPARATION_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/"
    "shoe_preparation"
)
RUNNER = PROJECT_ROOT / "scripts/run_alignment.py"
ARTIFACT_NAMES = {
    "initial_placement.json",
    "foot_initial.ply",
    "footbed_normalized.ply",
    "initial_placement_overlay.ply",
}


def _synthetic_raw_foot() -> TriangleMesh:
    remapped_vertices = np.asarray(
        [
            [0.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [1.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    raw_vertices = transform_points(
        remapped_vertices, make_supr_to_shoe_axis_remap()
    )
    faces = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    return TriangleMesh(raw_vertices, faces)


def _sloped_support() -> TriangleMesh:
    vertices = np.asarray(
        [
            [-0.1, 0.19, -1.0],
            [1.0, 0.30, -1.0],
            [1.0, 0.30, 1.0],
            [-0.1, 0.19, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return TriangleMesh(vertices, faces)


def _run_alignment(*arguments: object) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUNNER), *(str(value) for value in arguments)]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_exact_axis_remap_and_point_round_trip() -> None:
    remap = make_supr_to_shoe_axis_remap()
    basis = np.eye(3, dtype=np.float64)
    expected = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(transform_points(basis, remap), expected)
    np.testing.assert_allclose(remap @ remap, np.eye(4), atol=0.0, rtol=0.0)

    points = np.asarray([[0.2, -0.4, 0.7], [-1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        transform_points(transform_points(points, remap), np.linalg.inv(remap)),
        points,
        atol=1e-12,
        rtol=0.0,
    )


def test_synthetic_placement_anchors_length_centerline_and_contact() -> None:
    foot = _synthetic_raw_foot()
    support = _sloped_support()
    shoe_to_normalized = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.3],
            [0.0, 2.0, 0.0, -0.4],
            [0.0, 0.0, 2.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    normalized_to_shoe = np.linalg.inv(shoe_to_normalized)
    centerline = np.asarray([[0.0, 0.30], [0.85, 0.40], [1.0, 0.42]])

    result = build_initial_placement(
        foot_mesh=foot,
        normalized_shoe_mesh=support,
        normalized_support_mesh=support,
        normalized_centerline_xz=centerline,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        foot_length_ratio=0.85,
    )

    assert result.achieved_plantar_length_ratio == pytest.approx(0.85)
    assert result.heel_reference_normalized[0] == pytest.approx(0.0)
    assert result.lateral_rms_after <= result.lateral_rms_before
    assert result.plantar_support_coverage == pytest.approx(1.0)
    assert result.minimum_support_gap == pytest.approx(0.0, abs=1e-12)
    assert result.maximum_support_gap >= 0.0
    np.testing.assert_allclose(
        result.normalized_shoe_to_foot @ result.foot_to_normalized_shoe,
        np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.original_shoe_to_foot @ result.foot_to_original_shoe,
        np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )
    round_trip = result.normalized_shoe_points_to_foot(
        result.foot_points_to_normalized_shoe(foot.vertices)
    )
    np.testing.assert_allclose(round_trip, foot.vertices, atol=1e-12, rtol=0.0)


def test_runner_uses_saved_preparation_and_preserves_unrelated_outputs(
    tmp_path: Path,
) -> None:
    source = PREPARATION_ROOT / "canvas_shoe"
    if not source.is_dir():
        pytest.skip(f"prepared canvas shoe is unavailable: {source}")

    preparation = tmp_path / "prepared"
    preparation.mkdir()
    for name in ("shoe_normalized.ply", "footbed_surface.ply"):
        shutil.copy2(source / name, preparation / name)
    metadata = json.loads((source / "shoe_preparation.json").read_text())
    metadata["inputs"]["shoe_mesh"] = str(tmp_path / "missing-original-shoe.ply")
    (preparation / "shoe_preparation.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("preserve me", encoding="utf-8")
    base_arguments = (
        "--preparation-dir",
        preparation,
        "--supr-model",
        SUPR_MODEL,
        "--output-dir",
        output,
    )
    first = _run_alignment(*base_arguments)
    assert first.returncode == 0, first.stderr
    assert {path.name for path in output.iterdir()} == ARTIFACT_NAMES | {"keep.txt"}

    payload = json.loads((output / "initial_placement.json").read_text())
    assert payload["shoe_profile"] == "normal"
    assert payload["placement"]["achieved_plantar_length_ratio"] == pytest.approx(
        0.85
    )
    assert payload["plantar_contact"]["sample_count"] == 107
    assert payload["plantar_contact"]["covered_sample_count"] == 104
    np.testing.assert_allclose(
        np.asarray(payload["transforms"]["normalized_shoe_to_foot"])
        @ np.asarray(payload["transforms"]["foot_to_normalized_shoe"]),
        np.eye(4),
        atol=1e-10,
        rtol=0.0,
    )
    foot = load_triangle_mesh(output / "foot_initial.ply")
    assert foot.vertices.shape == (266, 3)
    assert foot.faces.shape == (515, 3)

    refused = _run_alignment(*base_arguments)
    assert refused.returncode != 0
    assert "pass --overwrite" in refused.stderr
    overwritten = _run_alignment(*base_arguments, "--overwrite")
    assert overwritten.returncode == 0, overwritten.stderr
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_runner_rejects_high_heels_before_writing(tmp_path: Path) -> None:
    preparation = PREPARATION_ROOT / "red_high_heel_shoes"
    if not preparation.is_dir():
        pytest.skip(f"prepared high heel is unavailable: {preparation}")
    output = tmp_path / "high-heel-output"
    result = _run_alignment(
        "--preparation-dir",
        preparation,
        "--supr-model",
        SUPR_MODEL,
        "--output-dir",
        output,
    )
    assert result.returncode != 0
    assert "shoe_profile='normal' only" in result.stderr
    assert not output.exists()
