"""Tests for fixed-axis SUPR-to-shoe alignment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from foot_prior.alignment import (
    build_axis_scale_xz_transform,
    make_supr_to_shoe_axis_remap,
    transform_points,
)
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
