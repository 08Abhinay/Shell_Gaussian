"""Tests for triangle-mesh primitives and file I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from foot_prior.mesh import (
    TriangleMesh,
    combine_colored_meshes,
    load_triangle_mesh,
    save_triangle_mesh,
    transform_mesh,
)


DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation"
)


def evaluation_root() -> Path:
    return Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT))


def triangle(offset: float = 0.0) -> TriangleMesh:
    return TriangleMesh(
        np.array(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]]
        ),
        np.array([[0, 1, 2]], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("vertices", "faces", "error"),
    [
        (np.empty((0, 3)), np.array([[0, 0, 0]]), ValueError),
        (np.zeros((3, 2)), np.array([[0, 1, 2]]), ValueError),
        (np.array([[0.0, 0.0, np.nan]] * 3), np.array([[0, 1, 2]]), ValueError),
        (np.zeros((3, 3)), np.empty((0, 3), dtype=int), ValueError),
        (np.zeros((3, 3)), np.array([[0, 1, 3]]), ValueError),
        (np.zeros((3, 3)), np.array([[0.0, 1.0, 2.0]]), TypeError),
    ],
)
def test_rejects_invalid_meshes(
    vertices: np.ndarray,
    faces: np.ndarray,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TriangleMesh(vertices, faces)


def test_bounds_extents_and_center() -> None:
    mesh = triangle(offset=-2.0)
    np.testing.assert_allclose(mesh.bounds, [[-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    np.testing.assert_allclose(mesh.extents, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(mesh.center, [-1.5, 0.5, 0.0])


def test_binary_ply_round_trip_preserves_topology(tmp_path: Path) -> None:
    mesh = TriangleMesh(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]],
        dtype=np.uint8,
    )
    path = tmp_path / "mesh.ply"
    save_triangle_mesh(path, mesh, colors)
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian")
    loaded = load_triangle_mesh(path)
    np.testing.assert_allclose(loaded.vertices, mesh.vertices)
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
    np.testing.assert_array_equal(loaded.vertex_colors, colors)


def test_transform_and_combine_preserve_face_order() -> None:
    first = triangle()
    matrix = np.eye(4)
    matrix[:3, 3] = [2.0, 3.0, 4.0]
    second = transform_mesh(first, matrix)
    combined = combine_colored_meshes(
        (first, (128, 128, 128)), (second, (0, 80, 255, 255))
    )
    np.testing.assert_array_equal(combined.faces, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_allclose(second.vertices, first.vertices + [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(combined.vertex_colors[0], [128, 128, 128, 255])
    np.testing.assert_array_equal(combined.vertex_colors[-1], [0, 80, 255, 255])


def test_known_canvas_mesh_counts_bounds_and_convention() -> None:
    scene = evaluation_root() / "canvas_shoe"
    if not scene.is_dir():
        pytest.skip(f"external canvas dataset is unavailable: {scene}")
    mesh = load_triangle_mesh(scene / "reference_mesh.ply")
    assert mesh.vertices.shape == (14764, 3)
    assert mesh.faces.shape == (26368, 3)
    np.testing.assert_allclose(
        mesh.bounds,
        [
            [-0.1072927713394165, -0.04618929326534271, -0.03994980826973915],
            [0.1072927713394165, 0.04618929699063301, 0.03994980826973915],
        ],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        mesh.extents,
        [0.214585542678833, 0.09237859025597572, 0.0798996165394783],
        atol=1e-8,
    )
    metadata = json.loads((scene / "blender_canonicalization.json").read_text())
    assert "x_length_y_down_z_width" in metadata["reference_mesh"]["coordinate_system"]
