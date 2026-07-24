from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from mesh_metrics.alignment import (
    AlignmentConfig,
    align_point_clouds,
    apply_transform,
    compose_similarity,
    decompose_similarity,
    estimate_similarity,
)
from mesh_metrics.mesh_io import load_mesh, sample_surface, transform_mesh


def _rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(angle_degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _asymmetric_points(count: int = 1_500, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = directions * np.array([1.8, 0.7, 0.45])
    points[:, 2] += 0.13 * points[:, 0] ** 2 + 0.07 * points[:, 1]
    points[:, 1] += 0.06 * points[:, 0] ** 2 + 0.02 * points[:, 0]
    return points


class SimilarityAlignmentTest(unittest.TestCase):
    def test_umeyama_recovers_rotation_translation_and_uniform_scale(self) -> None:
        source = _asymmetric_points(500)
        expected = compose_similarity(
            1.7,
            _rotation(np.array([0.2, 0.7, 0.4]), 43.0),
            np.array([2.0, -0.5, 1.2]),
        )
        target = apply_transform(source, expected)
        actual = estimate_similarity(source, target)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_multistart_similarity_icp_recovers_large_misalignment(self) -> None:
        ground_truth = _asymmetric_points()
        applied = compose_similarity(
            2.3,
            _rotation(np.array([0.3, 0.8, 0.5]), 67.0),
            np.array([3.0, -1.5, 0.7]),
        )
        prediction = apply_transform(ground_truth, applied)
        config = AlignmentConfig(
            sample_count=len(prediction),
            coarse_sample_count=500,
            candidate_count=4,
            max_iterations=60,
            tolerance=1e-9,
            seed=3,
        )
        result = align_point_clouds(prediction, ground_truth, config)
        aligned = apply_transform(prediction, result.transform)

        np.testing.assert_allclose(aligned, ground_truth, atol=2e-5)
        self.assertAlmostEqual(result.scale, 1.0 / 2.3, places=5)
        self.assertAlmostEqual(result.rotation_determinant, 1.0, places=7)
        self.assertLess(result.after_error, 1e-5)

    def test_reflection_is_not_returned_as_a_rotation(self) -> None:
        source = _asymmetric_points(500)
        target = source.copy()
        target[:, 0] *= -1.0
        transform = estimate_similarity(source, target)
        _, rotation, _ = decompose_similarity(transform)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)
        residual = np.mean(np.linalg.norm(apply_transform(source, transform) - target, axis=1))
        self.assertGreater(residual, 0.05)

    def test_trimmed_alignment_ignores_outliers_and_missing_region(self) -> None:
        ground_truth = _asymmetric_points(1_800, seed=19)
        partial = ground_truth[ground_truth[:, 0] < np.quantile(ground_truth[:, 0], 0.9)]
        applied = compose_similarity(
            1.8,
            _rotation(np.array([0.5, 0.2, 0.7]), 52.0),
            np.array([1.4, -0.7, 0.8]),
        )
        prediction_core = apply_transform(partial, applied)
        rng = np.random.default_rng(8)
        outliers = rng.uniform(-8.0, 8.0, size=(250, 3))
        prediction = np.concatenate((prediction_core, outliers), axis=0)
        rng.shuffle(prediction)

        config = AlignmentConfig(
            sample_count=len(prediction),
            coarse_sample_count=500,
            candidate_count=4,
            inlier_fraction=0.8,
            max_iterations=60,
            tolerance=1e-8,
        )
        result = align_point_clouds(prediction, ground_truth, config)
        aligned_core = apply_transform(prediction_core, result.transform)
        np.testing.assert_allclose(aligned_core, partial, atol=2e-5)
        self.assertAlmostEqual(result.scale, 1.0 / 1.8, places=5)
        self.assertTrue(result.converged)

    def test_surface_sampling_and_transform_do_not_modify_input(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 0.8, 0.5])
        original = mesh.vertices.copy()
        first, _ = sample_surface(mesh, 500, seed=7)
        second, _ = sample_surface(mesh, 500, seed=7)
        np.testing.assert_array_equal(first, second)

        transform = compose_similarity(1.2, np.eye(3), np.array([1.0, 2.0, 3.0]))
        moved = transform_mesh(mesh, transform)
        np.testing.assert_array_equal(mesh.vertices, original)
        self.assertFalse(np.allclose(moved.vertices, original))

    def test_scene_loader_bakes_hierarchy_transforms(self) -> None:
        scene = trimesh.Scene()
        box = trimesh.creation.box(extents=[1.0, 0.5, 0.25])
        transform = trimesh.transformations.translation_matrix([3.0, -2.0, 1.0])
        scene.add_geometry(box, geom_name="box", node_name="box_node", transform=transform)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            scene.export(path)
            loaded = load_mesh(path)
        np.testing.assert_allclose(loaded.bounds.mean(axis=0), [3.0, -2.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
