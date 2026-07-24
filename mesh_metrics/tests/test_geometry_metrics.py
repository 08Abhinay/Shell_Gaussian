from __future__ import annotations

import unittest

import numpy as np
import trimesh

from mesh_metrics.geometry_metrics import GeometryMetricConfig, compute_geometry_metrics


class GeometryMetricsTest(unittest.TestCase):
    config = GeometryMetricConfig(sample_count=10_000, seed=91)

    def test_identical_mesh_has_perfect_metrics(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        result = compute_geometry_metrics(mesh.copy(), mesh, self.config)
        headline = result["headline"]
        self.assertLess(headline["chamfer_l1_percent"], 1e-4)
        self.assertAlmostEqual(headline["f_score_1_percent"], 1.0, places=6)
        self.assertGreater(headline["normal_consistency"], 0.9999)
        self.assertLess(headline["p95_distance_percent"], 1e-4)

    def test_missing_component_worsens_completeness(self) -> None:
        left = trimesh.creation.box(extents=[1.0, 0.7, 0.5])
        left.apply_translation([-0.8, 0.0, 0.0])
        right = trimesh.creation.box(extents=[0.7, 0.5, 0.4])
        right.apply_translation([0.9, 0.0, 0.0])
        ground_truth = trimesh.util.concatenate((left, right))
        result = compute_geometry_metrics(left.copy(), ground_truth, self.config)
        headline = result["headline"]
        self.assertGreater(
            headline["completeness_percent"],
            5.0 * headline["accuracy_percent"],
        )

    def test_extra_component_worsens_accuracy(self) -> None:
        ground_truth = trimesh.creation.box(extents=[1.0, 0.7, 0.5])
        extra = trimesh.creation.box(extents=[0.5, 0.4, 0.3])
        extra.apply_translation([2.0, 0.0, 0.0])
        prediction = trimesh.util.concatenate((ground_truth.copy(), extra))
        result = compute_geometry_metrics(prediction, ground_truth, self.config)
        headline = result["headline"]
        self.assertGreater(
            headline["accuracy_percent"],
            5.0 * headline["completeness_percent"],
        )
        self.assertEqual(
            result["diagnostics"]["prediction"]["connected_components"],
            2,
        )

    def test_reversed_winding_does_not_penalize_normals(self) -> None:
        ground_truth = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        prediction = ground_truth.copy()
        prediction.faces = np.fliplr(prediction.faces)
        result = compute_geometry_metrics(prediction, ground_truth, self.config)
        self.assertGreater(result["headline"]["normal_consistency"], 0.9999)

    def test_metrics_are_deterministic(self) -> None:
        ground_truth = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        prediction = ground_truth.copy()
        prediction.vertices[:, 0] *= 0.97
        first = compute_geometry_metrics(prediction, ground_truth, self.config)
        second = compute_geometry_metrics(prediction, ground_truth, self.config)
        self.assertEqual(first["headline"], second["headline"])
        self.assertEqual(first["f_scores"], second["f_scores"])


if __name__ == "__main__":
    unittest.main()
