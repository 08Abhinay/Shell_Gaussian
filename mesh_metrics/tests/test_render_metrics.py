from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from mesh_metrics.camera_io import load_test_cameras, project_rotation_x
from mesh_metrics.mesh_io import load_mesh
from mesh_metrics.render_metrics import (
    HeldOutRenderer,
    RenderMetricConfig,
    _evaluation_protocol,
    boundary_f_score,
    evaluate_rendered_view,
    load_ground_truth_view,
)
from mesh_metrics.surface_queries import TriangleSurface


SCENE = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "gshell/golden_set_evaluation/air_jordan_1"
)
TURNTABLE_SCENE = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "gshell/golden_set_evaluation_turntable/air_jordan_1"
)


class RenderMetricsTest(unittest.TestCase):
    def test_project_rotation_matches_dataset_convention(self) -> None:
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(project_rotation_x(math.pi / 2.0), expected, atol=1e-12)

    @unittest.skipUnless(SCENE.is_dir(), "Air Jordan evaluation scene is unavailable")
    def test_heldout_split_has_six_views_per_ring(self) -> None:
        cameras = load_test_cameras(SCENE)
        self.assertEqual(len(cameras), 30)
        counts: dict[float, int] = {}
        for camera in cameras:
            counts[camera.elevation_deg] = counts.get(camera.elevation_deg, 0) + 1
            center_direction = camera.effective_c2w[:3, :3] @ np.array([0.0, 0.0, -1.0])
            center_direction /= np.linalg.norm(center_direction)
            expected = -camera.effective_c2w[:3, 3]
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(center_direction, expected, atol=1e-7)
        self.assertEqual(counts, {0.0: 6, -25.0: 6, 20.0: 6, 45.0: 6, 65.0: 6})

    @unittest.skipUnless(TURNTABLE_SCENE.is_dir(), "Turntable scene is unavailable")
    def test_turntable_split_has_six_level_views(self) -> None:
        cameras = load_test_cameras(TURNTABLE_SCENE)
        self.assertEqual(len(cameras), 6)
        self.assertEqual({camera.elevation_deg for camera in cameras}, {0.0})
        for camera in cameras:
            self.assertAlmostEqual(camera.radius, 1.0)

    def test_boundary_f_score_detects_shift(self) -> None:
        ground_truth = np.zeros((64, 64), dtype=bool)
        ground_truth[16:48, 12:52] = True
        exact = boundary_f_score(ground_truth, ground_truth, tolerance_px=2.0)
        self.assertEqual(exact["f_score"], 1.0)
        shifted = np.roll(ground_truth, shift=8, axis=1)
        score = boundary_f_score(shifted, ground_truth, tolerance_px=2.0)
        self.assertLess(score["f_score"], 0.8)

    def test_supported_evaluation_protocols_are_strict(self) -> None:
        full = [
            {"elevation_deg": elevation}
            for elevation in (-25.0, 0.0, 20.0, 45.0, 65.0)
            for _ in range(6)
        ]
        turntable = [{"elevation_deg": 0.0} for _ in range(6)]
        self.assertEqual(_evaluation_protocol(full), "full_view")
        self.assertEqual(_evaluation_protocol(turntable), "turntable")
        with self.assertRaisesRegex(ValueError, "Held-out cameras must follow"):
            _evaluation_protocol([{"elevation_deg": 0.0} for _ in range(5)])

    @unittest.skipUnless(SCENE.is_dir(), "Air Jordan evaluation scene is unavailable")
    def test_reference_mesh_reproduces_first_blender_test_frame(self) -> None:
        cameras = load_test_cameras(SCENE)
        camera = cameras[0]
        mesh = load_mesh(SCENE / "reference_mesh.ply")
        diagonal = float(np.linalg.norm(mesh.extents))
        renderer = HeldOutRenderer(camera.width, camera.height, camera.fov_x_rad)
        predicted_mask, predicted_depth = renderer.render(
            TriangleSurface(mesh),
            camera,
            chunk_size=250_000,
        )
        ground_truth_mask, ground_truth_depth = load_ground_truth_view(camera)
        result = evaluate_rendered_view(
            predicted_mask,
            predicted_depth,
            ground_truth_mask,
            ground_truth_depth,
            camera,
            diagonal,
            RenderMetricConfig(),
        )
        self.assertGreaterEqual(result["silhouette_iou"], 0.98)
        self.assertLessEqual(result["depth_relative_p95"], 0.01)


if __name__ == "__main__":
    unittest.main()
