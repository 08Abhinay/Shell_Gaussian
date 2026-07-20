from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_tools_blender import pipeline


class DirectBlenderPipelineTest(unittest.TestCase):
    def test_manifest_covers_every_external_glb(self) -> None:
        records = pipeline.load_manifest(
            pipeline.DEFAULT_MANIFEST,
            pipeline.DEFAULT_SOURCE_ROOT,
        )
        self.assertEqual(len(records), 22)
        self.assertTrue(all(record["reviewed"] for record in records))

    def test_audit_can_open_an_unreviewed_manifest_but_build_cannot(self) -> None:
        payload = json.loads(pipeline.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        payload["shoes"][0]["reviewed"] = False
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not reviewed"):
                pipeline.load_manifest(manifest, pipeline.DEFAULT_SOURCE_ROOT)
            records = pipeline.load_manifest(
                manifest,
                pipeline.DEFAULT_SOURCE_ROOT,
                require_reviewed=False,
            )
        self.assertFalse(records[0]["reviewed"])

    def test_first_ring_matches_turntable_phase_and_direction(self) -> None:
        expected_centers = {
            0: (0.0, 1.0, 0.0),
            9: (1.0, 0.0, 0.0),
            18: (0.0, -1.0, 0.0),
            27: (-1.0, 0.0, 0.0),
        }
        for index, expected in expected_centers.items():
            saved, _, _ = pipeline.expected_frame(index)
            np.testing.assert_allclose(saved[:3, 3], expected, atol=1e-7)

        angles = []
        for index in range(pipeline.VIEWS_PER_RING):
            saved, _, _ = pipeline.expected_frame(index)
            center = saved[:3, 3]
            angles.append(math.atan2(center[1], center[0]))
        steps = np.degrees(np.diff(np.unwrap(angles)))
        np.testing.assert_allclose(steps, -10.0, atol=1e-7)

    def test_orbit_matches_processed_turntable_reference(self) -> None:
        root = Path(
            "/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell_shoes_size_metadata"
        )
        reference_json = next(path for path in sorted(root.glob("*/transforms.json")))
        frames = json.loads(reference_json.read_text(encoding="utf-8"))["frames"][:36]
        centers = np.asarray(
            [np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames]
        )
        self.assertAlmostEqual(float(np.median(np.linalg.norm(centers, axis=1))), 1.0, delta=0.01)
        self.assertAlmostEqual(float(centers[0, 0]), 0.0, delta=0.01)
        self.assertAlmostEqual(float(centers[0, 1]), 1.0, delta=0.01)
        angles = np.unwrap(np.arctan2(centers[:, 1], centers[:, 0]))
        self.assertAlmostEqual(float(np.degrees(np.median(np.diff(angles)))), -10.0, delta=0.01)

    def test_saved_pose_accounts_for_the_unchanged_gshell_loader(self) -> None:
        for index in range(pipeline.VIEW_COUNT):
            saved, effective, _ = pipeline.expected_frame(index)
            recovered = pipeline.GSHELL_LOADER_LEFT_ROTATION @ saved
            np.testing.assert_allclose(recovered, effective, atol=1e-7)
            self.assertAlmostEqual(np.linalg.norm(saved[:3, 3]), 1.0, places=7)

    def test_split_is_distributed_across_all_five_rings(self) -> None:
        self.assertEqual(len(pipeline.TRAIN_INDICES), 150)
        self.assertEqual(len(pipeline.TEST_INDICES), 30)
        ring_counts = [0] * len(pipeline.ELEVATIONS_DEG)
        for index in pipeline.TEST_INDICES:
            ring_counts[index // pipeline.VIEWS_PER_RING] += 1
        self.assertEqual(ring_counts, [6, 6, 6, 6, 6])

    def test_sugar_camera_conversion_round_trip(self) -> None:
        for index in range(pipeline.VIEW_COUNT):
            _, effective, _ = pipeline.expected_frame(index)
            world_to_camera = pipeline.effective_to_colmap_w2c(effective)
            recovered = pipeline.colmap_w2c_to_effective(world_to_camera)
            np.testing.assert_allclose(recovered, effective, atol=1e-12)

    def test_seed_colmap_model_preserves_all_exact_cameras(self) -> None:
        frames = [
            (f"img{index + 1:03d}.jpg", pipeline.expected_frame(index)[1])
            for index in range(pipeline.VIEW_COUNT)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            pipeline.write_seed_colmap_model(model, frames)
            camera = pipeline.parse_colmap_camera(model / "cameras.txt")
            poses = pipeline.parse_colmap_images(model / "images.txt")

        self.assertEqual(camera["model"], "PINHOLE")
        self.assertEqual((camera["width"], camera["height"]), pipeline.RESOLUTION)
        self.assertEqual(len(poses), pipeline.VIEW_COUNT)
        for name, expected in frames:
            np.testing.assert_allclose(poses[name], expected, atol=1e-12)

    def test_sparse_bbox_rejects_outliers_without_ground_truth_geometry(self) -> None:
        core = np.asarray(
            [[x, y, z] for x in (-0.2, 0.2) for y in (-0.1, 0.1) for z in (-0.05, 0.05)],
            dtype=np.float64,
        )
        points = np.repeat(core, 20, axis=0)
        points = np.vstack((points, [[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]))
        bbox = pipeline.robust_sparse_bbox(points)
        self.assertLessEqual(bbox["points_outside_fraction"], 0.05)
        self.assertLess(bbox["max"][0], 1.0)
        self.assertGreater(bbox["min"][0], -1.0)


if __name__ == "__main__":
    unittest.main()
