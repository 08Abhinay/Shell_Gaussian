from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_tools_blender import pipeline
from dataset_tools_blender.milo import pipeline as milo_pipeline


class MiloDatasetPipelineTest(unittest.TestCase):
    def _source_scene(
        self, root: Path, variant: milo_pipeline.MiloVariant
    ) -> Path:
        source = root / "air_jordan_1"
        source.mkdir()
        frames = []
        for index in variant.indices:
            saved, _, _ = pipeline.expected_frame(index)
            frames.append(
                {
                    "file_path": f"image/img{index + 1:03d}.jpg",
                    "transform_matrix": saved.tolist(),
                }
            )
        (source / "transforms.json").write_text(
            json.dumps({"frames": frames}), encoding="utf-8"
        )
        return source

    def test_full_payload_uses_exact_150_30_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_scene(
                Path(temporary), milo_pipeline.FULL_VARIANT
            )
            frames = milo_pipeline.effective_milo_frames(
                source, milo_pipeline.FULL_VARIANT
            )
        train = milo_pipeline.milo_transform_payload(
            frames, milo_pipeline.FULL_VARIANT.train_indices
        )
        test = milo_pipeline.milo_transform_payload(
            frames, milo_pipeline.FULL_VARIANT.test_indices
        )
        self.assertEqual(len(train["frames"]), 150)
        self.assertEqual(len(test["frames"]), 30)
        self.assertEqual(test["frames"][0]["file_path"], "images/img001")
        self.assertEqual(test["frames"][1]["file_path"], "images/img007")
        self.assertTrue(
            all(not frame["file_path"].endswith(".png") for frame in train["frames"])
        )
        self.assertAlmostEqual(
            train["camera_angle_x"], np.deg2rad(21.0), places=12
        )

    def test_milo_camera_payload_matches_effective_gshell_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_scene(
                Path(temporary), milo_pipeline.TURNTABLE_VARIANT
            )
            frames = milo_pipeline.effective_milo_frames(
                source, milo_pipeline.TURNTABLE_VARIANT
            )
        for index in milo_pipeline.TURNTABLE_VARIANT.indices:
            _, expected, _ = pipeline.expected_frame(index)
            np.testing.assert_allclose(frames[index][1], expected, atol=1e-12)
            milo_opencv_c2w = frames[index][1].copy()
            milo_opencv_c2w[:3, 1:3] *= -1
            np.testing.assert_allclose(
                np.linalg.inv(milo_opencv_c2w),
                pipeline.effective_to_colmap_w2c(expected),
                atol=1e-12,
            )

    def test_milo_ply_has_required_fields_and_round_trips(self) -> None:
        points = np.asarray(
            [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float64
        )
        colors = np.asarray([[1, 2, 3], [253, 254, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points3d.ply"
            milo_pipeline.write_milo_ply(path, points, colors)
            vertices = milo_pipeline.read_milo_ply(path)
        self.assertEqual(
            vertices.dtype.names,
            ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"),
        )
        np.testing.assert_allclose(
            np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
            points,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            np.column_stack(
                (vertices["red"], vertices["green"], vertices["blue"])
            ),
            colors,
        )
        np.testing.assert_array_equal(vertices["nx"], 0.0)

    def test_colmap_point_parser_exposes_training_tracks(self) -> None:
        text = (
            "# 3D point list\n"
            "1 0.1 0.2 0.3 10 20 30 0.4 7 11 9 13\n"
            "2 -0.1 -0.2 -0.3 40 50 60 0.5 9 21\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points3D.txt"
            path.write_text(text, encoding="utf-8")
            points, colors, errors, image_ids = (
                milo_pipeline.parse_colmap_points_detailed(path)
            )
        self.assertEqual(points.shape, (2, 3))
        self.assertEqual(colors.dtype, np.uint8)
        np.testing.assert_allclose(errors, [0.4, 0.5])
        self.assertEqual(image_ids, {7, 9})


if __name__ == "__main__":
    unittest.main()
