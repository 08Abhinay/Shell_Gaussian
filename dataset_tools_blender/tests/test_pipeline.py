from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.NeuralUDF.dataset.dataset import load_K_Rt_from_K_P
from dataset_tools_blender import pipeline
from dataset_tools_blender.gshell import pipeline as gshell_pipeline
from dataset_tools_blender.neuraludf import pipeline as neuraludf_pipeline
from dataset_tools_blender.neus2 import pipeline as neus2_pipeline
from dataset_tools_blender.sugar import pipeline as sugar_pipeline


class DirectBlenderPipelineTest(unittest.TestCase):
    def test_shared_turntable_split_is_36_views_with_six_held_out(self) -> None:
        self.assertEqual(pipeline.TURNTABLE_INDICES, tuple(range(36)))
        self.assertEqual(
            pipeline.TURNTABLE_TEST_INDICES,
            (0, 6, 12, 18, 24, 30),
        )
        self.assertEqual(len(pipeline.TURNTABLE_TRAIN_INDICES), 30)
        self.assertFalse(
            set(pipeline.TURNTABLE_TRAIN_INDICES)
            & set(pipeline.TURNTABLE_TEST_INDICES)
        )
        self.assertEqual(
            set(pipeline.TURNTABLE_TRAIN_INDICES)
            | set(pipeline.TURNTABLE_TEST_INDICES),
            set(pipeline.TURNTABLE_INDICES),
        )

    def test_derived_manifest_identity_is_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_scene = Path(temporary) / "air_jordan_1"
            source_scene.mkdir()
            (source_scene / "transforms.json").write_text(
                '{"frames": []}\n', encoding="utf-8"
            )
            manifest = pipeline.source_manifest_fields(source_scene)

        self.assertEqual(manifest["version"], 2)
        self.assertEqual(
            manifest["source_dataset"],
            "gshell/golden_set_evaluation",
        )
        self.assertEqual(manifest["scene"], "air_jordan_1")
        self.assertNotIn("source_scene", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            source_scene = Path(temporary) / "air_jordan_1"
            source_scene.mkdir()
            (source_scene / "transforms.json").write_text(
                '{"frames": []}\n', encoding="utf-8"
            )
            current = pipeline.source_manifest_fields(source_scene)
            self.assertEqual(
                pipeline.validate_source_manifest(current, source_scene),
                [],
            )
            current["source_scene"] = "/machine-specific/path"
            self.assertIn(
                "manifest contains a deprecated absolute source path",
                pipeline.validate_source_manifest(current, source_scene),
            )

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
            "/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/shoes_size_metadata"
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
            sugar_pipeline.write_seed_colmap_model(model, frames)
            camera = sugar_pipeline.parse_colmap_camera(model / "cameras.txt")
            poses = sugar_pipeline.parse_colmap_images(model / "images.txt")

        self.assertEqual(camera["model"], "PINHOLE")
        self.assertEqual((camera["width"], camera["height"]), pipeline.RESOLUTION)
        self.assertEqual(len(poses), pipeline.VIEW_COUNT)
        for name, expected in frames:
            np.testing.assert_allclose(poses[name], expected, atol=1e-12)

    def test_seed_colmap_model_uses_database_image_ids(self) -> None:
        frames = [
            ("img002.jpg", pipeline.expected_frame(1)[1]),
            ("img003.jpg", pipeline.expected_frame(2)[1]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "database.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE images(image_id INTEGER, name TEXT)"
                )
                connection.executemany(
                    "INSERT INTO images VALUES (?, ?)",
                    [(17, "img002.jpg"), (4, "img003.jpg")],
                )
            image_ids = sugar_pipeline.colmap_database_image_ids(database)
            model = root / "model"
            sugar_pipeline.write_seed_colmap_model(model, frames, image_ids)
            written_ids = sugar_pipeline.parse_colmap_image_ids(
                model / "images.txt"
            )
        self.assertEqual(written_ids, image_ids)

    def test_sugar_turntable_test_cameras_have_no_sparse_tracks(self) -> None:
        train_frames = [
            (f"img{index + 1:03d}.jpg", pipeline.expected_frame(index)[1])
            for index in pipeline.TURNTABLE_TRAIN_INDICES
        ]
        test_frames = [
            (f"img{index + 1:03d}.jpg", pipeline.expected_frame(index)[1])
            for index in pipeline.TURNTABLE_TEST_INDICES
        ]
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            sugar_pipeline.write_seed_colmap_model(model, train_frames)
            sugar_pipeline.append_unobserved_colmap_images(
                model / "images.txt", test_frames
            )
            poses = sugar_pipeline.parse_colmap_images(model / "images.txt")
            image_ids = sugar_pipeline.parse_colmap_image_ids(
                model / "images.txt"
            )
            tracked_ids = sugar_pipeline.point_track_image_ids(
                model / "points3D.txt"
            )

        self.assertEqual(len(poses), 36)
        self.assertEqual(len(image_ids), 36)
        self.assertFalse(tracked_ids)
        for name, expected in [*train_frames, *test_frames]:
            np.testing.assert_allclose(poses[name], expected, atol=1e-12)

    def test_sugar_split_preserves_exact_png_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            for split, indices in (
                ("train", pipeline.TRAIN_INDICES),
                ("test", pipeline.TEST_INDICES),
            ):
                payload = {
                    "frames": [
                        {"file_path": f"image/img{index + 1:03d}.jpg"}
                        for index in indices
                    ]
                }
                (source / f"transforms_{split}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            split_info = sugar_pipeline.write_sugar_splits(
                source, destination
            )
            errors = sugar_pipeline.validate_sugar_splits(destination)

        self.assertEqual(
            split_info, {"train_count": 150, "test_count": 30}
        )
        self.assertEqual(errors, [])

    def test_neuraludf_uses_only_the_existing_training_split(self) -> None:
        scene = pipeline.DEFAULT_OUTPUT_ROOT / "air_jordan_1"
        frames = neuraludf_pipeline.effective_neuraludf_frames(scene)
        self.assertEqual(len(frames), len(pipeline.TRAIN_INDICES))
        self.assertEqual(frames[0][0], "000.png")
        self.assertEqual(frames[0][1], "img002.jpg")
        self.assertEqual(frames[-1][0], "149.png")
        for (_, source_name, actual), source_index in zip(frames, pipeline.TRAIN_INDICES):
            self.assertEqual(source_name, f"img{source_index + 1:03d}.jpg")
            expected = pipeline.expected_frame(source_index)[1]
            np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_neuraludf_turntable_uses_only_30_level_training_views(self) -> None:
        scene = pipeline.DEFAULT_OUTPUT_ROOT / "air_jordan_1"
        frames = neuraludf_pipeline.effective_neuraludf_turntable_frames(scene)
        self.assertEqual(len(frames), 30)
        for output_index, ((output_name, source_name, actual), source_index) in enumerate(
            zip(frames, pipeline.TURNTABLE_TRAIN_INDICES)
        ):
            self.assertEqual(output_name, f"{output_index:03d}.png")
            self.assertEqual(source_name, f"img{source_index + 1:03d}.jpg")
            np.testing.assert_allclose(
                actual,
                pipeline.expected_frame(source_index)[1],
                atol=1e-7,
            )

    def test_gshell_turntable_protocol_does_not_include_reference_mesh(self) -> None:
        self.assertIn("turntable_36", gshell_pipeline.GSHELL_TURNTABLE_PROTOCOL)

    def test_neuraludf_camera_matrices_follow_idr_contract(self) -> None:
        effective = pipeline.expected_frame(17)[1]
        scale = np.diag([0.2, 0.2, 0.2, 1.0])
        scale[:3, 3] = [0.01, -0.02, 0.03]
        matrices = neuraludf_pipeline.neuraludf_camera_matrices(effective, scale)
        self.assertEqual(
            set(matrices),
            {
                "camera_mat",
                "camera_mat_inv",
                "world_mat",
                "world_mat_inv",
                "scale_mat",
                "scale_mat_inv",
            },
        )
        np.testing.assert_allclose(
            matrices["camera_mat"] @ matrices["camera_mat_inv"], np.eye(4), atol=1e-4
        )
        np.testing.assert_allclose(
            matrices["world_mat"] @ matrices["world_mat_inv"], np.eye(4), atol=1e-4
        )
        expected_pose = neuraludf_pipeline.normalized_neuraludf_pose(effective, scale)
        rigid_projection = (
            neuraludf_pipeline.neuraludf_intrinsic() @ np.linalg.inv(expected_pose)
        )
        projective_projection = matrices["world_mat"] @ matrices["scale_mat"]
        projective_w2c = np.linalg.inv(matrices["camera_mat"]) @ projective_projection
        projective_scale = np.linalg.norm(projective_w2c[:3, :3], axis=1).mean()
        np.testing.assert_allclose(
            projective_projection[:3, :4] / projective_scale,
            rigid_projection[:3, :4],
            rtol=2e-7,
            atol=1e-3,
        )
        _, recovered_pose = load_K_Rt_from_K_P(
            matrices["camera_mat"], projective_projection
        )
        np.testing.assert_allclose(recovered_pose, expected_pose, atol=1e-5)
        rotation = recovered_pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)

    def test_neuraludf_loader_recovers_rigid_unit_rays_for_every_pose(self) -> None:
        scale = np.diag([0.137, 0.137, 0.137, 1.0])
        scale[:3, 3] = [0.031, -0.047, 0.019]
        intrinsic = neuraludf_pipeline.neuraludf_intrinsic().astype(np.float32)
        indices = [0, *pipeline.TRAIN_INDICES]
        for index in indices:
            effective = pipeline.expected_frame(index)[1]
            matrices = neuraludf_pipeline.neuraludf_camera_matrices(effective, scale)
            _, pose = load_K_Rt_from_K_P(
                intrinsic, matrices["world_mat"] @ matrices["scale_mat"]
            )
            expected = neuraludf_pipeline.normalized_neuraludf_pose(effective, scale)
            np.testing.assert_allclose(pose, expected, atol=1e-5)

            rotation = pose[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)
            center_ray = rotation @ np.array([0.0, 0.0, 1.0])
            self.assertAlmostEqual(float(np.linalg.norm(center_ray)), 1.0, places=5)

    def test_neus2_uses_exact_opencv_cameras_and_existing_split(self) -> None:
        scene = pipeline.DEFAULT_OUTPUT_ROOT / "air_jordan_1"
        frames = neus2_pipeline.effective_neus2_frames(scene)
        self.assertEqual(len(frames), pipeline.VIEW_COUNT)
        for index, source_name, effective, opencv in frames:
            self.assertEqual(source_name, f"img{index + 1:03d}.jpg")
            np.testing.assert_allclose(
                effective,
                pipeline.expected_frame(index)[1],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                opencv,
                effective @ pipeline.OPENGL_TO_OPENCV_CAMERA,
                atol=1e-12,
            )
            rotation = opencv[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)

        scale = 2.5
        offset = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        train = neus2_pipeline.neus2_transform_payload(
            frames,
            pipeline.TRAIN_INDICES,
            scale,
            offset,
        )
        test = neus2_pipeline.neus2_transform_payload(
            frames,
            pipeline.TEST_INDICES,
            scale,
            offset,
        )
        self.assertEqual(len(train["frames"]), 150)
        self.assertEqual(len(test["frames"]), 30)
        self.assertTrue(train["from_na"])
        self.assertEqual(train["scale"], scale)
        self.assertEqual(train["offset"], offset.tolist())
        train_indices = {frame["source_view_index"] for frame in train["frames"]}
        test_indices = {frame["source_view_index"] for frame in test["frames"]}
        self.assertFalse(train_indices & test_indices)
        self.assertEqual(train_indices | test_indices, set(range(pipeline.VIEW_COUNT)))

    def test_neus2_visual_hull_sphere_maps_to_unit_cube(self) -> None:
        sphere = np.diag([0.2, 0.2, 0.2, 1.0])
        sphere[:3, 3] = [0.03, -0.04, 0.01]
        scale, offset = neus2_pipeline.neus2_scale_offset(sphere)
        np.testing.assert_allclose(scale * sphere[:3, 3] + offset, 0.5, atol=1e-12)
        self.assertAlmostEqual(scale * sphere[0, 0], 0.5, places=12)

        nonuniform = sphere.copy()
        nonuniform[1, 1] = 0.3
        with self.assertRaisesRegex(ValueError, "must be uniform"):
            neus2_pipeline.neus2_scale_offset(nonuniform)

    def test_neus2_turntable_matches_the_existing_36_view_split(self) -> None:
        self.assertEqual(
            neus2_pipeline.TURNTABLE_INDICES,
            tuple(range(36)),
        )
        self.assertEqual(
            neus2_pipeline.TURNTABLE_TEST_INDICES,
            (0, 6, 12, 18, 24, 30),
        )
        self.assertEqual(len(neus2_pipeline.TURNTABLE_TRAIN_INDICES), 30)
        self.assertFalse(
            set(neus2_pipeline.TURNTABLE_TRAIN_INDICES)
            & set(neus2_pipeline.TURNTABLE_TEST_INDICES)
        )

        scene = pipeline.DEFAULT_OUTPUT_ROOT / "air_jordan_1"
        frames = neus2_pipeline.effective_neus2_frames(scene)
        scale = 2.5
        offset = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        all_views = neus2_pipeline.neus2_transform_payload(
            frames,
            neus2_pipeline.TURNTABLE_INDICES,
            scale,
            offset,
        )
        train = neus2_pipeline.neus2_transform_payload(
            frames,
            neus2_pipeline.TURNTABLE_TRAIN_INDICES,
            scale,
            offset,
        )
        test = neus2_pipeline.neus2_transform_payload(
            frames,
            neus2_pipeline.TURNTABLE_TEST_INDICES,
            scale,
            offset,
        )
        self.assertEqual(len(all_views["frames"]), 36)
        self.assertEqual(len(train["frames"]), 30)
        self.assertEqual(len(test["frames"]), 6)
        self.assertEqual(
            [frame["file_path"] for frame in test["frames"]],
            [
                "images/img001.png",
                "images/img007.png",
                "images/img013.png",
                "images/img019.png",
                "images/img025.png",
                "images/img031.png",
            ],
        )
        train_indices = {
            frame["source_view_index"] for frame in train["frames"]
        }
        test_indices = {
            frame["source_view_index"] for frame in test["frames"]
        }
        self.assertEqual(
            train_indices | test_indices,
            set(neus2_pipeline.TURNTABLE_INDICES),
        )

    def test_custom_shoe_configs_use_masked_white_contract(self) -> None:
        config_root = Path("baselines/NeuralUDF/confs")
        config_names = (
            "udf_shoes_smoke.conf",
            "udf_shoes.conf",
            "udf_shoes_dtu_probe.conf",
            "udf_shoes_garment_probe.conf",
        )
        for config_name in config_names:
            text = (config_root / config_name).read_text(encoding="utf-8")
            self.assertIn("use_white_bkgd = True", text)
            self.assertIn("mask_weight = 0.1", text)
            self.assertIn("n_outside = 0", text)
            self.assertIn("masked_white", text)
        garment = (config_root / "udf_shoes_garment_probe.conf").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("udf_shift", garment)
        self.assertNotIn("predict_grad", garment)

    def test_sparse_bbox_rejects_outliers_without_ground_truth_geometry(self) -> None:
        core = np.asarray(
            [[x, y, z] for x in (-0.2, 0.2) for y in (-0.1, 0.1) for z in (-0.05, 0.05)],
            dtype=np.float64,
        )
        points = np.repeat(core, 20, axis=0)
        points = np.vstack((points, [[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]))
        bbox = sugar_pipeline.robust_sparse_bbox(points)
        self.assertLessEqual(bbox["points_outside_fraction"], 0.05)
        self.assertLess(bbox["max"][0], 1.0)
        self.assertGreater(bbox["min"][0], -1.0)


if __name__ == "__main__":
    unittest.main()
