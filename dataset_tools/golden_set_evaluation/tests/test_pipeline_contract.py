from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_tools.golden_set_evaluation import pipeline
from dataset_tools.golden_set_evaluation.align_colmap_masks import compact_scene
from dataset_tools.golden_set_evaluation.gshell_adapter import resolve_scene_layout


class PipelineContractTest(unittest.TestCase):
    def test_glb_discovery_normalizes_names_and_ignores_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Shoe Name .glb").touch()
            (root / "render_manifest.json").touch()

            self.assertEqual(
                pipeline.discover_glbs(root),
                [{"name": "shoe_name", "model": "Shoe Name .glb", "source_axes": "auto"}],
            )

    def test_raw_scene_accepts_only_the_numbered_image_mask_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            image_dir = scene / "images"
            mask_dir = scene / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            for index in range(1, pipeline.EXPECTED_VIEW_COUNT + 1):
                (image_dir / f"img{index:03d}.jpg").touch()
                (mask_dir / f"img{index:03d}.png").touch()

            self.assertEqual(pipeline.validate_raw_scene(scene), [])
            (scene / "transforms.json").touch()
            self.assertIn("unexpected raw artifacts", pipeline.validate_raw_scene(scene)[0])

    def test_gshell_adapter_prefers_compact_layout_and_supports_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            for directory in (
                compact / "undistorted" / "images",
                compact / "undistorted" / "masks",
                compact / "undistorted" / "sparse" / "0",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            model = compact / "undistorted" / "sparse" / "0"
            (model / "cameras.txt").touch()
            (model / "images.txt").touch()
            self.assertEqual(resolve_scene_layout(compact).name, "compact_undistorted")

            legacy = root / "legacy"
            for directory in (legacy / "images", legacy / "masks", legacy / "colmap"):
                directory.mkdir(parents=True, exist_ok=True)
            (legacy / "colmap" / "cameras.txt").touch()
            (legacy / "colmap" / "images.txt").touch()
            self.assertEqual(resolve_scene_layout(legacy).name, "legacy_raw_colmap")

    def test_compaction_removes_only_redundant_processed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            for directory in (
                scene / "images",
                scene / "masks",
                scene / "colmap",
                scene / "undistorted" / "stereo",
                scene / "undistorted" / "sparse" / "0",
                scene / "undistorted" / "images",
                scene / "undistorted" / "masks",
                scene / "logs",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (scene / "undistorted" / "sparse" / "0" / "points3D.ply").touch()
            (scene / "undistorted" / "sparse" / "0" / "points3D.txt").touch()

            compact_scene(scene)

            self.assertFalse((scene / "images").exists())
            self.assertFalse((scene / "masks").exists())
            self.assertFalse((scene / "colmap").exists())
            self.assertFalse((scene / "undistorted" / "stereo").exists())
            self.assertFalse((scene / "undistorted" / "sparse" / "0" / "points3D.ply").exists())
            self.assertTrue((scene / "undistorted" / "sparse" / "0" / "points3D.txt").exists())
            self.assertTrue((scene / "logs").is_dir())


if __name__ == "__main__":
    unittest.main()
