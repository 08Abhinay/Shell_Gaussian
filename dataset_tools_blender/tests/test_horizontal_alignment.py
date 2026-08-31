from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_tools_blender import core
from dataset_tools_blender.horizontal_alignment import (
    METHOD,
    estimate_horizontal_alignment,
    validate_horizontal_alignment_config,
    validate_horizontal_alignment_metadata,
)


def gridded_plate(
    length: float,
    width: float,
    angle_degrees: float,
    *,
    nx: int = 20,
    ny: int = 8,
    height_marker: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-0.5 * length, 0.5 * length, nx + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1)
    vertices = np.asarray(
        [(x, y, 0.0) for x in xs for y in ys], dtype=np.float64
    )
    faces = []
    for ix in range(nx):
        for iy in range(ny):
            lower_left = ix * (ny + 1) + iy
            lower_right = (ix + 1) * (ny + 1) + iy
            faces.extend(
                [
                    (lower_left, lower_right, lower_right + 1),
                    (lower_left, lower_right + 1, lower_left + 1),
                ]
            )
    vertices = np.vstack([vertices, [0.0, 0.0, height_marker]])
    angle = math.radians(angle_degrees)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return vertices @ rotation.T, np.asarray(faces, dtype=np.int64)


class HorizontalAlignmentTest(unittest.TestCase):
    def test_recovers_positive_and_negative_heading(self) -> None:
        for angle in (20.0, -15.0):
            with self.subTest(angle=angle):
                vertices, faces = gridded_plate(4.0, 1.0, angle)
                result = estimate_horizontal_alignment(vertices, faces)
                self.assertAlmostEqual(
                    result.measured_angle_degrees, angle, delta=0.05
                )
                self.assertAlmostEqual(
                    result.correction_angle_degrees, -angle, delta=0.05
                )
                self.assertAlmostEqual(result.residual_angle_degrees, 0.0, places=10)
                self.assertGreater(result.direction_xy[0], 0.0)

    def test_rotation_is_rigid_and_preserves_topology(self) -> None:
        vertices, faces = gridded_plate(4.0, 1.0, 12.0)
        original_faces = faces.copy()
        result = estimate_horizontal_alignment(vertices, faces)
        np.testing.assert_allclose(
            result.rotation_3x3.T @ result.rotation_3x3,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(float(np.linalg.det(result.rotation_3x3)), 1.0)
        np.testing.assert_array_equal(faces, original_faces)

    def test_tall_shaft_does_not_control_heading(self) -> None:
        vertices, faces = gridded_plate(
            4.0, 1.0, 17.0, height_marker=5.0
        )
        shaft = np.array(
            [
                [-0.4, -2.0, 1.5],
                [-0.4, 2.0, 1.5],
                [-0.4, -2.0, 5.0],
                [-0.4, 2.0, 5.0],
            ],
            dtype=np.float64,
        )
        shaft_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        offset = len(vertices)
        vertices = np.vstack([vertices, shaft])
        faces = np.vstack([faces, shaft_faces + offset])
        result = estimate_horizontal_alignment(vertices, faces)
        self.assertAlmostEqual(
            result.measured_angle_degrees, 17.0, delta=0.05
        )

    def test_rejects_ambiguous_square_geometry(self) -> None:
        vertices, faces = gridded_plate(2.0, 2.0, 10.0, nx=16, ny=16)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            estimate_horizontal_alignment(vertices, faces)

    def test_rejects_degenerate_geometry(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            estimate_horizontal_alignment(
                vertices, np.array([[0, 1, 2]], dtype=np.int64)
            )

    def test_rejects_correction_inconsistent_with_reviewed_axes(self) -> None:
        vertices, faces = gridded_plate(4.0, 1.0, 50.0)
        with self.assertRaisesRegex(ValueError, "too large"):
            estimate_horizontal_alignment(vertices, faces)

    def test_validates_configuration_strictly(self) -> None:
        config = {
            "method": METHOD,
            "lower_height_fraction": 0.2,
            "minimum_axis_ratio": 2.0,
            "maximum_abs_angle_degrees": 45.0,
        }
        validate_horizontal_alignment_config(config)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_horizontal_alignment_config({**config, "method": "pca"})

    def test_validates_recorded_transform_metadata(self) -> None:
        vertices, faces = gridded_plate(4.0, 1.0, 12.0)
        result = estimate_horizontal_alignment(vertices, faces)
        config = {
            "method": METHOD,
            "lower_height_fraction": 0.2,
            "minimum_axis_ratio": 2.0,
            "maximum_abs_angle_degrees": 45.0,
        }
        canonical = {
            "horizontal_alignment": result.to_dict(),
            "source_to_canonical_matrix": np.eye(4).tolist(),
            "canonical_bbox_min": [-2.0, -1.0, -0.5],
            "canonical_bbox_max": [2.0, 1.0, 0.5],
        }
        self.assertEqual(
            validate_horizontal_alignment_metadata(canonical, config), []
        )
        canonical["horizontal_alignment"]["residual_angle_degrees"] = 0.2
        self.assertIn(
            "horizontal residual exceeds 0.1 degrees",
            validate_horizontal_alignment_metadata(canonical, config),
        )


class HeadingManifestTest(unittest.TestCase):
    def test_golden_manifest_has_all_nineteen_explicit_profiles(self) -> None:
        root = Path(
            "/home/ab5298/dataset/datasets/external/"
            "golden_set_eval_glb/curated_subsets/footbed_clean"
        )
        manifest = core.SCRIPT_DIR / "golden_set_evaluation_manifest.json"
        records = core.load_manifest(manifest, root)

        self.assertEqual(len(records), 19)
        self.assertEqual(
            sum(record["shoe_profile"] == "normal" for record in records),
            17,
        )
        self.assertEqual(
            {
                record["name"]
                for record in records
                if record["shoe_profile"] == "high_heel"
            },
            {"plateau_sandal_heels", "red_high_heel_shoes"},
        )

    def manifest_payload(
        self, model: Path, inventory_policy: str
    ) -> dict[str, object]:
        return {
            "version": 1,
            "inventory_policy": inventory_policy,
            "horizontal_alignment": {
                "method": METHOD,
                "lower_height_fraction": 0.2,
                "minimum_axis_ratio": 2.0,
                "maximum_abs_angle_degrees": 45.0,
            },
            "shoes": [
                {
                    "name": "pilot_shoe",
                    "model": model.name,
                    "sha256": core.sha256_file(model),
                    "reviewed": True,
                    "shoe_profile": "normal",
                    "source_axes": {
                        "length": "X",
                        "width": "Y",
                        "up": "Z",
                    },
                    "selection": {"mode": "all"},
                    "mirror_width": False,
                }
            ],
        }

    def test_listed_subset_ignores_future_raw_glbs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pilot_shoe.glb"
            model.write_bytes(b"pilot")
            (root / "future_shoe.glb").write_bytes(b"future")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(self.manifest_payload(model, "listed_subset")),
                encoding="utf-8",
            )
            records = core.load_manifest(manifest, root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["horizontal_alignment"]["method"], METHOD)

    def test_exact_inventory_still_rejects_extra_glbs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pilot_shoe.glb"
            model.write_bytes(b"pilot")
            (root / "future_shoe.glb").write_bytes(b"future")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(self.manifest_payload(model, "exact")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Manifest/source mismatch"):
                core.load_manifest(manifest, root)

    def test_listed_file_must_exist_and_match_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pilot_shoe.glb"
            model.write_bytes(b"pilot")
            payload = self.manifest_payload(model, "listed_subset")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                core.load_manifest(manifest, root)
            model.unlink()
            with self.assertRaises(FileNotFoundError):
                core.load_manifest(manifest, root)

    def test_rejects_unknown_shoe_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pilot_shoe.glb"
            model.write_bytes(b"pilot")
            payload = self.manifest_payload(model, "listed_subset")
            payload["shoes"][0]["shoe_profile"] = "heel"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid shoe_profile"):
                core.load_manifest(manifest, root)

    def test_rejects_partially_tagged_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "pilot_shoe.glb"
            second = root / "second_shoe.glb"
            first.write_bytes(b"pilot")
            second.write_bytes(b"second")
            payload = self.manifest_payload(first, "exact")
            second_record = dict(payload["shoes"][0])
            second_record.update(
                {
                    "name": "second_shoe",
                    "model": second.name,
                    "sha256": core.sha256_file(second),
                }
            )
            second_record.pop("shoe_profile")
            payload["shoes"].append(second_record)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tagging is incomplete"):
                core.load_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
