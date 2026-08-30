"""Shared direct-Blender dataset building and validation primitives."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_tools_blender.horizontal_alignment import (
    validate_horizontal_alignment_config,
    validate_horizontal_alignment_metadata,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "gshell/golden_set_evaluation"
)
DEFAULT_SUGAR_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "sugar/golden_set_evaluation"
)
DEFAULT_NEURALUDF_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "neuraludf/golden_set_evaluation"
)
DEFAULT_NEUS2_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "neus2/golden_set_evaluation"
)
DEFAULT_GSHELL_TURNTABLE_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "gshell/golden_set_evaluation_turntable"
)
DEFAULT_NEURALUDF_TURNTABLE_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "neuraludf/golden_set_evaluation_turntable"
)
DEFAULT_SUGAR_TURNTABLE_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "sugar/golden_set_evaluation_turntable"
)
DEFAULT_BLENDER = Path(
    "/storage/Abhinay/home_ab5298/anaconda3/envs/"
    "shellgaussianenv/bin/blender"
)
DEFAULT_COLMAP = Path("/storage/Abhinay/conda_envs/colmap/bin/colmap")
DEFAULT_MANIFEST = SCRIPT_DIR / "evaluation_manifest.json"

RESOLUTION = (1536, 1024)
FOV_X_DEG = 21.0
CAMERA_RADIUS = 1.0
ELEVATIONS_DEG = (0.0, -25.0, 20.0, 45.0, 65.0)
VIEWS_PER_RING = 36
VIEW_COUNT = len(ELEVATIONS_DEG) * VIEWS_PER_RING
TEST_STRIDE = 6
TEST_INDICES = tuple(range(0, VIEW_COUNT, TEST_STRIDE))
TRAIN_INDICES = tuple(
    index for index in range(VIEW_COUNT) if index not in TEST_INDICES
)
TURNTABLE_INDICES = tuple(range(VIEWS_PER_RING))
TURNTABLE_TEST_INDICES = tuple(range(0, VIEWS_PER_RING, TEST_STRIDE))
TURNTABLE_TRAIN_INDICES = tuple(
    index
    for index in TURNTABLE_INDICES
    if index not in TURNTABLE_TEST_INDICES
)
MIN_INVDEPTH_MASK_IOU = 0.98
SUGAR_PROTOCOL = "exact_blender_cameras_colmap_triangulation_v1"
SUGAR_CAMERA_ATOL = 1e-6
SUGAR_BBOX_LOW_QUANTILE = 0.01
SUGAR_BBOX_HIGH_QUANTILE = 0.99
SUGAR_BBOX_MARGIN = 0.25
NEURALUDF_GRID_RESOLUTION = 96
NEURALUDF_SEARCH_HALF_EXTENT = 0.25
NEURALUDF_SCALE_MARGIN = 1.1
NEURALUDF_CAMERA_ATOL = 5e-6
NEUS2_PROTOCOL = "exact_blender_cameras_visual_hull_normalization_v1"
NEUS2_CAMERA_ATOL = 5e-6
OPENGL_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
DERIVED_MANIFEST_VERSION = 2
SOURCE_DATASET_ID = "gshell/golden_set_evaluation"


def rotation_x(angle_rad: float) -> np.ndarray:
    sine, cosine = math.sin(angle_rad), math.cos(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, sine, 0.0],
            [0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


BLENDER_TO_EFFECTIVE_GSHELL = rotation_x(-math.pi / 2.0)
GSHELL_LOADER_LEFT_ROTATION = rotation_x(math.pi / 2.0)
BLENDER_TO_SAVED_GSHELL = rotation_x(-math.pi)


def orbit_eye(
    radius: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return np.array(
        [
            radius * math.cos(azimuth) * cos_elevation,
            radius * math.sin(azimuth) * cos_elevation,
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def c2w_from_eye(eye: np.ndarray) -> np.ndarray:
    forward = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def expected_frame(
    index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    ring_index, azimuth_index = divmod(index, VIEWS_PER_RING)
    elevation_deg = ELEVATIONS_DEG[ring_index]
    azimuth_deg = -90.0 + 10.0 * azimuth_index
    blender_c2w = c2w_from_eye(
        orbit_eye(CAMERA_RADIUS, azimuth_deg, elevation_deg)
    )
    saved_c2w = BLENDER_TO_SAVED_GSHELL @ blender_c2w
    effective_c2w = BLENDER_TO_EFFECTIVE_GSHELL @ blender_c2w
    metadata: dict[str, float | int] = {
        "ring_index": ring_index,
        "azimuth_index": azimuth_index,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
    }
    return saved_c2w, effective_c2w, metadata


def normalized_name(model_name: str) -> str:
    name = re.sub(
        r"[^a-z0-9]+", "_", Path(model_name).stem.lower()
    ).strip("_")
    if not name:
        raise ValueError(
            f"Cannot derive a scene name from {model_name!r}"
        )
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_fields(source_scene: Path) -> dict[str, Any]:
    """Return portable identity fields for a derived dataset manifest."""
    return {
        "version": DERIVED_MANIFEST_VERSION,
        "scene": source_scene.name,
        "source_dataset": SOURCE_DATASET_ID,
        "source_transforms_sha256": sha256_file(
            source_scene / "transforms.json"
        ),
    }


def validate_source_manifest(
    manifest: dict[str, Any], source_scene: Path
) -> list[str]:
    """Validate source identity without comparing absolute filesystem paths."""
    errors: list[str] = []
    if manifest.get("version") != DERIVED_MANIFEST_VERSION:
        errors.append("incorrect derived manifest version")
    if "source_scene" in manifest:
        errors.append("manifest contains a deprecated absolute source path")
    if manifest.get("source_dataset") != SOURCE_DATASET_ID:
        errors.append("incorrect source dataset identifier")
    if manifest.get("scene") != source_scene.name:
        errors.append("source scene name does not match")
    source_transforms = source_scene / "transforms.json"
    if not source_transforms.is_file():
        errors.append("source transforms.json is missing")
    elif manifest.get("source_transforms_sha256") != sha256_file(
        source_transforms
    ):
        errors.append("source transforms.json changed after preparation")
    return errors


def load_manifest(
    path: Path,
    source_root: Path,
    verify_hashes: bool = True,
    require_reviewed: bool = True,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(
        payload.get("shoes"), list
    ):
        raise ValueError(f"Unsupported manifest schema: {path}")

    inventory_policy = payload.get("inventory_policy", "exact")
    if inventory_policy not in {"exact", "listed_subset"}:
        raise ValueError(
            f"Unsupported inventory_policy: {inventory_policy!r}"
        )
    horizontal_alignment = payload.get("horizontal_alignment")
    if horizontal_alignment is not None:
        validate_horizontal_alignment_config(horizontal_alignment)

    records: list[dict[str, Any]] = []
    names: set[str] = set()
    models: set[str] = set()
    axis_tokens = {"X", "Y", "Z", "-X", "-Y", "-Z"}
    for raw_record in payload["shoes"]:
        record = dict(raw_record)
        if "horizontal_alignment" in record:
            raise ValueError(
                "horizontal_alignment is a manifest-level setting, not a "
                f"per-shoe setting: {record.get('name', '')}"
            )
        if horizontal_alignment is not None:
            record["horizontal_alignment"] = dict(horizontal_alignment)
        name = str(record.get("name", ""))
        model = str(record.get("model", ""))
        if name != normalized_name(model):
            raise ValueError(
                f"Manifest name/model mismatch: {name!r}, {model!r}"
            )
        if name in names or model in models:
            raise ValueError(f"Duplicate manifest entry: {name}")
        if require_reviewed and record.get("reviewed") is not True:
            raise ValueError(f"Production entry is not reviewed: {name}")
        axes = record.get("source_axes")
        if (
            not isinstance(axes, dict)
            or set(axes) != {"length", "width", "up"}
        ):
            raise ValueError(f"Invalid source_axes for {name}")
        tokens = [
            str(axes[key]) for key in ("length", "width", "up")
        ]
        if any(token not in axis_tokens for token in tokens):
            raise ValueError(
                f"Invalid source axis token for {name}: {tokens}"
            )
        if len({token[-1] for token in tokens}) != 3:
            raise ValueError(
                f"Source axes are not orthogonal for {name}: {tokens}"
            )
        selection = record.get("selection", {"mode": "all"})
        if selection.get("mode") not in {"all", "axis-side"}:
            raise ValueError(f"Invalid selection mode for {name}")
        if selection.get("mode") == "axis-side":
            if selection.get("axis") not in {"X", "Y", "Z"}:
                raise ValueError(
                    f"Invalid selection axis for {name}"
                )
            if selection.get("side") not in {"min", "max"}:
                raise ValueError(
                    f"Invalid selection side for {name}"
                )
        model_path = source_root / model
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if verify_hashes:
            actual_hash = sha256_file(model_path)
            if actual_hash != record.get("sha256"):
                raise ValueError(
                    f"GLB checksum changed for {name}: expected "
                    f"{record.get('sha256')}, got {actual_hash}"
                )
        names.add(name)
        models.add(model)
        records.append(record)

    source_models = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.glb")
        if path.is_file()
    }
    inventory_mismatch = (
        source_models != models
        if inventory_policy == "exact"
        else not models.issubset(source_models)
    )
    if inventory_mismatch:
        missing = sorted(models - source_models)
        unreviewed = sorted(source_models - models)
        raise ValueError(
            "Manifest/source mismatch; "
            f"missing={missing}, unreviewed={unreviewed}"
        )
    return sorted(records, key=lambda record: str(record["name"]))


def selected_records(
    records: list[dict[str, Any]],
    shoe: str | None,
    all_shoes: bool,
) -> list[dict[str, Any]]:
    if all_shoes:
        return records
    available = {
        str(record["name"]): record for record in records
    }
    if shoe not in available:
        raise ValueError(
            f"Unknown shoe {shoe!r}; available: "
            f"{', '.join(sorted(available))}"
        )
    return [available[str(shoe)]]


def numbered_names(folder: str, extension: str) -> set[str]:
    del folder
    return {
        f"img{index:03d}.{extension}"
        for index in range(1, VIEW_COUNT + 1)
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_array(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"))
    return image > 127


def copy_sparse_npy(source: Path, destination: Path) -> None:
    """Copy a dense-shape NPY without allocating its zero background pages."""
    values = np.load(source, mmap_mode="r")
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError(f"Expected a 2D float32 inverse-depth array: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=values.shape,
    )
    for row in np.flatnonzero(np.any(values != 0.0, axis=1)):
        columns = np.flatnonzero(values[row] != 0.0)
        start, stop = int(columns[0]), int(columns[-1]) + 1
        mapped[row, start:stop] = values[row, start:stop]
    mapped.flush()
    del mapped


def inverse_depth_mask_iou(
    mask: np.ndarray, inverse_depth: np.ndarray
) -> float:
    depth_mask = np.isfinite(inverse_depth) & (inverse_depth > 0.0)
    intersection = np.logical_and(mask, depth_mask).sum()
    union = np.logical_or(mask, depth_mask).sum()
    return float(intersection / union) if union else 1.0


def ply_counts(path: Path) -> tuple[int, int]:
    vertices = faces = None
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode(
                "ascii", errors="strict"
            ).strip()
            if line.startswith("element vertex "):
                vertices = int(line.rsplit(" ", 1)[1])
            elif line.startswith("element face "):
                faces = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
    if not vertices or not faces:
        raise ValueError(f"Invalid or empty PLY mesh: {path}")
    return vertices, faces


def validate_frame_payload(
    frames: list[dict[str, Any]],
    expected_indices: tuple[int, ...],
    scene: Path,
) -> list[str]:
    del scene
    errors: list[str] = []
    if len(frames) != len(expected_indices):
        return [
            f"expected {len(expected_indices)} frames, found {len(frames)}"
        ]
    for frame, index in zip(frames, expected_indices):
        saved_expected, effective_expected, metadata = expected_frame(index)
        expected_name = f"img{index + 1:03d}.jpg"
        if frame.get("file_path") != f"image/{expected_name}":
            errors.append(
                f"frame {index}: unexpected file_path "
                f"{frame.get('file_path')!r}"
            )
        if (
            frame.get("invdepth_path")
            != f"invdepth/img{index + 1:03d}.npy"
        ):
            errors.append(
                f"frame {index}: unexpected invdepth_path"
            )
        if not math.isclose(
            float(frame.get("camera_angle_x", -1.0)),
            math.radians(FOV_X_DEG),
            abs_tol=1e-10,
        ):
            errors.append(
                f"frame {index}: incorrect horizontal FOV"
            )
        for key, expected_value in metadata.items():
            actual = frame.get(key)
            if isinstance(expected_value, float):
                if not math.isclose(
                    float(actual), expected_value, abs_tol=1e-9
                ):
                    errors.append(f"frame {index}: incorrect {key}")
            elif actual != expected_value:
                errors.append(f"frame {index}: incorrect {key}")
        matrix = np.asarray(
            frame.get("transform_matrix"), dtype=np.float64
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append(
                f"frame {index}: transform_matrix is not finite 4x4"
            )
            continue
        if not np.allclose(matrix, saved_expected, atol=1e-7):
            errors.append(
                f"frame {index}: saved camera does not match "
                "the deterministic orbit"
            )
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6
        ):
            errors.append(
                f"frame {index}: camera rotation is not orthonormal"
            )
        if not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
        ):
            errors.append(
                f"frame {index}: camera rotation determinant is not one"
            )
        if not math.isclose(
            float(np.linalg.norm(matrix[:3, 3])),
            CAMERA_RADIUS,
            abs_tol=1e-7,
        ):
            errors.append(f"frame {index}: camera radius is not one")
        loader_effective = GSHELL_LOADER_LEFT_ROTATION @ matrix
        if not np.allclose(
            loader_effective, effective_expected, atol=1e-7
        ):
            errors.append(
                f"frame {index}: GShell loader would recover "
                "the wrong camera"
            )
    return errors


def validate_scene(
    scene: Path,
    validate_pixels: bool = True,
    expected_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_files = {
        "image": numbered_names("image", "jpg"),
        "mask": numbered_names("mask", "png"),
        "invdepth": numbered_names("invdepth", "npy"),
    }
    for folder, expected in expected_files.items():
        directory = scene / folder
        actual = (
            {path.name for path in directory.iterdir()}
            if directory.is_dir()
            else set()
        )
        if actual != expected:
            errors.append(
                f"{folder}: missing={sorted(expected - actual)[:5]}, "
                f"unexpected={sorted(actual - expected)[:5]}"
            )

    payload_specs = (
        ("transforms.json", tuple(range(VIEW_COUNT))),
        ("transforms_train.json", TRAIN_INDICES),
        ("transforms_test.json", TEST_INDICES),
    )
    for filename, indices in payload_specs:
        path = scene / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        payload = read_json(path)
        if (
            payload.get("pose_convention")
            != "legacy_gshell_saved_c2w_for_fixed_loader"
        ):
            errors.append(f"{filename}: incorrect pose_convention")
        errors.extend(
            f"{filename}: {error}"
            for error in validate_frame_payload(
                payload.get("frames", []), indices, scene
            )
        )

    metadata_path = scene / "blender_canonicalization.json"
    if not metadata_path.is_file():
        errors.append("missing blender_canonicalization.json")
        metadata: dict[str, Any] = {}
    else:
        metadata = read_json(metadata_path)
        camera = metadata.get("camera_contract", {})
        if camera.get("view_count") != VIEW_COUNT:
            errors.append(
                "canonicalization metadata has the wrong view count"
            )
        if camera.get("radius") != CAMERA_RADIUS:
            errors.append(
                "canonicalization metadata has the wrong radius"
            )
        projection = metadata.get("reference_mesh_projection", {})
        if not projection.get("passed", False):
            errors.append(
                "reference mesh projection validation did not pass"
            )
        expected_alignment = (
            expected_record.get("horizontal_alignment")
            if expected_record is not None
            else None
        )
        if expected_alignment is not None:
            canonical = metadata.get("canonical_geometry", {})
            errors.extend(
                validate_horizontal_alignment_metadata(
                    canonical, expected_alignment
                )
            )

    mesh_path = scene / "reference_mesh.ply"
    if not mesh_path.is_file():
        errors.append("missing reference_mesh.ply")
    else:
        try:
            vertex_count, face_count = ply_counts(mesh_path)
            mesh_metadata = metadata.get("reference_mesh", {})
            if (
                expected_record is not None
                and expected_record.get("horizontal_alignment") is not None
                and mesh_metadata.get("coordinate_system")
                != "effective_gshell_x_length_y_down_z_width"
            ):
                errors.append(
                    "reference mesh has the wrong coordinate system"
                )
            if (
                vertex_count != mesh_metadata.get("vertices")
                or face_count != mesh_metadata.get("faces")
            ):
                errors.append(
                    "reference mesh counts do not match "
                    "canonicalization metadata"
                )
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    minimum_iou = 1.0
    if validate_pixels and not errors:
        for index in range(1, VIEW_COUNT + 1):
            image_path = scene / "image" / f"img{index:03d}.jpg"
            mask_path = scene / "mask" / f"img{index:03d}.png"
            inverse_depth_path = (
                scene / "invdepth" / f"img{index:03d}.npy"
            )
            with Image.open(image_path) as image:
                if image.size != RESOLUTION:
                    errors.append(
                        f"img{index:03d}: image resolution is {image.size}"
                    )
                    continue
            mask = mask_array(mask_path)
            if mask.shape != (RESOLUTION[1], RESOLUTION[0]):
                errors.append(
                    f"img{index:03d}: mask shape is {mask.shape}"
                )
                continue
            if not mask.any():
                errors.append(f"img{index:03d}: mask is empty")
            if (
                mask[0].any()
                or mask[-1].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ):
                errors.append(
                    f"img{index:03d}: mask touches an image border"
                )
            inverse_depth = np.load(inverse_depth_path)
            if (
                inverse_depth.shape != mask.shape
                or inverse_depth.dtype != np.float32
            ):
                errors.append(
                    f"img{index:03d}: invalid inverse-depth shape or dtype"
                )
                continue
            iou = inverse_depth_mask_iou(mask, inverse_depth)
            minimum_iou = min(minimum_iou, iou)
            if iou < MIN_INVDEPTH_MASK_IOU:
                errors.append(
                    f"img{index:03d}: inverse-depth/mask IoU is {iou:.6f}"
                )

    if errors:
        raise RuntimeError(
            f"Validation failed for {scene}:\n"
            + "\n".join(errors[:100])
        )
    return {
        "scene": scene.name,
        "view_count": VIEW_COUNT,
        "train_count": len(TRAIN_INDICES),
        "test_count": len(TEST_INDICES),
        "minimum_invdepth_mask_iou": minimum_iou,
    }


def sugar_focal_length() -> float:
    return (
        0.5
        * RESOLUTION[0]
        / math.tan(0.5 * math.radians(FOV_X_DEG))
    )


def effective_to_colmap_w2c(
    effective_c2w: np.ndarray,
) -> np.ndarray:
    opencv_c2w = effective_c2w @ OPENGL_TO_OPENCV_CAMERA
    return np.linalg.inv(opencv_c2w)


def colmap_w2c_to_effective(
    world_to_camera: np.ndarray,
) -> np.ndarray:
    opencv_c2w = np.linalg.inv(world_to_camera)
    return opencv_c2w @ OPENGL_TO_OPENCV_CAMERA


def install_transactionally(
    temporary: Path, target: Path, overwrite: bool
) -> None:
    backup = target.with_name(
        f".{target.name}.backup-{os.getpid()}"
    )
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def blender_command(
    args: argparse.Namespace,
    action: str,
    record: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        str(args.blender.resolve()),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(SCRIPT_DIR / "blender_worker.py"),
        "--",
        action,
        "--manifest",
        str(args.manifest.resolve()),
        "--source-root",
        str(args.source_root.resolve()),
        "--shoe",
        str(record["name"]),
        "--output",
        str(output),
    ]


def parse_gpus(args: argparse.Namespace) -> list[int]:
    if getattr(args, "gpu", None) is not None:
        return [int(args.gpu)]
    raw = getattr(args, "gpus", None)
    if raw:
        values = [
            int(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
        if (
            not values
            or len(values) != len(set(values))
            or any(value < 0 for value in values)
        ):
            raise ValueError(f"Invalid GPU list: {raw!r}")
        return values
    return [0]


def run_blender(command: list[str], gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, check=True, env=environment)


def build_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    gpu_pool: queue.Queue[int],
) -> dict[str, Any]:
    name = str(record["name"])
    target = args.output_root.resolve() / name
    if target.exists() and not args.overwrite:
        validation = validate_scene(target, expected_record=record)
        print(f"[skip] {name}: existing scene is valid", flush=True)
        return validation
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve())
    )
    gpu = gpu_pool.get()
    try:
        print(f"[build] {name} on physical GPU {gpu}", flush=True)
        run_blender(blender_command(args, "build", record, temporary), gpu)
        validation = validate_scene(temporary, expected_record=record)
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
        print(
            f"[ok] {name}: {VIEW_COUNT} exact Blender views",
            flush=True,
        )
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        gpu_pool.put(gpu)


def run_build(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(gpus)
    ) as executor:
        futures = {
            executor.submit(
                build_record, args, record, gpu_pool
            ): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[failed] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Build failures:\n" + "\n".join(failures))


def run_audit(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(
            args.manifest.resolve(),
            args.source_root.resolve(),
            require_reviewed=False,
        ),
        args.shoe,
        args.all,
    )
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def audit_record(record: dict[str, Any]) -> None:
        gpu = gpu_pool.get()
        try:
            target = (
                args.output_dir.resolve() / str(record["name"])
            )
            if target.exists():
                shutil.rmtree(target)
            run_blender(
                blender_command(args, "audit", record, target), gpu
            )
            print(
                f"[audit] {record['name']} -> {target}", flush=True
            )
        finally:
            gpu_pool.put(gpu)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(gpus)
    ) as executor:
        futures = {
            executor.submit(audit_record, record): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(
                    f"[audit-failed] {failures[-1]}", flush=True
                )
    if failures:
        raise RuntimeError("Audit failures:\n" + "\n".join(failures))


def run_validate(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        result = validate_scene(
            args.output_root.resolve() / str(record["name"]),
            expected_record=record,
        )
        print(json.dumps(result, sort_keys=True))
