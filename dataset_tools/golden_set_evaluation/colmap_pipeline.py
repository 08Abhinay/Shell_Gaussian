#!/usr/bin/env python3
"""Build compact COLMAP datasets directly from raw evaluation RGB images.

COLMAP reads ``<raw>/<scene>/images/*.jpg`` in place. Source masks are recorded
for provenance and aligned after reconstruction, but neither raw RGB images nor
raw masks are copied into the processed dataset. Blender transforms,
inverse-depth maps, canonical boxes, and meshes are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


TOOL_VERSION = "1.2.0"
DEFAULT_COLMAP = Path("/storage/Abhinay/conda_envs/colmap/bin/colmap")
EXPECTED_IMAGE_COUNT = 180
MODEL_FILE_SUFFIXES = {".bin", ".txt"}
SYNTHETIC_RETRY_CAMERA_PARAMS = "3842.99,768,512,0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an RGB-only COLMAP reconstruction for FAB evaluation scenes and "
            "write a Golden-compatible images/colmap/undistorted layout."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", action="append", help="Scene name; repeatable.")
    group.add_argument("--all", action="store_true", help="Process every valid scene.")
    parser.add_argument("--colmap-bin", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--gpu-index", type=int, default=0, help="Physical CUDA GPU index.")
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument(
        "--reconstruction-profile",
        choices=("default", "synthetic-low-texture"),
        default="default",
        help=(
            "The low-texture profile fixes shared SIMPLE_RADIAL intrinsics and "
            "uses more permissive SIFT/mapping settings. Both profiles estimate "
            "camera poses independently from the current scene's RGB images."
        ),
    )
    parser.add_argument("--min-registered-images", type=int, default=1)
    parser.add_argument(
        "--batch-manifest-path",
        type=Path,
        default=None,
        help="Optional path for the invocation-level manifest; avoids shared-file races in parallel batches.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep database.db and intermediate mapper models in the final scene.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_scene(path: Path) -> bool:
    image_dir = path / "images"
    mask_dir = path / "masks"
    return (
        image_dir.is_dir()
        and mask_dir.is_dir()
        and len(list(image_dir.glob("*.jpg"))) == EXPECTED_IMAGE_COUNT
        and len(list(mask_dir.glob("*.png"))) == EXPECTED_IMAGE_COUNT
    )


def discover_scenes(source_root: Path, requested: list[str] | None) -> list[str]:
    if requested is not None:
        names = requested
    else:
        names = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    missing = [name for name in names if not valid_scene(source_root / name)]
    if missing:
        raise ValueError(f"Invalid or incomplete FAB scenes: {missing}")
    return names


def command_string(command: Iterable[object]) -> str:
    return shlex.join(str(item) for item in command)


def tail_text(path: Path, line_count: int = 60) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-line_count:])


def run_stage(
    stage: str,
    command: list[object],
    log_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    log_path = log_dir / f"{stage}.log"
    printable = command_string(command)
    print(f"[{stage}] {printable}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n\n")
        log.flush()
        completed = subprocess.run(
            [str(item) for item in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLMAP stage {stage!r} failed with exit code {completed.returncode}.\n"
            f"Log: {log_path}\n{tail_text(log_path)}"
        )
    print(f"[{stage}] completed in {elapsed:.1f}s", flush=True)
    return {
        "stage": stage,
        "command": printable,
        "log": str(log_path.name),
        "elapsed_seconds": elapsed,
        "return_code": completed.returncode,
    }


ANALYZER_PATTERNS = {
    "cameras": re.compile(r"Cameras:\s+(\d+)"),
    "registered_images": re.compile(r"Registered images:\s+(\d+)"),
    "points": re.compile(r"Points:\s+(\d+)"),
    "observations": re.compile(r"Observations:\s+(\d+)"),
    "mean_track_length": re.compile(r"Mean track length:\s+([0-9.eE+-]+)"),
    "mean_observations_per_image": re.compile(
        r"Mean observations per image:\s+([0-9.eE+-]+)"
    ),
    "mean_reprojection_error_px": re.compile(
        r"Mean reprojection error:\s+([0-9.eE+-]+)px"
    ),
}


def analyze_model(colmap_bin: Path, model_path: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(colmap_bin), "model_analyzer", "--path", str(model_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        check=True,
    )
    result: dict[str, Any] = {}
    integer_keys = {"cameras", "registered_images", "points", "observations"}
    for key, pattern in ANALYZER_PATTERNS.items():
        match = pattern.search(completed.stdout)
        if match:
            result[key] = int(match.group(1)) if key in integer_keys else float(match.group(1))
    result["analyzer_output"] = completed.stdout
    if "registered_images" not in result or "points" not in result:
        raise RuntimeError(f"Could not parse COLMAP model analyzer output:\n{completed.stdout}")
    return result


def database_stats(database_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    with sqlite3.connect(database_path) as connection:
        for table in ("cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"):
            try:
                stats[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.OperationalError:
                stats[table] = 0
        try:
            stats["total_keypoints"] = int(
                connection.execute("SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0]
            )
        except sqlite3.OperationalError:
            stats["total_keypoints"] = 0
    return stats


def copy_model(model_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in model_path.iterdir():
        if path.is_file() and path.suffix in MODEL_FILE_SUFFIXES:
            shutil.copy2(path, destination / path.name)


def normalize_undistorted_sparse(undistorted: Path) -> Path:
    sparse = undistorted / "sparse"
    nested = sparse / "0"
    if nested.is_dir() and (nested / "cameras.bin").exists():
        return nested
    if not (sparse / "cameras.bin").exists():
        raise RuntimeError(f"COLMAP undistorter did not write a model under {sparse}")
    nested.mkdir(parents=True, exist_ok=True)
    for path in list(sparse.iterdir()):
        if path.is_file():
            shutil.move(str(path), nested / path.name)
    return nested


def atomic_install(temporary: Path, destination: Path, overwrite: bool) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene already exists: {destination}")
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def convert_scene(args: argparse.Namespace, scene_name: str) -> dict[str, Any]:
    source_scene = args.source_root / scene_name
    source_images = source_scene / "images"
    source_masks = source_scene / "masks"
    destination = args.output_root / scene_name
    temporary = args.output_root / f".{scene_name}.tmp-{uuid.uuid4().hex}"
    workspace = temporary / "workspace"
    mapper_root = workspace / "sparse"
    database_path = workspace / "database.db"
    log_dir = temporary / "logs"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Output scene already exists: {destination}")
    temporary.mkdir(parents=True)
    mapper_root.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    print(f"Working directory: {temporary}", flush=True)

    try:
        image_paths = sorted(source_images.glob("*.jpg"))
        if len(image_paths) != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"Expected {EXPECTED_IMAGE_COUNT} JPEGs, found {len(image_paths)}")
        mask_paths = sorted(source_masks.glob("*.png"))
        if len(mask_paths) != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"Expected {EXPECTED_IMAGE_COUNT} masks, found {len(mask_paths)}")

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        colmap = args.colmap_bin.resolve()
        stages: list[dict[str, Any]] = []

        retry_profile = args.reconstruction_profile == "synthetic-low-texture"
        feature_options: list[object] = []
        matcher_options: list[object] = []
        mapper_options: list[object] = []
        if retry_profile:
            feature_options = [
                "--ImageReader.camera_params",
                SYNTHETIC_RETRY_CAMERA_PARAMS,
                "--SiftExtraction.peak_threshold",
                "0.002",
                "--SiftExtraction.edge_threshold",
                "20",
            ]
            matcher_options = [
                "--FeatureMatching.guided_matching",
                "1",
                "--SiftMatching.max_ratio",
                "0.9",
                "--TwoViewGeometry.min_num_inliers",
                "8",
            ]
            mapper_options = [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
                "--Mapper.init_min_num_inliers",
                "15",
                "--Mapper.init_min_tri_angle",
                "4",
                "--Mapper.abs_pose_min_num_inliers",
                "15",
                "--Mapper.abs_pose_min_inlier_ratio",
                "0.1",
            ]

        stages.append(
            run_stage(
                "01_feature_extractor",
                [
                    colmap,
                    "feature_extractor",
                    "--database_path",
                    database_path,
                    "--image_path",
                    source_images,
                    "--ImageReader.camera_model",
                    "SIMPLE_RADIAL",
                    "--ImageReader.single_camera",
                    "1",
                    "--FeatureExtraction.use_gpu",
                    "1",
                    "--FeatureExtraction.gpu_index",
                    "0",
                    "--SiftExtraction.max_num_features",
                    max(args.max_num_features, 16384) if retry_profile else args.max_num_features,
                    *feature_options,
                    "--default_random_seed",
                    "0",
                ],
                log_dir,
                environment,
            )
        )
        stages.append(
            run_stage(
                "02_exhaustive_matcher",
                [
                    colmap,
                    "exhaustive_matcher",
                    "--database_path",
                    database_path,
                    "--FeatureMatching.use_gpu",
                    "1",
                    "--FeatureMatching.gpu_index",
                    "0",
                    *matcher_options,
                    "--default_random_seed",
                    "0",
                ],
                log_dir,
                environment,
            )
        )
        db_stats = database_stats(database_path)
        stages.append(
            run_stage(
                "03_mapper",
                [
                    colmap,
                    "mapper",
                    "--database_path",
                    database_path,
                    "--image_path",
                    source_images,
                    "--output_path",
                    mapper_root,
                    "--Mapper.multiple_models",
                    "1",
                    "--Mapper.max_num_models",
                    "5",
                    "--Mapper.min_model_size",
                    "10",
                    *mapper_options,
                    "--Mapper.random_seed",
                    "0",
                    "--default_random_seed",
                    "0",
                ],
                log_dir,
                environment,
            )
        )
        candidates = sorted(path for path in mapper_root.iterdir() if path.is_dir())
        if not candidates:
            raise RuntimeError("COLMAP mapper did not produce a sparse model")
        analyses = [(path, analyze_model(colmap, path, environment)) for path in candidates]
        best_model, best_analysis = max(
            analyses,
            key=lambda item: (item[1]["registered_images"], item[1]["points"]),
        )
        if best_analysis["registered_images"] < args.min_registered_images:
            raise RuntimeError(
                f"Best COLMAP model registered only {best_analysis['registered_images']} images; "
                f"minimum required is {args.min_registered_images}"
            )

        raw_model = temporary / "colmap"
        copy_model(best_model, raw_model)
        stages.append(
            run_stage(
                "04_raw_model_txt",
                [colmap, "model_converter", "--input_path", best_model, "--output_path", raw_model, "--output_type", "TXT"],
                log_dir,
                environment,
            )
        )
        undistorted = temporary / "undistorted"
        stages.append(
            run_stage(
                "05_image_undistorter",
                [
                    colmap,
                    "image_undistorter",
                    "--image_path",
                    source_images,
                    "--input_path",
                    best_model,
                    "--output_path",
                    undistorted,
                    "--output_type",
                    "COLMAP",
                    "--max_image_size",
                    "-1",
                ],
                log_dir,
                environment,
            )
        )
        undistorted_model = normalize_undistorted_sparse(undistorted)
        stages.append(
            run_stage(
                "06_undistorted_model_txt",
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    undistorted_model,
                    "--output_path",
                    undistorted_model,
                    "--output_type",
                    "TXT",
                ],
                log_dir,
                environment,
            )
        )
        undistorted_analysis = analyze_model(colmap, undistorted_model, environment)
        manifest = {
            "tool": "golden_set_evaluation/colmap_pipeline.py",
            "tool_version": TOOL_VERSION,
            "scene": scene_name,
            "protocol": "rgb_only_fresh_colmap_all_180_views",
            "source_scene": str(source_scene.resolve()),
            "colmap": {
                "binary": str(colmap),
                "version": subprocess.run(
                    [str(colmap), "-h"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                    check=True,
                ).stdout.splitlines()[0],
                "camera_model": "SIMPLE_RADIAL",
                "camera_params": SYNTHETIC_RETRY_CAMERA_PARAMS if retry_profile else "estimated",
                "reconstruction_profile": args.reconstruction_profile,
                "intrinsics_source": (
                    "fixed_synthetic_camera_calibration"
                    if retry_profile
                    else "estimated_per_scene_by_colmap"
                ),
                "camera_pose_source": "estimated_independently_from_this_scene_rgb",
                "single_shared_camera": True,
                "gpu_index_physical": args.gpu_index,
                "max_num_features": (
                    max(args.max_num_features, 16384)
                    if retry_profile
                    else args.max_num_features
                ),
                "matching": "exhaustive",
                "default_random_seed": 0,
            },
            "inputs": {
                "image_count": len(image_paths),
                "images": [
                    {"name": path.name, "sha256": sha256(path)} for path in image_paths
                ],
                "masks": [
                    {"name": path.name, "sha256": sha256(path)} for path in mask_paths
                ],
            },
            "storage_contract": {
                "raw_rgb_and_masks_are_external": True,
                "duplicates_raw_inputs": False,
                "final_layout": [
                    "undistorted/images",
                    "undistorted/masks",
                    "undistorted/sparse/0",
                    "logs",
                    "conversion_manifest.json",
                ],
                "raw_colmap_model_is_transient_for_mask_alignment": True,
                "raw_model_retained": True,
                "compaction_complete": False,
            },
            "prohibited_training_inputs": {
                "uses_blender_camera_transforms": False,
                "uses_inverse_depth": False,
                "uses_masks_in_colmap": False,
                "uses_ground_truth_mesh": False,
                "uses_canonical_bbox": False,
            },
            "database": db_stats,
            "mapper_models": [
                {"directory": path.name, **{k: v for k, v in analysis.items() if k != "analyzer_output"}}
                for path, analysis in analyses
            ],
            "selected_model": best_model.name,
            "raw_model": {k: v for k, v in best_analysis.items() if k != "analyzer_output"},
            "undistorted_model": {
                k: v for k, v in undistorted_analysis.items() if k != "analyzer_output"
            },
            "stages": stages,
            "workspace_retained": args.keep_workspace,
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if not args.keep_workspace:
            shutil.rmtree(workspace)
        atomic_install(temporary, destination, args.overwrite)
        print(
            f"Completed {scene_name}: registered={best_analysis['registered_images']}/"
            f"{EXPECTED_IMAGE_COUNT}, points={best_analysis['points']}",
            flush=True,
        )
        return manifest
    except BaseException:
        failed = args.output_root / f".{scene_name}.failed-{uuid.uuid4().hex}"
        if temporary.exists():
            os.replace(temporary, failed)
            print(f"Failed workspace retained at: {failed}", file=sys.stderr, flush=True)
        raise


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.colmap_bin = args.colmap_bin.resolve()
    if not args.colmap_bin.is_file() or not os.access(args.colmap_bin, os.X_OK):
        raise FileNotFoundError(f"COLMAP executable not found: {args.colmap_bin}")
    if args.gpu_index < 0:
        raise ValueError("--gpu-index must be non-negative")
    if args.max_num_features <= 0:
        raise ValueError("--max-num-features must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    scenes = discover_scenes(args.source_root, args.scene)
    manifests = []
    for index, scene_name in enumerate(scenes, start=1):
        print(f"=== [{index}/{len(scenes)}] {scene_name} ===", flush=True)
        manifests.append(convert_scene(args, scene_name))
    batch = {
        "tool_version": TOOL_VERSION,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "scene_count": len(manifests),
        "scenes": [
            {
                "scene": manifest["scene"],
                "registered_images": manifest["raw_model"]["registered_images"],
                "points": manifest["raw_model"]["points"],
            }
            for manifest in manifests
        ],
    }
    batch_manifest_path = (
        args.batch_manifest_path.resolve()
        if args.batch_manifest_path is not None
        else args.output_root / "conversion_batch_manifest.json"
    )
    batch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    batch_manifest_path.write_text(
        json.dumps(batch, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
