#!/usr/bin/env python3
"""Build and convert reviewed Blender shoe datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools_blender.core import *  # noqa: F403
from dataset_tools_blender.gshell import pipeline as gshell_pipeline
from dataset_tools_blender.neuraludf import pipeline as neuraludf_pipeline
from dataset_tools_blender.neuraludf.pipeline import (
    effective_neuraludf_frames,
    neuraludf_camera_matrices,
    neuraludf_intrinsic,
    neuraludf_scale_matrix,
    normalized_neuraludf_pose,
    prepare_neuraludf_record,
    recover_neuraludf_pose,
    run_prepare_neuraludf,
    run_validate_neuraludf,
    validate_neuraludf_scene,
    write_neuraludf_scene,
)
from dataset_tools_blender.neus2 import pipeline as neus2_pipeline
from dataset_tools_blender.neus2.pipeline import (
    effective_neus2_frames,
    neus2_intrinsic,
    neus2_normalization,
    neus2_scale_offset,
    neus2_transform_payload,
    prepare_neus2_record,
    run_prepare_neus2,
    run_validate_neus2,
    validate_neus2_scene,
    write_neus2_scene,
)
from dataset_tools_blender.sugar import pipeline as sugar_pipeline
from dataset_tools_blender.sugar.pipeline import (
    effective_sugar_frames,
    parse_colmap_camera,
    parse_colmap_images,
    parse_colmap_points,
    prepare_sugar_record,
    qvec_to_rotmat,
    rewrite_colmap_image_extensions,
    robust_sparse_bbox,
    rotmat_to_qvec,
    run_colmap_stage,
    run_prepare_sugar,
    run_validate_sugar,
    validate_sugar_scene,
    write_seed_colmap_model,
    write_sugar_images,
)


def add_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shoe")
    selection.add_argument("--all", action="store_true")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT  # noqa: F405
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST  # noqa: F405
    )


def add_gpus(parser: argparse.ArgumentParser) -> None:
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int)
    devices.add_argument("--gpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit", help="Render temporary semantic audit views."
    )
    add_common(audit)
    add_selection(audit)
    add_gpus(audit)
    audit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/golden_set_evaluation_blender_audit"),
    )
    audit.add_argument(
        "--blender", type=Path, default=DEFAULT_BLENDER  # noqa: F405
    )

    build = commands.add_parser(
        "build", help="Build transactional GShell-ready scenes."
    )
    add_common(build)
    add_selection(build)
    add_gpus(build)
    build.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    build.add_argument(
        "--blender", type=Path, default=DEFAULT_BLENDER  # noqa: F405
    )
    build.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser(
        "validate", help="Validate completed scenes."
    )
    add_common(validate)
    add_selection(validate)
    validate.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )

    prepare_sugar = commands.add_parser(
        "prepare-sugar",
        help=(
            "Create SuGaR data with exact cameras and triangulated "
            "sparse points."
        ),
    )
    add_common(prepare_sugar)
    add_selection(prepare_sugar)
    add_gpus(prepare_sugar)
    prepare_sugar.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_sugar.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SUGAR_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_sugar.add_argument(
        "--colmap-bin", type=Path, default=DEFAULT_COLMAP  # noqa: F405
    )
    prepare_sugar.add_argument("--overwrite", action="store_true")

    validate_sugar = commands.add_parser(
        "validate-sugar", help="Validate SuGaR-ready scenes."
    )
    add_common(validate_sugar)
    add_selection(validate_sugar)
    validate_sugar.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_sugar.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SUGAR_OUTPUT_ROOT,  # noqa: F405
    )

    prepare_neuraludf = commands.add_parser(
        "prepare-neuraludf",
        help="Create NeuralUDF data from exact Blender cameras.",
    )
    add_common(prepare_neuraludf)
    add_selection(prepare_neuraludf)
    prepare_neuraludf.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_neuraludf.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEURALUDF_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_neuraludf.add_argument("--overwrite", action="store_true")

    validate_neuraludf = commands.add_parser(
        "validate-neuraludf",
        help="Validate NeuralUDF-ready scenes.",
    )
    add_common(validate_neuraludf)
    add_selection(validate_neuraludf)
    validate_neuraludf.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_neuraludf.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEURALUDF_OUTPUT_ROOT,  # noqa: F405
    )

    prepare_neus2 = commands.add_parser(
        "prepare-neus2",
        help="Create NeuS2 data from exact Blender cameras.",
    )
    add_common(prepare_neus2)
    add_selection(prepare_neus2)
    prepare_neus2.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_neus2.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEUS2_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_neus2.add_argument("--overwrite", action="store_true")

    validate_neus2 = commands.add_parser(
        "validate-neus2", help="Validate NeuS2-ready scenes."
    )
    add_common(validate_neus2)
    add_selection(validate_neus2)
    validate_neus2.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_neus2.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEUS2_OUTPUT_ROOT,  # noqa: F405
    )

    prepare_neus2_turntable = commands.add_parser(
        "prepare-neus2-turntable",
        help="Create a 36-view NeuS2 turntable scene with a 30/6 split.",
    )
    add_common(prepare_neus2_turntable)
    add_selection(prepare_neus2_turntable)
    prepare_neus2_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_neus2_turntable.add_argument(
        "--output-root",
        type=Path,
        default=neus2_pipeline.DEFAULT_NEUS2_TURNTABLE_OUTPUT_ROOT,
    )
    prepare_neus2_turntable.add_argument("--overwrite", action="store_true")

    validate_neus2_turntable = commands.add_parser(
        "validate-neus2-turntable",
        help="Validate 36-view NeuS2 turntable scenes.",
    )
    add_common(validate_neus2_turntable)
    add_selection(validate_neus2_turntable)
    validate_neus2_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_neus2_turntable.add_argument(
        "--output-root",
        type=Path,
        default=neus2_pipeline.DEFAULT_NEUS2_TURNTABLE_OUTPUT_ROOT,
    )

    prepare_gshell_turntable = commands.add_parser(
        "prepare-gshell-turntable",
        help="Create a 36-view G-Shell turntable scene with a 30/6 split.",
    )
    add_common(prepare_gshell_turntable)
    add_selection(prepare_gshell_turntable)
    prepare_gshell_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_gshell_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GSHELL_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_gshell_turntable.add_argument("--overwrite", action="store_true")

    validate_gshell_turntable = commands.add_parser(
        "validate-gshell-turntable",
        help="Validate 36-view G-Shell turntable scenes.",
    )
    add_common(validate_gshell_turntable)
    add_selection(validate_gshell_turntable)
    validate_gshell_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_gshell_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GSHELL_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )

    prepare_neuraludf_turntable = commands.add_parser(
        "prepare-neuraludf-turntable",
        help="Create a 30-view NeuralUDF turntable training scene.",
    )
    add_common(prepare_neuraludf_turntable)
    add_selection(prepare_neuraludf_turntable)
    prepare_neuraludf_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_neuraludf_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEURALUDF_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_neuraludf_turntable.add_argument("--overwrite", action="store_true")

    validate_neuraludf_turntable = commands.add_parser(
        "validate-neuraludf-turntable",
        help="Validate NeuralUDF turntable training scenes.",
    )
    add_common(validate_neuraludf_turntable)
    add_selection(validate_neuraludf_turntable)
    validate_neuraludf_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_neuraludf_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEURALUDF_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )

    prepare_sugar_turntable = commands.add_parser(
        "prepare-sugar-turntable",
        help="Create a 36-view SuGaR turntable scene with train-only points.",
    )
    add_common(prepare_sugar_turntable)
    add_selection(prepare_sugar_turntable)
    add_gpus(prepare_sugar_turntable)
    prepare_sugar_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    prepare_sugar_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SUGAR_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )
    prepare_sugar_turntable.add_argument(
        "--colmap-bin", type=Path, default=DEFAULT_COLMAP  # noqa: F405
    )
    prepare_sugar_turntable.add_argument("--overwrite", action="store_true")

    validate_sugar_turntable = commands.add_parser(
        "validate-sugar-turntable",
        help="Validate 36-view SuGaR turntable scenes.",
    )
    add_common(validate_sugar_turntable)
    add_selection(validate_sugar_turntable)
    validate_sugar_turntable.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT  # noqa: F405
    )
    validate_sugar_turntable.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SUGAR_TURNTABLE_OUTPUT_ROOT,  # noqa: F405
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "audit": run_audit,  # noqa: F405
        "build": run_build,  # noqa: F405
        "validate": run_validate,  # noqa: F405
        "prepare-sugar": sugar_pipeline.run_prepare_sugar,
        "validate-sugar": sugar_pipeline.run_validate_sugar,
        "prepare-neuraludf": neuraludf_pipeline.run_prepare_neuraludf,
        "validate-neuraludf": neuraludf_pipeline.run_validate_neuraludf,
        "prepare-neus2": neus2_pipeline.run_prepare_neus2,
        "validate-neus2": neus2_pipeline.run_validate_neus2,
        "prepare-neus2-turntable": (
            neus2_pipeline.run_prepare_neus2_turntable
        ),
        "validate-neus2-turntable": (
            neus2_pipeline.run_validate_neus2_turntable
        ),
        "prepare-gshell-turntable": (
            gshell_pipeline.run_prepare_gshell_turntable
        ),
        "validate-gshell-turntable": (
            gshell_pipeline.run_validate_gshell_turntable
        ),
        "prepare-neuraludf-turntable": (
            neuraludf_pipeline.run_prepare_neuraludf_turntable
        ),
        "validate-neuraludf-turntable": (
            neuraludf_pipeline.run_validate_neuraludf_turntable
        ),
        "prepare-sugar-turntable": (
            sugar_pipeline.run_prepare_sugar_turntable
        ),
        "validate-sugar-turntable": (
            sugar_pipeline.run_validate_sugar_turntable
        ),
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
