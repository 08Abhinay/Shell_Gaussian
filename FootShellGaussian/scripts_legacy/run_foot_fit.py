"""Fit a SUPR foot into one prepared normal shoe in a single joint stage.

This runner replaces the ``run_alignment`` then ``run_cavity_analysis`` then
``run_containment_fit`` sequence with one step, and reads only the shoe
preparation. It writes into its own output directory, so the retired pipeline's
artifacts -- and everything Checkpoints 11-B3, 11-C and 11-D were built on --
stay exactly where they are.

The record it writes is a superset of ``containment_fit.json``: the fields the
downstream stages validate keep their names, values and meanings, and the new
seating verdict travels alongside them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.foot_fit import FootFitError, build_foot_fit
from foot_prior.foot_fit.report import CONTAINMENT_SCHEMA_VERSION
from foot_prior.foot_fit.seating import COMPRESSION_ALLOWANCE_MM
from foot_prior.mesh import TriangleMesh, save_triangle_mesh, transform_mesh
from foot_prior.supr_foot import load_neutral_supr_foot, load_posable_supr_foot

from run_alignment import (
    FOOT_COLOR,
    SHOE_COLOR,
    _load_prepared_inputs,
)


ARTIFACT_NAMES = (
    "foot_fit.json",
    "foot_fitted.ply",
    "foot_clearance_colored.ply",
    "foot_fit_overlay.ply",
)
NORMAL_SHOE_PROFILE = "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit SUPR shape and pose together against one seating objective, "
            "solving vertical placement instead of dropping to first contact."
        )
    )
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--compression-allowance-mm",
        type=float,
        default=COMPRESSION_ALLOWANCE_MM,
        help=(
            "Depth the foot may press into the footbed before the seating "
            "penalty becomes a barrier."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known foot-fit artifacts.",
    )
    return parser.parse_args()


def _footbed_face_indices(preparation: dict[str, Any]) -> np.ndarray:
    selection = preparation.get("footbed_selection")
    if not isinstance(selection, dict):
        raise ValueError("shoe_preparation.json is missing footbed_selection")
    indices = np.asarray(
        selection.get("original_face_indices"), dtype=np.int64
    )
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("footbed_selection is missing original_face_indices")
    return indices


def _make_overlay(
    shoe: TriangleMesh, foot: TriangleMesh, foot_colors: np.ndarray
) -> TriangleMesh:
    shoe_colors = np.tile(SHOE_COLOR, (len(shoe.vertices), 1))
    return TriangleMesh(
        np.concatenate((shoe.vertices, foot.vertices), axis=0),
        np.concatenate((shoe.faces, foot.faces + len(shoe.vertices)), axis=0),
        np.concatenate((shoe_colors, foot_colors), axis=0),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    preparation_dir = args.preparation_dir.expanduser().resolve(strict=True)
    supr_path = args.supr_model.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"foot-fit artifacts already exist: {formatted}; "
            "pass --overwrite to replace them"
        )

    (
        normalized_shoe,
        original_footbed,
        preparation,
        shoe_to_normalized,
        normalized_to_shoe,
        centerline_xz,
        support_grid_cell_spacing,
    ) = _load_prepared_inputs(preparation_dir)
    normalized_footbed = transform_mesh(original_footbed, shoe_to_normalized)
    neutral_foot = load_neutral_supr_foot(supr_path)
    posable_foot = load_posable_supr_foot(supr_path, num_betas=10)

    fit = build_foot_fit(
        supr_model=posable_foot,
        neutral_foot_mesh=neutral_foot,
        normalized_shoe_mesh=normalized_shoe,
        normalized_support_mesh=normalized_footbed,
        normalized_centerline_xz=centerline_xz,
        footbed_source_face_indices=_footbed_face_indices(preparation),
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        support_grid_cell_spacing=support_grid_cell_spacing,
        compression_allowance_mm=args.compression_allowance_mm,
    )

    fitted_foot = fit.fitted_foot
    clearance_colors = fit.final_cavity.foot_vertex_colors(
        fitted_foot, support_grid_cell_spacing
    )
    plain_colors = np.tile(FOOT_COLOR, (len(fitted_foot.vertices), 1))
    payload: dict[str, Any] = {
        "schema_version": CONTAINMENT_SCHEMA_VERSION,
        "shoe_profile": NORMAL_SHOE_PROFILE,
        "inputs": {
            "preparation_directory": str(preparation_dir),
            "shoe_preparation": str(preparation_dir / "shoe_preparation.json"),
            "normalized_shoe": str(preparation_dir / "shoe_normalized.ply"),
            "original_footbed": str(preparation_dir / "footbed_surface.ply"),
            "supr_model": str(supr_path),
        },
        "source_preparation_schema_version": preparation.get("schema_version"),
        "support_grid_cell_spacing": support_grid_cell_spacing,
        **fit.to_dict(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        save_triangle_mesh(staging / "foot_fitted.ply", fitted_foot, plain_colors)
        save_triangle_mesh(
            staging / "foot_clearance_colored.ply", fitted_foot, clearance_colors
        )
        save_triangle_mesh(
            staging / "foot_fit_overlay.ply",
            _make_overlay(normalized_shoe, fitted_foot, clearance_colors),
        )
        (staging / "foot_fit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ARTIFACT_NAMES:
            os.replace(staging / name, targets[name])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return payload


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
    except (
        FileExistsError,
        FileNotFoundError,
        FootFitError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"foot fit failed: {error}") from error
    output_dir = args.output_dir.expanduser().resolve()
    seating = payload["seating"]["acceptance"]
    print(f"wrote foot-fit artifacts to {output_dir}")
    print(
        f"seating={payload['seating_status']} "
        f"legacy_status={payload['status']} "
        f"worst_region={seating['worst_region_name']} "
        f"rms_gap={seating['worst_region_rms_gap']:.5f} "
        f"({seating['worst_region_rms_gap_mm']:.2f} mm)"
    )
    print(
        f"median_gap={seating['overall_median_gap']:.5f} "
        f"min_region_coverage={seating['minimum_region_coverage']:.4f} "
        f"compression={seating['compression_depth_mm']:.2f} mm"
    )


if __name__ == "__main__":
    main()
