"""Prepare one canonical right shoe for later SUPR fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foot_prior.footbed import (
    HighHeelSupport,
    identify_footbed_surface,
    identify_high_heel_support,
)
from foot_prior.mesh import (
    combine_colored_meshes,
    load_triangle_mesh,
    save_triangle_mesh,
)
from foot_prior.normalization import (
    EXPECTED_SHOE_COORDINATE_SYSTEM,
    HIGH_HEEL_SHOE_PROFILE,
    SHOE_AXIS_SEMANTICS,
    SHOE_SIDE,
    build_shoe_normalization,
    validate_shoe_frame_metadata,
)


ARTIFACT_NAMES = (
    "shoe_preparation.json",
    "footbed_surface.ply",
    "footbed_overlay.ply",
    "shoe_normalized.ply",
)
SHOE_COLOR = np.asarray([150, 150, 150, 255], dtype=np.uint8)
FOOTBED_COLOR = np.asarray([40, 180, 90, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect support in one canonical right shoe and functionally "
            "normalize the shoe."
        )
    )
    parser.add_argument("--shoe-mesh", required=True, type=Path)
    parser.add_argument("--canonicalization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known artifacts for the shoe's profile.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    shoe_path = args.shoe_mesh.expanduser().resolve(strict=True)
    canonicalization_path = args.canonicalization.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    shoe_profile = validate_shoe_frame_metadata(canonicalization_path)
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "shoe-preparation artifacts already exist: "
            f"{formatted}; pass --overwrite to replace them"
        )

    shoe = load_triangle_mesh(shoe_path)
    high_heel_support: HighHeelSupport | None = None
    if shoe_profile == HIGH_HEEL_SHOE_PROFILE:
        high_heel_support = identify_high_heel_support(shoe)
        footbed = high_heel_support.surface
    else:
        footbed = identify_footbed_surface(shoe)
    normalization = build_shoe_normalization(shoe, footbed)
    normalized_shoe = normalization.shoe_mesh_to_normalized(shoe)
    overlay = combine_colored_meshes(
        (shoe, SHOE_COLOR), (footbed.mesh, FOOTBED_COLOR)
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "shoe_profile": shoe_profile,
        "inputs": {
            "shoe_mesh": str(shoe_path),
            "canonicalization": str(canonicalization_path),
        },
        "coordinate_contract": {
            "coordinate_system": EXPECTED_SHOE_COORDINATE_SYSTEM,
            "side": SHOE_SIDE,
            "axes": SHOE_AXIS_SEMANTICS,
        },
        "footbed_selection": footbed.to_dict(),
        "normalization": normalization.to_dict(),
    }
    if high_heel_support is not None:
        payload["preparation_status"] = "support_detected_and_normalized"
        payload["high_heel_support"] = high_heel_support.to_dict()

    payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    footbed_colors = np.tile(
        FOOTBED_COLOR, (len(footbed.mesh.vertices), 1)
    )
    save_triangle_mesh(
        targets["footbed_surface.ply"], footbed.mesh, footbed_colors
    )
    save_triangle_mesh(targets["footbed_overlay.ply"], overlay)
    save_triangle_mesh(targets["shoe_normalized.ply"], normalized_shoe)
    targets["shoe_preparation.json"].write_text(
        payload_json, encoding="utf-8"
    )
    return payload


def main() -> None:
    parser_args = parse_args()
    try:
        payload = run(parser_args)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"shoe preparation failed: {error}") from error
    if payload["shoe_profile"] == HIGH_HEEL_SHOE_PROFILE:
        print(
            "[normalized] shoe_profile='high_heel'; "
            "inclined support geometry preserved"
        )
    normalization = payload["normalization"]
    assert isinstance(normalization, dict)
    print(
        f"wrote shoe-preparation artifacts to "
        f"{parser_args.output_dir.expanduser().resolve()}"
    )
    print(
        "functional length "
        f"{float(normalization['functional_length']):.9f}; "
        f"outer ratio {float(normalization['outer_length_ratio']):.2%}"
    )


if __name__ == "__main__":
    main()
