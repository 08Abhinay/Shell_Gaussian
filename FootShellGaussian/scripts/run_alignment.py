"""Command-line entry point for a single SUPR-to-shoe alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foot_prior.alignment import build_initial_alignment
from foot_prior.footbed import identify_footbed_surface
from foot_prior.mesh import (
    TriangleMesh,
    combine_colored_meshes,
    load_triangle_mesh,
    save_triangle_mesh,
)
from foot_prior.supr_foot import load_neutral_supr_foot


ARTIFACT_NAMES = (
    "alignment.json",
    "foot_aligned.ply",
    "footbed_surface.ply",
    "alignment_overlay.ply",
)
SHOE_COLOR = np.asarray([150, 150, 150, 255], dtype=np.uint8)
FOOT_COLOR = np.asarray([45, 105, 220, 255], dtype=np.uint8)
FOOTBED_COLOR = np.asarray([40, 180, 90, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align one neutral SUPR right foot to one shoe footbed."
    )
    parser.add_argument("--shoe-mesh", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--length-ratio", type=float, default=0.85)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known alignment artifacts if present.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    shoe_path = args.shoe_mesh.expanduser().resolve(strict=True)
    supr_path = args.supr_model.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"alignment artifacts already exist: {formatted}; pass --overwrite to replace them"
        )

    shoe = load_triangle_mesh(shoe_path)
    foot = load_neutral_supr_foot(supr_path)
    footbed = identify_footbed_surface(shoe)
    alignment = build_initial_alignment(
        foot, shoe, footbed, length_ratio=args.length_ratio
    )
    aligned_foot = TriangleMesh(
        alignment.foot_points_to_shoe(foot.vertices), foot.faces
    )
    overlay = combine_colored_meshes(
        (shoe, SHOE_COLOR), (aligned_foot, FOOT_COLOR)
    )

    foot_colors = np.tile(FOOT_COLOR, (len(aligned_foot.vertices), 1))
    footbed_colors = np.tile(FOOTBED_COLOR, (len(footbed.mesh.vertices), 1))
    save_triangle_mesh(targets["foot_aligned.ply"], aligned_foot, foot_colors)
    save_triangle_mesh(
        targets["footbed_surface.ply"], footbed.mesh, footbed_colors
    )
    save_triangle_mesh(targets["alignment_overlay.ply"], overlay)

    payload: dict[str, object] = {
        "schema_version": 1,
        "inputs": {
            "shoe_mesh": str(shoe_path),
            "supr_model": str(supr_path),
        },
        **alignment.to_dict(),
        "footbed_selection": footbed.to_dict(),
    }
    targets["alignment.json"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser_args = parse_args()
    try:
        payload = run(parser_args)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"alignment failed: {error}") from error
    contact = payload["plantar_contact"]
    assert isinstance(contact, dict)
    print(f"wrote alignment artifacts to {parser_args.output_dir.expanduser().resolve()}")
    print(
        "plantar coverage "
        f"{contact['covered_sample_count']}/{contact['sample_count']} "
        f"({float(contact['coverage']):.2%})"
    )


if __name__ == "__main__":
    main()
