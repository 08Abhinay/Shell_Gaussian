"""Canonical right-shoe coordinate-frame validation."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED_SHOE_COORDINATE_SYSTEM = "effective_gshell_x_length_y_down_z_width"
SHOE_SIDE = "right"
SHOE_AXIS_SEMANTICS = {
    "+X": "heel_to_toe",
    "+Y": "down_toward_sole",
    "+Z": "shoe_width",
}


def validate_shoe_frame_metadata(path: str | Path) -> None:
    """Require metadata for the project's fixed canonical right-shoe frame."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid shoe-frame metadata JSON: {source}") from error
    if not isinstance(metadata, dict):
        raise ValueError("shoe-frame metadata must contain a JSON object")

    reference_mesh = metadata.get("reference_mesh")
    if not isinstance(reference_mesh, dict):
        raise ValueError("shoe-frame metadata is missing reference_mesh")
    coordinate_system = reference_mesh.get("coordinate_system")
    if coordinate_system != EXPECTED_SHOE_COORDINATE_SYSTEM:
        raise ValueError(
            "unsupported shoe coordinate system: "
            f"expected {EXPECTED_SHOE_COORDINATE_SYSTEM!r}, "
            f"received {coordinate_system!r}"
        )
