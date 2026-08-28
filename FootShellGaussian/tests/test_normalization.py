"""Tests for the fixed canonical right-shoe coordinate contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from foot_prior.normalization import (
    EXPECTED_SHOE_COORDINATE_SYSTEM,
    SHOE_AXIS_SEMANTICS,
    SHOE_SIDE,
    validate_shoe_frame_metadata,
)


DEFAULT_EVAL_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right"
)


def evaluation_root() -> Path:
    return Path(os.environ.get("FOOTSHELL_EVAL_ROOT", DEFAULT_EVAL_ROOT))


def write_metadata(path: Path, coordinate_system: object, mirror_width: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "canonical_geometry": {"mirror_width": mirror_width},
                "reference_mesh": {"coordinate_system": coordinate_system},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("mirror_width", [False, True])
def test_accepts_canonical_metadata_regardless_of_source_mirroring(
    tmp_path: Path, mirror_width: bool
) -> None:
    path = tmp_path / "blender_canonicalization.json"
    write_metadata(path, EXPECTED_SHOE_COORDINATE_SYSTEM, mirror_width)
    assert validate_shoe_frame_metadata(path) is None


def test_rejects_incompatible_coordinate_system(tmp_path: Path) -> None:
    path = tmp_path / "blender_canonicalization.json"
    write_metadata(path, "x_width_y_up_z_length", False)
    with pytest.raises(ValueError, match="unsupported shoe coordinate system"):
        validate_shoe_frame_metadata(path)


def test_rejects_missing_reference_mesh_metadata(tmp_path: Path) -> None:
    path = tmp_path / "blender_canonicalization.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing reference_mesh"):
        validate_shoe_frame_metadata(path)


def test_rejects_malformed_or_missing_metadata(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shoe-frame metadata JSON"):
        validate_shoe_frame_metadata(malformed)
    with pytest.raises(FileNotFoundError):
        validate_shoe_frame_metadata(tmp_path / "missing.json")


def test_right_shoe_axis_contract_is_explicit() -> None:
    assert SHOE_SIDE == "right"
    assert SHOE_AXIS_SEMANTICS == {
        "+X": "heel_to_toe",
        "+Y": "down_toward_sole",
        "+Z": "shoe_width",
    }


def test_all_evaluation_metadata_uses_the_canonical_frame() -> None:
    root = evaluation_root()
    if not root.is_dir():
        pytest.skip(f"external evaluation dataset is unavailable: {root}")
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(scenes) == 17
    for scene in scenes:
        validate_shoe_frame_metadata(scene / "blender_canonicalization.json")
