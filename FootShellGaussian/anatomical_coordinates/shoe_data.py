"""Load one prepared golden-set shoe without duplicating any dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.mesh import TriangleMesh, load_triangle_mesh


GOLDEN_SET_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation"
)
SUPR_MODEL_PATH = Path(
    "/storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


@dataclass(frozen=True)
class ShoeCase:
    """Everything the differentiable fitter needs for one normalized shoe."""

    name: str
    normalized_shoe: TriangleMesh
    normalized_footbed: TriangleMesh
    baseline_foot: TriangleMesh
    footbed_source_face_indices: np.ndarray
    plantar_vertex_indices: np.ndarray
    plantar_face_indices: np.ndarray
    normalized_centerline_xz: np.ndarray
    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    support_grid_cell_spacing: float
    support_compression_allowance: float
    baseline_pose: np.ndarray
    baseline_betas: np.ndarray
    preparation: dict[str, Any]
    support_fit: dict[str, Any]
    #: Where each input came from. Downstream stages record provenance and
    #: ``instance_volume_mapping`` re-reads the normalized shoe by path, so the
    #: locations have to survive loading rather than be reconstructed later.
    preparation_paths: dict[str, Path]

    @property
    def baseline_ankle_degrees(self) -> float:
        return float(
            self.support_fit["supr"]["selected_angles_degrees"]["ankle_pitch"]
        )

    @property
    def baseline_midfoot_degrees(self) -> float:
        return float(
            self.support_fit["supr"]["selected_angles_degrees"]["midfoot_pitch"]
        )


def available_shoes(root: Path = GOLDEN_SET_ROOT) -> list[str]:
    preparation = root / "shoe_preparation"
    return sorted(
        item.name
        for item in preparation.iterdir()
        if (item / "shoe_preparation.json").is_file()
    )


def load_shoe_case(name: str, root: Path = GOLDEN_SET_ROOT) -> ShoeCase:
    """Read the the preparation and seating stages artifacts for one shoe, read-only."""

    preparation_dir = root / "shoe_preparation" / name
    support_dir = root / "support_fit" / name
    preparation = _load_json(preparation_dir / "shoe_preparation.json")
    support_fit = _load_json(support_dir / "support_fit.json")
    if preparation.get("shoe_profile") != "normal":
        raise ValueError(f"{name}: only the 'normal' shoe profile is supported")

    shoe = load_triangle_mesh(preparation_dir / "shoe_normalized.ply")
    footbed = load_triangle_mesh(support_dir / "footbed_normalized.ply")
    baseline_foot = load_triangle_mesh(support_dir / "foot_support_fitted.ply")

    selection = preparation["footbed_selection"]
    footbed_faces = np.asarray(selection["original_face_indices"], dtype=np.int64)
    if len(footbed_faces) != len(footbed.faces):
        raise ValueError(f"{name}: footbed face count disagrees with preparation")

    regions = support_fit["contact_regions"]
    normalization = preparation["normalization"]
    centerline = np.asarray(
        normalization["centerline"]["normalized_xz"], dtype=np.float64
    )
    grid_spacing = float(support_fit["support_grid_cell_spacing"])
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        raise ValueError(f"{name}: invalid support_grid_cell_spacing")

    seating = support_fit.get("seating") or {}
    allowance_mm = float(seating.get("allowance_mm", 0.0)) if seating else 0.0
    functional_length_mm = float(
        support_fit["sizing"]["shoe_functional_length_mm"]
    )

    return ShoeCase(
        name=name,
        normalized_shoe=shoe,
        normalized_footbed=footbed,
        baseline_foot=baseline_foot,
        footbed_source_face_indices=footbed_faces,
        plantar_vertex_indices=np.asarray(
            regions["plantar_vertex_indices"], dtype=np.int64
        ),
        plantar_face_indices=np.asarray(
            regions["plantar_face_indices"], dtype=np.int64
        ),
        normalized_centerline_xz=centerline,
        shoe_to_normalized=_matrix(
            normalization["shoe_to_normalized"], "shoe_to_normalized"
        ),
        normalized_to_shoe=_matrix(
            normalization["normalized_to_shoe"], "normalized_to_shoe"
        ),
        support_grid_cell_spacing=grid_spacing,
        support_compression_allowance=allowance_mm / functional_length_mm,
        baseline_pose=np.asarray(
            support_fit["supr"]["pose_parameters_radians"], dtype=np.float64
        ),
        baseline_betas=np.asarray(support_fit["supr"]["betas"], dtype=np.float64),
        preparation=preparation,
        support_fit=support_fit,
        preparation_paths={
            "preparation_dir": preparation_dir,
            "support_fit_dir": support_dir,
            "normalized_shoe": preparation_dir / "shoe_normalized.ply",
            "normalized_footbed": support_dir / "footbed_normalized.ply",
            "baseline_foot": support_dir / "foot_support_fitted.ply",
            "supr_model": SUPR_MODEL_PATH,
        },
    )
