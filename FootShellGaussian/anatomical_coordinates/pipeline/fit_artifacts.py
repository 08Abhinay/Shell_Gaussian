"""Write a torch fit in the ``containment_fit`` schema the CPU stages expect.

The downstream chain never learns that a different optimizer produced these.
That is the whole point of the bridge: ``foot_prior`` stays untouched and
keeps validating exactly what it always validated.

The contract, read out of the consumers rather than assumed:

``scripts/run_anatomical_surface.py``   schema_version == 5, shoe_profile
                                        "normal", ``bounds.aligned_foot``,
                                        ``supr.pose_parameters_radians``,
                                        ``supr.betas``, and the four
                                        transforms.
``foot_prior/instance_volume_mapping``  schema_version == 5, shoe_profile
                                        "normal", ``status`` in
                                        {contained_target_fit,
                                        residual_target_fit}, the four
                                        transforms as exact mutual inverses to
                                        1e-12, and ``inputs.normalized_shoe``
                                        pointing at a readable mesh.

Everything else written here is provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.alignment import make_supr_to_shoe_axis_remap
from foot_prior.mesh import TriangleMesh, save_triangle_mesh

from ..shoe_data import ShoeCase


SCHEMA_VERSION = 5
NORMAL_PROFILE = "normal"
SHOE_FUNCTIONAL_LENGTH_MM = 262.5

#: ``instance_volume_mapping`` accepts only these two.
STATUS_CONTAINED = "contained_target_fit"
STATUS_RESIDUAL = "residual_target_fit"


def placement_transforms(
    scale: float, translation: np.ndarray, normalized_to_shoe: np.ndarray
) -> dict[str, np.ndarray]:
    """The four 4x4 matrices, with inverses built analytically.

    ``_load_transforms`` checks each forward/inverse pair multiplies to the
    identity within 1e-12. The remap is a signed permutation, so its inverse is
    its transpose and the whole inverse is exact in floating point rather than
    merely well conditioned - a plain ``linalg.inv`` would usually pass too,
    but there is no reason to spend the accuracy.
    """

    remap = make_supr_to_shoe_axis_remap()[:3, :3]
    forward = np.eye(4, dtype=np.float64)
    forward[:3, :3] = scale * remap
    forward[:3, 3] = np.asarray(translation, dtype=np.float64)

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = remap.T / scale
    inverse[:3, 3] = -(remap.T / scale) @ forward[:3, 3]

    to_shoe = np.asarray(normalized_to_shoe, dtype=np.float64)
    original = to_shoe @ forward
    original_inverse = np.linalg.inv(original)
    # Six, not four. ``_CONTAINMENT_TRANSFORM_NAMES`` also requires the
    # shoe/normalized pair, and ``_INVERSE_TRANSFORM_PAIRS`` checks it inverts
    # to 1e-12 like the others. These two are the preparation's own validated
    # matrices, carried through unchanged.
    return {
        "posed_supr_to_normalized_shoe": forward,
        "normalized_shoe_to_posed_supr": inverse,
        "posed_supr_to_original_shoe": original,
        "original_shoe_to_posed_supr": original_inverse,
        "normalized_to_shoe": to_shoe,
        "shoe_to_normalized": np.linalg.inv(to_shoe),
    }


def _refine_inverse(forward: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """One Newton step on the inverse, so the 1e-12 identity check is safe."""

    for _ in range(3):
        residual = np.eye(4) - forward @ inverse
        if np.abs(residual).max() < 1e-15:
            break
        inverse = inverse + inverse @ residual
    return inverse


def write_containment_fit(
    output_dir: Path,
    case: ShoeCase,
    vertices: np.ndarray,
    faces: np.ndarray,
    betas: np.ndarray,
    pose: np.ndarray,
    scale: float,
    translation: np.ndarray,
    ankle_degrees: float,
    midfoot_degrees: float,
    metrics: dict[str, Any],
    fit_seconds: float,
    config: dict[str, Any],
) -> Path:
    """Write ``containment_fit.json`` and ``foot_containment_fitted.ply``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = TriangleMesh(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )
    save_triangle_mesh(output_dir / "foot_containment_fitted.ply", mesh)

    transforms = placement_transforms(
        scale, translation, case.normalized_to_shoe
    )
    for forward, inverse in (
        ("posed_supr_to_normalized_shoe", "normalized_shoe_to_posed_supr"),
        ("posed_supr_to_original_shoe", "original_shoe_to_posed_supr"),
        ("shoe_to_normalized", "normalized_to_shoe"),
    ):
        transforms[inverse] = _refine_inverse(
            transforms[forward], transforms[inverse]
        )
        worst = np.abs(
            np.eye(4) - transforms[forward] @ transforms[inverse]
        ).max()
        if worst > 1e-13:
            raise ValueError(
                f"{case.name}: {forward} inverse is only accurate to {worst:.2e}"
            )

    clear = (
        metrics.get("status") == "clear"
        and metrics.get("collision_area_fraction", 1.0) < 1e-12
        and metrics.get("outside_area_fraction", 1.0) < 1e-12
    )
    toe_x = float(np.max(mesh.vertices[:, 0]))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "differentiable_containment_fit",
        "shoe_name": case.name,
        "shoe_profile": NORMAL_PROFILE,
        "status": STATUS_CONTAINED if clear else STATUS_RESIDUAL,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "normalized_shoe": str(case.preparation_paths["normalized_shoe"]),
            "normalized_footbed": str(case.preparation_paths["normalized_footbed"]),
            "preparation_directory": str(case.preparation_paths["preparation_dir"]),
            "support_fit_directory": str(case.preparation_paths["support_fit_dir"]),
            "supr_model": str(case.preparation_paths["supr_model"]),
        },
        "source_schema_versions": {
            "shoe_preparation": case.preparation.get("schema_version"),
            "support_fit": case.support_fit.get("schema_version"),
        },
        "bounds": {"aligned_foot": mesh.bounds.tolist()},
        "supr": {
            "betas": np.asarray(betas, dtype=np.float64).tolist(),
            "pose_parameters_radians": np.asarray(pose, dtype=np.float64).tolist(),
            "selected_angles_degrees": {
                "ankle_pitch": float(ankle_degrees),
                "midfoot_pitch": float(midfoot_degrees),
            },
            "reproduction_note": (
                "pose and shape are non-rigid; reproduce them from the stored "
                "parameters through SUPR rather than from a matrix"
            ),
        },
        "placement": {
            "scale": float(scale),
            "translation": np.asarray(translation, dtype=np.float64).tolist(),
            "scale_source": (
                "anchored on the neutral template; foot length is an output of "
                "the betas and is never searched"
            ),
        },
        "transforms": {name: matrix.tolist() for name, matrix in transforms.items()},
        "sizing": {
            "shoe_functional_length_mm": SHOE_FUNCTIONAL_LENGTH_MM,
            "reference_foot_length_mm": 250.0,
            "toe_x": toe_x,
            "toe_allowance_mm": float((1.0 - toe_x) * SHOE_FUNCTIONAL_LENGTH_MM),
            "length_source": (
                "SUPR shape parameters under a fixed anchored scale; not searched"
            ),
        },
        "final_cavity_analysis": {
            "judge": "foot_prior.cavity exact evaluator, unmodified",
            **{
                key: metrics[key]
                for key in sorted(metrics)
                if not isinstance(metrics[key], (dict, list))
            },
        },
        "optimizer": {
            "method": "staged Adam on a baked cavity field, multi-start",
            "implementation": "anatomical_coordinates",
            "seconds": float(fit_seconds),
            "config": config,
        },
    }
    path = output_dir / "containment_fit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
