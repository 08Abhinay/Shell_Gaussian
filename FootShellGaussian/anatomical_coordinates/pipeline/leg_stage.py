"""Stage 3 (torch): fit the lower-leg exit and publish attachment artifacts.

``foot_prior.supr_lower_leg.attach_lower_leg_to_fitted_dense_foot`` still does
the joining and all of its validation - one 68-vertex knee opening, untouched
foot prefix, connected surface. Only the search that chose the shank pose and
shape is replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foot_prior.anatomy import (
    build_extended_canonical_supr_anatomy,
    load_dense_canonical_supr_reference,
)
from foot_prior.cavity import CavityEvaluator, _find_self_collision_pairs
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh
from foot_prior.supr_foot import build_supr_mesh_subdivision
from foot_prior.anatomical_volume import (
    ENVELOPE_CENTER,
    ENVELOPE_POWER,
    ENVELOPE_RADII,
)
from foot_prior.supr_lower_leg import (
    attach_lower_leg_to_fitted_dense_foot,
    load_posable_supr_lower_leg,
)

# The natural-shape test is imported rather than restated, so this and the
# original NumPy fit cannot drift apart on what counts as a plausible shank.
from foot_prior.lower_leg_fit import (
    _metric_ratios,
    _natural_shape,
    _shape_metrics,
)

from . import leg_search
from ..losses import millimetres
from ..cavity_field import build_cavity_field
from .leg_model import LegSetup, TorchLowerLegFitter
from .shoe_set import UNIFIED_INPUT_ROOT, load_case


DENSE_VERTEX_COUNT = 4151
SCHEMA_VERSION = 2
STAGE = "fitted_foot_natural_lower_leg_collar_fit"
ARTIFACTS = ("lower_leg_attachment.json", "foot_lower_leg.ply")


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_setup(anatomical_root: Path, body_model: Path):
    """Load the canonical anatomy and the posable shank, once per process."""

    reference = load_dense_canonical_supr_reference(anatomical_root)
    canonical = build_extended_canonical_supr_anatomy(reference, body_model)
    subdivision = build_supr_mesh_subdivision(
        canonical.lower_leg.mesh.faces, len(canonical.lower_leg.mesh.vertices), 2
    )
    model = load_posable_supr_lower_leg(
        body_model, canonical.lower_leg, subdivision, num_betas=10
    )
    correspondence = canonical.ankle_correspondence.copy()
    correspondence[:, 1] -= DENSE_VERTEX_COUNT
    return reference, LegSetup(
        model=model,
        reference_vertices=reference.vertices,
        ankle_loop=reference.ankle_loop,
        correspondence=correspondence,
        leg_faces=subdivision.faces,
        foot_faces=reference.faces,
    ), subdivision


def fit_shoe(
    name: str,
    setup: LegSetup,
    reference,
    fitter: TorchLowerLegFitter,
    anatomical_root: Path,
    input_root: Path,
    steps: int,
) -> dict[str, Any]:
    """Fit one shoe's shank and return the joined mesh plus an exact report."""

    case = load_case(name, input_root)
    dense_foot = load_triangle_mesh(
        anatomical_root / name / "foot_dense.ply"
    )
    if dense_foot.vertices.shape != (DENSE_VERTEX_COUNT, 3):
        raise ValueError(f"{name}: dense foot is not canonical")

    # The neutral shank fixes the topology, the face index ranges and the
    # constant ankle transform. Everything the optimizer changes is vertices.
    neutral = setup.model.evaluate(np.zeros(10), 0.0, 0.0)
    layout = attach_lower_leg_to_fitted_dense_foot(
        dense_foot,
        setup.reference_vertices,
        neutral.dense_vertices,
        setup.leg_faces,
        setup.ankle_loop,
        setup.correspondence,
    )
    query_faces = layout.mesh.faces[
        np.concatenate((layout.lower_leg_face_indices, layout.bridge_face_indices))
    ]

    field = build_cavity_field(
        case.normalized_shoe,
        case.normalized_footbed,
        case.footbed_source_face_indices,
        case.normalized_centerline_xz,
        np.stack((layout.mesh.vertices.min(axis=0), layout.mesh.vertices.max(axis=0))),
        target_spacing=millimetres(1.0),
        margin=0.06,
        # The leg lives mostly outside the cavity, where the longitudinal sign
        # is not meaningful; the collar constraint uses the directional
        # channel's own validity instead.
        longitudinal_sign=False,
    )
    # Exact judge, unchanged: triangle collisions of leg and bridge vs the shoe.
    evaluator = CavityEvaluator.build(
        case.normalized_shoe,
        case.normalized_footbed,
        case.footbed_source_face_indices,
        dense_foot,
        case.normalized_centerline_xz,
    )

    def envelope_excess(vertices: np.ndarray) -> float:
        """How far the anatomy pushes past the frozen outer envelope.

        the cage deformation stage rejects a shoe outright when any boundary-target vertex
        leaves this fixed superellipsoid, and the ankle pose is what usually
        pushes it out - roll especially, because the envelope is tightest
        across the width. Discovering that three stages later costs a whole
        batch, so the constraint is evaluated here, on the candidate itself.
        """

        equation = np.sum(
            np.abs((vertices - ENVELOPE_CENTER) / ENVELOPE_RADII) ** ENVELOPE_POWER,
            axis=1,
        )
        return float(np.max(equation))

    native_faces = setup.model.neutral_lower_leg.mesh.faces
    neutral_metrics = _shape_metrics(neutral, native_faces)
    neutral_triangles = layout.mesh.vertices[query_faces]
    neutral_areas = 0.5 * np.linalg.norm(
        np.cross(
            neutral_triangles[:, 1] - neutral_triangles[:, 0],
            neutral_triangles[:, 2] - neutral_triangles[:, 0],
        ),
        axis=1,
    )
    # One median triangle's worth of area. Two fits whose contact differs by
    # less than this are the same fit, and the choice between them should be
    # made on how natural the shank is rather than on a rounding difference.
    quantum = float(np.median(neutral_areas) / neutral_areas.sum())

    def evaluate(betas, pitch, roll):
        shape = np.asarray(betas, dtype=np.float64)
        posed = setup.model.evaluate(shape, pitch, roll)
        ratios = _metric_ratios(_shape_metrics(posed, native_faces), neutral_metrics)
        if not _natural_shape(shape, ratios):
            return None
        attachment = attach_lower_leg_to_fitted_dense_foot(
            dense_foot,
            setup.reference_vertices,
            posed.dense_vertices,
            setup.leg_faces,
            setup.ankle_loop,
            setup.correspondence,
        )
        indices = np.concatenate(
            (attachment.lower_leg_face_indices, attachment.bridge_face_indices)
        )
        query = TriangleMesh(
            attachment.mesh.vertices, attachment.mesh.faces[indices]
        )
        pairs, _ = evaluator.collision_pairs(query)
        hits = np.unique(pairs[:, 0]) if len(pairs) else np.empty(0, dtype=np.int64)
        triangles = query.vertices[query.faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        total = float(areas.sum())
        return leg_search.Candidate(
            betas=shape,
            pitch=float(pitch),
            roll=float(roll),
            collision_pairs=int(len(pairs)),
            collision_area_fraction=(
                float(areas[hits].sum() / total) if total else 0.0
            ),
            envelope_maximum=envelope_excess(attachment.mesh.vertices),
            payload={
                "attachment": attachment,
                "posed": posed,
                "colliding_face_count": int(len(hits)),
            },
        )

    def joined_surface_is_sound(candidate) -> bool:
        """Does the joined foot-and-leg surface pass through itself?

        Asked only of the finalists: it is far more expensive than the shoe
        collision test, and a fit that fails it is disqualified however
        little it touches the shoe.
        """

        mesh = candidate.payload["attachment"].mesh
        scale = max(float(np.max(np.abs(mesh.vertices))), 1.0)
        pairs = _find_self_collision_pairs(
            mesh.vertices, mesh.faces,
            512.0 * np.finfo(np.float64).eps * scale,
        )
        if not len(pairs):
            return True
        foot_faces = candidate.payload["attachment"].foot_face_count
        return not bool(np.any(np.any(pairs >= foot_faces, axis=1)))

    report = leg_search.search(
        evaluate, quantum, verify=joined_surface_is_sound
    )
    best = report.best
    result = {
        "betas": np.asarray(best.betas, dtype=np.float64),
        "ankle_pitch_degrees": float(best.pitch),
        "ankle_roll_degrees": float(best.roll),
        "history": [],
        "evaluated_candidates": report.evaluated,
        "rejected_unnatural": report.rejected,
        "stopping_reason": report.stopping_reason,
        "selected_start": report.selected_start,
        "collision_equivalence_area_fraction": quantum,
        "pose_only": {
            "ankle_pitch_degrees": float(report.best_pose_only.pitch),
            "ankle_roll_degrees": float(report.best_pose_only.roll),
            "collision_pair_count": report.best_pose_only.collision_pairs,
            "collision_area_fraction": report.best_pose_only.collision_area_fraction,
        },
        "neutral": {
            "collision_pair_count": report.baseline.collision_pairs,
            "collision_area_fraction": report.baseline.collision_area_fraction,
        },
    }
    attachment = best.payload["attachment"]
    return {
        "case": case,
        "attachment": attachment,
        "result": result,
        "selected_candidate": report.selected_start,
        "collision_pair_count": best.collision_pairs,
        "colliding_face_count": best.payload["colliding_face_count"],
        "collision_area_fraction": best.collision_area_fraction,
        "envelope_maximum": best.envelope_maximum,
        "ankle_loop_rms_residual": float(attachment.ankle_loop_rms_residual),
        "donor_foot_anchor_rms": float(best.payload["posed"].donor_foot_anchor_rms),
    }


def write_attachment(output_dir: Path, name: str, record: dict, inputs: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attachment = record["attachment"]
    save_triangle_mesh(output_dir / "foot_lower_leg.ply", attachment.mesh)
    result = record["result"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "shoe_name": name,
        "shoe_profile": "normal",
        "status": (
            "clear_exit" if record["collision_pair_count"] == 0 else "residual_collar_contact"
        ),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "fit": {
            "method": (
                "coarse pose grid, local pose refinement, shape coordinate "
                "descent, then joint pose-and-shape refinement from several "
                "starts; every candidate judged by the exact evaluator"
            ),
            # Read off the joined mesh rather than asserted, so a topology
            # change shows up here instead of being echoed back as canonical.
            "attachment": {
                "counts": {
                    "foot_vertices": int(attachment.foot_vertex_count),
                    "foot_faces": int(attachment.foot_face_count),
                    "lower_leg_vertices": int(len(attachment.lower_leg_vertex_indices)),
                    "lower_leg_faces": int(len(attachment.lower_leg_face_indices)),
                    "bridge_faces": int(len(attachment.bridge_face_indices)),
                },
                "canonical_leg_to_fitted_ankle": np.asarray(
                    attachment.canonical_leg_to_fitted_ankle, dtype=float
                ).tolist(),
                "ankle_loop_rms_residual": float(attachment.ankle_loop_rms_residual),
            },
            "selected": {
                "betas": np.asarray(result["betas"], dtype=float).tolist(),
                "ankle_pitch_degrees": result["ankle_pitch_degrees"],
                "ankle_roll_degrees": result["ankle_roll_degrees"],
                "collision_pair_count": record["collision_pair_count"],
                "colliding_query_face_count": record["colliding_face_count"],
                "collision_area_fraction": record["collision_area_fraction"],
                "ankle_loop_rms_residual": record["ankle_loop_rms_residual"],
                "donor_foot_anchor_rms": record["donor_foot_anchor_rms"],
                "candidate": record["selected_candidate"],
                "beta_l2": float(np.linalg.norm(np.asarray(result["betas"], dtype=float))),
                "evaluated_candidates": result["evaluated_candidates"],
                "envelope_maximum": record["envelope_maximum"],
            },
            # What the search had to spend to get there, so a poor fit can be
            # told apart from a shoe that simply cannot take this leg.
            "search": {
                "evaluated_candidates": result["evaluated_candidates"],
                "rejected_unnatural_shapes": result["rejected_unnatural"],
                "stopping_reason": result["stopping_reason"],
                "selected_start": result["selected_start"],
                "collision_equivalence_area_fraction": result[
                    "collision_equivalence_area_fraction"
                ],
                "neutral": result["neutral"],
                "best_pose_only": result["pose_only"],
            },
            "history": result["history"],
            "policy": (
                "a natural shaft is preserved - shape stays within the limits a "
                "real shank obeys - and a residual collar intersection is "
                "reported honestly rather than optimized away by distorting the leg"
            ),
        },
    }
    (output_dir / "lower_leg_attachment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anatomical-surface-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full-body-supr-model", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=UNIFIED_INPUT_ROOT)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--shoes", nargs="*", default=None)
    args = parser.parse_args()

    names = args.shoes or sorted(
        p.name for p in args.anatomical_surface_root.iterdir()
        if p.is_dir() and p.name != "reference"
    )
    reference, setup, subdivision = build_setup(
        args.anatomical_surface_root, args.full_body_supr_model
    )
    device = torch.device("cuda")
    fitter = TorchLowerLegFitter(setup, device)
    for name in names:
        try:
            record = fit_shoe(
                name, setup, reference, fitter,
                args.anatomical_surface_root, args.input_root, args.steps,
            )
            write_attachment(
                args.output_root / name, name, record,
                {
                    "anatomical_surface": str(args.anatomical_surface_root / name),
                    "fitted_dense_foot_sha256": _digest(
                        args.anatomical_surface_root / name / "foot_dense.ply"
                    ),
                    "full_body_supr_model": str(args.full_body_supr_model),
                    "shoe_preparation": str(args.input_root / "shoe_preparation" / name),
                    "support_fit": str(args.input_root / "support_fit" / name),
                    "source_anatomical_schema": 1,
                },
            )
            print(
                f"[leg] {name:46s} pairs {record['collision_pair_count']:5d} "
                f"area {record['collision_area_fraction']:.4f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            print(f"[leg] {name:46s} FAILED: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
