"""Judge a fitted foot with the existing exact NumPy evaluator.

Nothing here is differentiable and nothing here is reused by the optimizer.
That separation is the point: the torch objective decides where the foot goes,
and ``foot_prior.cavity`` decides whether that was any good, using the same
exact SAT collision and signed-clearance code the old fitter is scored by.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from foot_prior.cavity import CavityAnalysis, CavityEvaluator
from foot_prior.containment import affected_area_fraction, collision_area_fraction
from foot_prior.mesh import TriangleMesh

from .shoe_data import ShoeCase


SHOE_FUNCTIONAL_LENGTH_MM = 262.5


def build_evaluator(case: ShoeCase) -> CavityEvaluator:
    return CavityEvaluator.build(
        case.normalized_shoe,
        case.normalized_footbed,
        case.footbed_source_face_indices,
        case.baseline_foot,
        case.normalized_centerline_xz,
        case.support_compression_allowance,
    )


def exact_metrics(
    evaluator: CavityEvaluator,
    case: ShoeCase,
    vertices: np.ndarray,
    faces: np.ndarray,
    betas: np.ndarray | None = None,
    ankle_degrees: float | None = None,
    midfoot_degrees: float | None = None,
) -> dict[str, Any]:
    """Return the same measurements the old fitter is judged by."""

    mesh = TriangleMesh(np.asarray(vertices, dtype=np.float64), faces)
    analysis: CavityAnalysis = evaluator.analyze(
        mesh, case.plantar_vertex_indices, case.plantar_face_indices
    )
    signed = analysis.signed_clearance
    collision_area, collision_fraction, colliding = collision_area_fraction(
        mesh, analysis.collision_pairs
    )
    affected_area, affected_fraction, affected = affected_area_fraction(
        mesh, colliding, signed.outside_face_indices
    )
    protrusion = signed.protrusion_statistics
    plantar = analysis.support_contact["plantar_vertices"]
    centroids = analysis.support_contact["plantar_face_centroids"]
    toe_allowance = float(
        (1.0 - float(np.max(mesh.vertices[:, 0]))) * SHOE_FUNCTIONAL_LENGTH_MM
    )
    return {
        "status": analysis.status,
        "collision_pair_count": int(len(analysis.collision_pairs)),
        "colliding_foot_face_count": int(len(colliding)),
        "collision_area_fraction": float(collision_fraction),
        "affected_foot_face_count": int(len(affected)),
        "affected_area_fraction": float(affected_fraction),
        "outside_face_count": int(len(signed.outside_face_indices)),
        "outside_vertex_count": int(len(signed.outside_vertex_indices)),
        "outside_area_fraction": float(signed.outside_area_fraction),
        "maximum_protrusion_depth": float(
            protrusion.get("maximum_protrusion_depth") or 0.0
        ),
        "maximum_protrusion_mm": float(
            (protrusion.get("maximum_protrusion_depth") or 0.0)
            * SHOE_FUNCTIONAL_LENGTH_MM
        ),
        "protrusion_energy": float(signed.protrusion_energy),
        "minimum_signed_clearance": (
            None
            if protrusion.get("minimum_signed_clearance") is None
            else float(protrusion["minimum_signed_clearance"])
        ),
        "plantar_coverage": float(plantar["coverage"]),
        "plantar_median_gap_mm": (
            None
            if plantar["median_gap"] is None
            else float(plantar["median_gap"] * SHOE_FUNCTIONAL_LENGTH_MM)
        ),
        "plantar_minimum_gap_mm": (
            None
            if plantar["minimum_gap"] is None
            else float(plantar["minimum_gap"] * SHOE_FUNCTIONAL_LENGTH_MM)
        ),
        "plantar_contacting_count": int(plantar["contacting_sample_count"]),
        "plantar_excess_penetration_count": int(
            plantar["excess_penetration_count"]
        ),
        "centroid_excess_penetration_count": int(
            centroids["excess_penetration_count"]
        ),
        "toe_allowance_mm": toe_allowance,
        # Where the heel actually sits, in millimetres past the functional
        # heel. ``alignment.py`` placed the foot so this was >= 0 by
        # construction; a free translation can make it negative, which means
        # the heel has gone into or through the heel counter. Reported, never
        # optimized here - the objective's own heel term is separate.
        "heel_x_mm": float(
            np.min(mesh.vertices[case.plantar_vertex_indices, 0])
            * SHOE_FUNCTIONAL_LENGTH_MM
        ),
        "foot_length_mm": float(
            np.ptp(mesh.vertices[:, 0]) * SHOE_FUNCTIONAL_LENGTH_MM
        ),
        "foot_width_mm": float(
            np.ptp(mesh.vertices[:, 2]) * SHOE_FUNCTIONAL_LENGTH_MM
        ),
        "foot_height_mm": float(
            np.ptp(mesh.vertices[:, 1]) * SHOE_FUNCTIONAL_LENGTH_MM
        ),
        "beta_l2_norm": (
            None if betas is None else float(np.linalg.norm(betas))
        ),
        "beta_max_abs": (
            None if betas is None else float(np.max(np.abs(betas)))
        ),
        "ankle_pitch_degrees": ankle_degrees,
        "midfoot_pitch_degrees": midfoot_degrees,
    }


def selection_metrics(
    evaluator: CavityEvaluator,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    """A cheap exact score for ranking restarts.

    ``CavityEvaluator.analyze`` spends most of its time in
    ``_closest_points_on_triangles``, which is only needed for the clearance
    report. Ranking restarts needs the two quantities the fit is judged on -
    exact SAT collisions and signed-outside area - so this computes just those.
    It is still the exact evaluator, never the differentiable loss.
    """

    mesh = TriangleMesh(np.asarray(vertices, dtype=np.float64), faces)
    pairs, _ = evaluator.collision_pairs(mesh)
    signed = evaluator.signed_clearances(mesh)
    _, collision_fraction, colliding = collision_area_fraction(mesh, pairs)
    _, affected_fraction, _ = affected_area_fraction(
        mesh, colliding, signed.outside_face_indices
    )
    return {
        "collision_area_fraction": float(collision_fraction),
        "outside_area_fraction": float(signed.outside_area_fraction),
        "affected_area_fraction": float(affected_fraction),
    }
