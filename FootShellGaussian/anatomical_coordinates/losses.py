"""The four objective terms, each normalized to a physical millimetre scale.

Weights are only meaningful once every residual is dimensionless, so each term
divides by an explicit tolerance expressed in millimetres rather than in raw
normalized-shoe units.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .cavity_field import ShoeCavityField, sample_field, sample_footbed_height


SHOE_FUNCTIONAL_LENGTH_MM = 262.5


def millimetres(value_mm: float) -> float:
    """Convert millimetres to normalized shoe units."""

    return float(value_mm) / SHOE_FUNCTIONAL_LENGTH_MM


@dataclass(frozen=True)
class LossTerms:
    containment: torch.Tensor
    support: torch.Tensor
    penetration: torch.Tensor
    beta_prior: torch.Tensor
    pose_prior: torch.Tensor
    length: torch.Tensor
    total: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "containment",
                "support",
                "penetration",
                "beta_prior",
                "pose_prior",
                "length",
                "total",
            )
        }


def containment_loss(
    samples: torch.Tensor,
    field: ShoeCavityField,
    margin_mm: float = 0.0,
    softness_mm: float = 0.5,
    scale_mm: float = 1.0,
    exempt_mask: torch.Tensor | None = None,
    area_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize foot samples that approach or cross their cavity boundary.

    A softplus barrier rather than a hard ReLU, so a vertex starts feeling the
    wall slightly before it touches: that gives the optimizer a gradient while
    the configuration is still feasible.

    Open space - where the directional clearance finds no boundary at all, such
    as above the collar - is *not* weighted out. It is scored on the unsigned
    distance channel alone, so the foot is still forbidden from intersecting the
    collar while never being charged for legitimately passing through the
    opening. No boundary is ever invented where the source defines none.
    """

    directional = sample_field(
        field.clearance,
        samples,
        field.lower,
        field.upper,
        extend_outside=True,
    )[:, :, 0]
    valid = sample_field(
        field.valid, samples, field.lower, field.upper
    )[:, :, 0].clamp(0.0, 1.0)
    signed_distance = sample_field(
        field.distance,
        samples,
        field.lower,
        field.upper,
        extend_outside=True,
    )[:, :, 0]
    # The signed distance is the term that actually stops the foot, because it
    # is the only one that sees the toe box and the heel counter. Where the
    # directional clearance is defined it is taken as well and the more
    # conservative of the two wins: inside the cavity the true distance to the
    # surface is never larger than the axis-aligned clearance, and outside it
    # the pair bracket the violation.
    clearance = torch.where(
        valid > 0.5, torch.minimum(signed_distance, directional), signed_distance
    )
    if exempt_mask is not None:
        # The mask comes from a bilinear read of the shoe's own opening, so it
        # is soft at the rim. Blending rather than thresholding keeps the loss
        # continuous as a sample crosses it; a binary mask reduces to the
        # previous behaviour exactly.
        weight = exempt_mask.clamp(0.0, 1.0)
        clearance = (1.0 - weight) * clearance + weight * signed_distance.abs()
    # Every sample is scored. Open space is not masked out, because the collar
    # is still a surface the foot may not intersect - the sign there is
    # positive, so only genuine proximity costs anything.
    #
    # Samples are weighted by the surface area they represent rather than
    # counted equally. A plain mean makes the loss depend on the sampling
    # density: subdividing the foot quadruples the interior samples and dilutes
    # the few that actually violate, which was measured undoing an otherwise
    # good fit. Area weighting makes the term invariant to subdivision and puts
    # it in the same units as the area fraction the exact judge reports.
    weight = (
        torch.ones_like(clearance) if area_weight is None else area_weight
    )
    margin = millimetres(margin_mm)
    softness = millimetres(softness_mm)
    violation = softness * functional.softplus(
        (margin - clearance) / softness
    )
    residual = violation / millimetres(scale_mm)
    total_weight = weight.sum(dim=1).clamp(min=1.0)
    loss = (weight * residual.square()).sum(dim=1) / total_weight
    return loss.mean(), clearance


def support_loss(
    vertices: torch.Tensor,
    field: ShoeCavityField,
    region_indices: dict[str, torch.Tensor],
    plantar_indices: torch.Tensor,
    compression_allowance: float,
    contact_scale_mm: float = 2.0,
    penetration_scale_mm: float = 0.5,
    softmin_mm: float = 1.0,
    safety_mm: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seat the load-bearing sole on the footbed without flattening the arch.

    Two distinct statements, kept separate on purpose:

      contact     - each load-bearing region (heel, forefoot) should have some
                    part resting on the footbed. This is a soft *minimum* over
                    the region, not a mean: a plantar surface is curved, so
                    demanding that every heel vertex simultaneously touch is
                    over-constrained and was found to crowd out containment
                    entirely. For reference the existing pipeline accepts a
                    region RMS gap up to 7.9 mm, so a per-vertex 2 mm target was
                    far stricter than anything downstream ever required.
      penetration - no plantar sample may sink through the footbed beyond the
                    compression allowance the pipeline already defines. This one
                    *is* a mean over every plantar sample, because it must hold
                    everywhere rather than somewhere.

    The gap sign follows ``foot_prior.cavity._support_record``: shoe +Y points
    down, so ``gap = footbed_y - vertex_y`` is positive when the foot floats.
    """

    plantar = vertices[:, plantar_indices]
    height, coverage = sample_footbed_height(field, plantar[:, :, (0, 2)])
    gap = height - plantar[:, :, 1]

    softness = millimetres(softmin_mm)
    contact_terms = []
    for indices in region_indices.values():
        mask = torch.zeros(
            vertices.shape[1], dtype=torch.bool, device=vertices.device
        )
        mask[indices] = True
        selected = mask[plantar_indices][None, :] & (coverage > 0.5)
        if not bool(selected.any()):
            continue
        masked = torch.where(selected, gap, torch.full_like(gap, 1e6))
        # Smooth minimum: the shallowest gap in the region, differentiably.
        region_minimum = -softness * torch.logsumexp(
            -masked / softness, dim=1
        )
        contact_terms.append(
            (functional.relu(region_minimum) / millimetres(contact_scale_mm)).square()
        )
    contact = (
        torch.stack(contact_terms, dim=0).mean(dim=0).mean()
        if contact_terms
        else vertices.new_zeros(())
    )

    # ``safety_mm`` asks for a little daylight rather than merely non-negative
    # gap. A soft penalty settles wherever its gradient balances, which was
    # measured landing 4-62 um *below* the footbed - geometrically contained
    # fits that the exact judge still refuses to call clear. The old fitter
    # never has this problem because it seats at first contact by construction.
    excess = functional.relu(
        millimetres(safety_mm) - gap - compression_allowance
    )
    penetration = (
        coverage * (excess / millimetres(penetration_scale_mm)).square()
    ).sum(dim=1) / coverage.sum(dim=1).clamp(min=1.0)
    return contact, penetration.mean(), gap


def beta_prior(betas: torch.Tensor) -> torch.Tensor:
    """Keep shape plausible so containment is not solved by shrinking."""

    return betas.square().mean()


def pose_prior(
    ankle_radians: torch.Tensor,
    midfoot_radians: torch.Tensor,
    ankle_initial: torch.Tensor,
    midfoot_initial: torch.Tensor,
    scale_degrees: float = 5.0,
) -> torch.Tensor:
    """Charge for articulation away from the deterministic initialization."""

    scale = torch.deg2rad(
        torch.as_tensor(scale_degrees, device=ankle_radians.device)
    )
    return (
        ((ankle_radians - ankle_initial) / scale).square()
        + ((midfoot_radians - midfoot_initial) / scale).square()
    ).mean()


def heel_seating_loss(
    vertices: torch.Tensor,
    plantar_indices: torch.Tensor,
    behind_scale_mm: float = 0.5,
    ahead_scale_mm: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the heel seated at the functional heel origin.

    ``alignment.py`` places the foot by construction: ``translation_x =
    heel_offset - min(plantar_x)`` with ``heel_offset >= 0``, so the heel sits
    at the shoe's functional heel and never behind it. Replacing that with a
    free 3-DOF translation threw the constraint away, and the foot was measured
    drifting backwards until heel vertices sat 0.02-0.26 mm from the heel
    counter - touching or through it - while 26-40 mm of toe space went unused.

    The penalty is deliberately asymmetric, because the two directions are not
    equally wrong: going *behind* the origin means passing into the heel
    counter, so it is steep; drifting *ahead* of it only means the foot is not
    pushed fully back, so it is gentle.
    """

    heel_x = vertices[:, plantar_indices, 0].min(dim=1).values
    behind = functional.relu(-heel_x) / millimetres(behind_scale_mm)
    ahead = functional.relu(heel_x) / millimetres(ahead_scale_mm)
    return (behind.square() + ahead.square()).mean(), heel_x


def length_residual(
    vertices: torch.Tensor, target_toe_allowance_mm: float = 20.0
) -> torch.Tensor:
    """Toe allowance implied by the current foot, in millimetres.

    Reported for every fit. It only enters the objective when its weight is
    non-zero, which keeps the old hard 18-22 mm gate out of the optimizer while
    still measuring the quantity it was protecting.
    """

    toe_x = vertices[:, :, 0].max(dim=1).values
    allowance_mm = (1.0 - toe_x) * SHOE_FUNCTIONAL_LENGTH_MM
    return allowance_mm - float(target_toe_allowance_mm)
