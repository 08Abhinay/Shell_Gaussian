"""Configuration for the differentiable fitter."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    """One optimization stage: which variables move, and how fast."""

    name: str
    steps: int
    learning_rates: dict[str, float]
    active: tuple[str, ...]


@dataclass(frozen=True)
class FitConfig:
    """Objective weights, stage schedule and field resolution."""

    # Objective weights. The ordering contain > support > priors is the
    # structural point from the research report: leaving the shoe must cost more
    # than imperfect seating, which must cost more than anatomical drift.
    w_containment: float = 30.0
    w_support: float = 2.0
    w_penetration: float = 50.0
    w_beta: float = 0.05
    w_pose: float = 0.1
    w_length: float = 1.0
    # ``heel_seating_loss`` restores alignment.py's longitudinal seating, which
    # a free translation otherwise discards. It is weighted 0 by default: it was
    # added to stop the heel reversing through the heel counter, and the
    # measured cause of that was the joint-box containment exemption, not the
    # free translation. With the exemption keyed on the shoe's own opening
    # instead, w_heel=10 costs the golden set 7 of its 8 clear fits by shoving
    # the foot forward into the toe box (mean foot length 233.9 -> 242.1 mm).
    # The term is kept because it is the right statement of the constraint if a
    # future change needs it; the weight is the finding.
    w_heel: float = 0.0
    heel_behind_scale_mm: float = 0.5
    heel_ahead_scale_mm: float = 8.0
    length_scale_mm: float = 5.0

    containment_margin_mm: float = 1.0
    containment_softness_mm: float = 0.5
    containment_scale_mm: float = 1.0
    support_contact_scale_mm: float = 2.0
    support_penetration_scale_mm: float = 0.5
    # Daylight the sole should keep above the footbed. 0.0 reproduces a plain
    # non-penetration constraint; a small positive value stops a soft penalty
    # from settling a few micrometres inside the sole.
    support_safety_mm: float = 0.0
    pose_prior_scale_degrees: float = 5.0
    target_toe_allowance_mm: float = 20.0

    # Anatomy budget. The per-coefficient bound alone permits ||beta||_2 up to
    # 9.49 across ten coefficients, and the fits that spend that much are the
    # ones the shared-volume stages reject: the cage deformation stage has to deform the
    # canonical tetrahedral volume onto the fitted anatomy without inverting an
    # element, and it stalls with "minimum_step_after_jacobian_determinant"
    # when the target is too far from canonical. Measured over 25 shoes, the
    # fits that failed the cage deformation stage averaged ||beta|| 6.85 against 4.79 for those that
    # passed. The norm hinge is the constraint that actually matters, so it is
    # stated directly rather than approximated by tightening the per-coefficient
    # bound.
    max_beta_norm: float = 5.0
    w_beta_norm: float = 5.0

    # Soft bounds, replacing the old hard clamps.
    max_abs_beta: float = 3.0
    max_abs_pitch_degrees: float = 20.0

    # In the ankle exit region the containment sign is dropped and only the
    # unsigned distance to the collar is scored. Without this the optimizer
    # sinks the sole through the footbed to hide an anatomically correct
    # protrusion; see the ablation in the report.
    use_ankle_exemption: bool = True

    # Ablation switch for the +/-X ray family that signs the distance channel.
    # Off restores the X-blind rule under which a sample behind the heel
    # counter was signed positive and the value grew with depth. Kept so the
    # fix can be measured against its own absence; not a tuning knob.
    # Off by default, and the reason matters. The ray family is *correct* -
    # it raises agreement with the exact SAT judge from 56 % to 63 % and stops
    # a sample 20 mm behind the heel counter being signed +9 mm. But turning it
    # on while the footbed is still only a soft penalty makes fits worse
    # (golden set: 7 clear -> 2, samples sunk through the footbed 6/16 ->
    # 12/16): closing the longitudinal escape without closing the vertical one
    # just redirects the violation downward, and a sunk fit is never clear.
    # Enable it together with a hard footbed constraint, not before.
    use_longitudinal_sign: bool = False

    # 0 keeps the native 266-vertex / 515-face topology; 1 and 2 apply the
    # repository's deterministic midpoint subdivision for denser containment
    # sampling only. The SUPR forward pass always stays at 266 vertices.
    # This was 2 while the runner's --subdivision-levels defaulted to 1, so
    # every stored run in fact used 1 and any text implying level 2 was wrong.
    # Set to 1 so the config states what is actually run.
    containment_subdivision_levels: int = 1

    field_spacing_mm: float = 1.0
    field_margin: float = 0.06

    seed: int = 0
    stages: tuple[StageConfig, ...] = field(
        default_factory=lambda: (
            StageConfig(
                "A_translation",
                150,
                {"translation": 2e-3},
                ("translation",),
            ),
            StageConfig(
                "B_pose",
                150,
                {"translation": 2e-3, "pose": 5e-3},
                ("translation", "pose"),
            ),
            StageConfig(
                "C_shape",
                300,
                {"translation": 1e-3, "pose": 3e-3, "betas": 1e-2},
                ("translation", "pose", "betas"),
            ),
            StageConfig(
                "D_refine",
                150,
                {"translation": 3e-4, "pose": 1e-3, "betas": 3e-3},
                ("translation", "pose", "betas"),
            ),
        )
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = [asdict(stage) for stage in self.stages]
        return payload
