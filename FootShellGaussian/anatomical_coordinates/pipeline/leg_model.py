"""Differentiable lower-leg exit, replacing the search in ``lower_leg_fit``.

The geometry definition is taken from ``foot_prior`` unchanged and only the
*optimizer* is different. The NumPy version enumerates candidates, evaluates
exact triangle collisions for each, and keeps the best natural one. Here the
same three quantities - full-body SUPR shank, rigid ankle anchoring, and the
joined surface - are built with autograd alive, and the collar objective is the
baked cavity barrier the foot fitter already uses.

Two facts make this cheap:

* the attachment transform maps the *canonical* dense foot's ankle loop onto
  the *fitted* one, so it depends on the fitted foot alone and is a constant
  for the whole optimization; it is solved once in NumPy;
* the donor-foot anchor inside the shank evaluation is an orthogonal Procrustes
  problem, which is differentiable through ``torch.linalg.svd``.

The exact judge is unchanged: the reported collision is
``CavityEvaluator.collision_pairs`` on the leg and bridge faces, exactly as
before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from foot_prior.supr_lower_leg import (
    FULL_BODY_POSE_PARAMETER_COUNT,
    RIGHT_ANKLE_PITCH_PARAMETER_INDEX,
    RIGHT_ANKLE_ROLL_PARAMETER_INDEX,
)

from ..losses import millimetres
from ..cavity_field import ShoeCavityField, sample_field


def _rigid_transform(source: torch.Tensor, target: torch.Tensor):
    """Orthogonal Procrustes, differentiable. Mirrors ``_rigid_correspondence_transform``."""

    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = torch.linalg.svd(covariance)
    # Reflect the smallest singular direction rather than branching, so the
    # correction stays differentiable.
    sign = torch.sign(torch.det(right_t.T @ left.T))
    correction = torch.diag(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign])
    )
    rotation = right_t.T @ correction @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _apply(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor):
    return points @ rotation.T + translation


@dataclass
class LegSetup:
    """Everything shared across shoes, built once."""

    model: Any                      # PosableSuprLowerLegModel
    reference_vertices: np.ndarray  # canonical dense foot
    ankle_loop: np.ndarray          # 60 dense-foot vertex indices
    correspondence: np.ndarray      # (60, 2), column 1 indexes the dense leg
    leg_faces: np.ndarray           # canonical dense lower-leg faces
    foot_faces: np.ndarray          # canonical dense foot faces


class TorchLowerLegFitter:
    """Optimize ankle pitch/roll and bounded shank betas against the collar."""

    def __init__(
        self,
        setup: LegSetup,
        device: torch.device,
        max_abs_beta: float = 1.0,
        max_beta_norm: float = 1.5,
        max_pitch_degrees: float = 20.0,
        max_roll_degrees: float = 15.0,
    ) -> None:
        # These are the search fitter's own natural limits, read out of a
        # stored fit rather than guessed: pitch +/-20, roll +/-15, |beta| <= 1
        # and ||beta||_2 <= 1.5. An earlier version used +/-12 degrees and
        # |beta| <= 2, which put canvas_shoe's 20-degree answer outside the
        # search space while letting the shank distort four times further than
        # the anatomy allows.
        self.setup = setup
        self.device = device
        self.max_abs_beta = float(max_abs_beta)
        self.max_beta_norm = float(max_beta_norm)
        self.max_pitch = float(np.deg2rad(max_pitch_degrees))
        self.max_roll = float(np.deg2rad(max_roll_degrees))

        model = setup.model
        self.supr = model._model
        self.num_betas = int(model.num_betas)
        body_to_reference = np.asarray(
            model.neutral_lower_leg.body_to_reference, dtype=np.float32
        )
        self.b2r_rotation = torch.as_tensor(
            body_to_reference[:3, :3], device=device
        )
        self.b2r_translation = torch.as_tensor(
            body_to_reference[:3, 3], device=device
        )
        self.donor = torch.as_tensor(
            np.asarray(model.donor_foot_vertex_indices, dtype=np.int64), device=device
        )
        self.neutral_reference = torch.as_tensor(
            np.asarray(model.neutral_reference_vertices, dtype=np.float32),
            device=device,
        )
        self.source_indices = torch.as_tensor(
            np.asarray(model.neutral_lower_leg.source_vertex_indices, dtype=np.int64),
            device=device,
        )
        # The subdivision is a fixed linear map on the native shank vertices.
        subdivision = model.subdivision
        weights = np.zeros(
            (subdivision.vertex_count, subdivision.source_vertex_count),
            dtype=np.float32,
        )
        rows = np.arange(subdivision.vertex_count)[:, None]
        valid = subdivision.vertex_source_indices >= 0
        weights[
            np.broadcast_to(rows, valid.shape)[valid],
            subdivision.vertex_source_indices[valid],
        ] = subdivision.vertex_source_weights[valid]
        self.subdivide = torch.as_tensor(weights, device=device)

    # -- forward -----------------------------------------------------------
    def shank(self, betas: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor):
        """Dense shank vertices in the canonical reference frame, with gradient."""

        pose = torch.zeros(
            (1, FULL_BODY_POSE_PARAMETER_COUNT), dtype=torch.float32, device=self.device
        )
        pose = pose.index_put(
            (torch.tensor([0], device=self.device),
             torch.tensor([RIGHT_ANKLE_PITCH_PARAMETER_INDEX], device=self.device)),
            pitch.reshape(1),
        )
        pose = pose.index_put(
            (torch.tensor([0], device=self.device),
             torch.tensor([RIGHT_ANKLE_ROLL_PARAMETER_INDEX], device=self.device)),
            roll.reshape(1),
        )
        translation = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        output = self.supr(pose, betas.reshape(1, -1), translation)
        vertices = output[0]
        vertices = _apply(vertices, self.b2r_rotation, self.b2r_translation)
        rotation, offset = _rigid_transform(
            vertices[self.donor], self.neutral_reference[self.donor]
        )
        vertices = _apply(vertices, rotation, offset)
        native = vertices[self.source_indices]
        return self.subdivide @ native

    def joined(
        self,
        dense_shank: torch.Tensor,
        fitted_foot: torch.Tensor,
        attach_rotation: torch.Tensor,
        attach_translation: torch.Tensor,
    ) -> torch.Tensor:
        """Fitted foot vertices followed by the placed shank vertices."""

        placed = _apply(dense_shank, attach_rotation, attach_translation)
        return torch.cat((fitted_foot, placed), dim=0)

    # -- objective ---------------------------------------------------------
    def collar_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        field: ShoeCavityField,
        margin_mm: float,
        softness_mm: float,
    ) -> torch.Tensor:
        """Penalize the leg only where the shoe actually defines a boundary.

        This is deliberately not the foot's containment barrier, for two
        reasons measured on real shoes.

        The leg leaves the shoe. Most of it is in open air above and behind the
        collar, and the signed field says nothing trustworthy there: with the
        longitudinal ray family on, a calf sitting behind the heel line reads
        as deeply outside. At a neutral leg with *zero* exact collisions the
        barrier read 22632, and minimizing it drove the shank to ||b|| = 4.9
        and 262 real collisions. The loss was anti-correlated with the judge.

        So the constraint is applied only where ``valid`` says a boundary was
        actually found - the collar, the walls, the upper - and only for
        penetration past it. Open space costs nothing, which is the source's
        own contact policy: the leg may pass through the opening but may not
        intersect the collar.
        """

        triangles = vertices[faces]
        centroids = triangles.mean(dim=1)
        corners = triangles.reshape(-1, 3)
        samples = torch.cat((corners, centroids), dim=0)[None]
        with torch.no_grad():
            area = 0.5 * torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=-1,
            ).norm(dim=-1)
            corner_weight = (area / 3.0).repeat_interleave(3)
            weight = torch.cat((corner_weight, area), dim=0)[None]
            weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
            valid = sample_field(
                field.valid, samples, field.lower, field.upper
            )[:, :, 0].clamp(0.0, 1.0)

        directional = sample_field(
            field.clearance, samples, field.lower, field.upper
        )[:, :, 0]
        margin = millimetres(margin_mm)
        softness = millimetres(max(softness_mm, 1e-6))
        violation = softness * functional.softplus(
            (margin - directional) / softness
        )
        residual = violation / millimetres(1.0)
        charged = weight * valid * residual.square()
        return charged.sum(dim=1).mean()

    def fit(
        self,
        fitted_foot: np.ndarray,
        attach: np.ndarray,
        field: ShoeCavityField,
        leg_face_offset: int,
        query_faces: np.ndarray,
        starts: tuple = ((0.0, 0.0),),
        steps: int = 220,
        margin_mm: float = 0.0,
        softness_mm: float = 0.25,
        beta_weight: float = 2.0,
        angle_weight: float = 0.5,
        learning_rate: float = 0.02,
    ) -> dict[str, Any]:
        """Optimize the shank against one shoe's collar."""

        foot = torch.as_tensor(
            np.asarray(fitted_foot, dtype=np.float32), device=self.device
        )
        attach = np.asarray(attach, dtype=np.float32)
        attach_rotation = torch.as_tensor(attach[:3, :3], device=self.device)
        attach_translation = torch.as_tensor(attach[:3, 3], device=self.device)
        faces = torch.as_tensor(
            np.asarray(query_faces, dtype=np.int64), device=self.device
        )

        def arctanh(value: float, bound: float) -> float:
            return float(np.arctanh(np.clip(value / bound, -0.99, 0.99)))

        candidates: list[dict[str, Any]] = []
        for pitch0, roll0 in starts:
            raw_betas = torch.zeros(
                self.num_betas, dtype=torch.float32, device=self.device,
                requires_grad=True,
            )
            raw_pitch = torch.full(
                (1,), arctanh(np.deg2rad(pitch0), self.max_pitch),
                dtype=torch.float32, device=self.device, requires_grad=True,
            )
            raw_roll = torch.full(
                (1,), arctanh(np.deg2rad(roll0), self.max_roll),
                dtype=torch.float32, device=self.device, requires_grad=True,
            )
            optimizer = torch.optim.Adam(
                [raw_betas, raw_pitch, raw_roll], lr=learning_rate
            )
            history: list[dict[str, float]] = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                betas = self.max_abs_beta * torch.tanh(raw_betas)
                pitch = self.max_pitch * torch.tanh(raw_pitch)
                roll = self.max_roll * torch.tanh(raw_roll)
                dense = self.shank(betas, pitch, roll)
                vertices = self.joined(
                    dense, foot, attach_rotation, attach_translation
                )
                collar = self.collar_loss(
                    vertices, faces, field, margin_mm, softness_mm
                )
                # The norm bound is the anatomy's, enforced as a hinge so a
                # shank never leaves the natural envelope the source defines.
                excess = functional.relu(betas.norm() - self.max_beta_norm)
                prior = (
                    beta_weight * betas.square().mean()
                    + 10.0 * excess.square()
                    + angle_weight
                    * ((pitch / self.max_pitch).square()
                       + (roll / self.max_roll).square()).mean()
                )
                (collar + prior).backward()
                optimizer.step()
                if step % 50 == 0 or step == steps - 1:
                    history.append(
                        {"step": step, "collar": float(collar.detach()),
                         "prior": float(prior.detach())}
                    )
            with torch.no_grad():
                betas = self.max_abs_beta * torch.tanh(raw_betas)
                pitch = self.max_pitch * torch.tanh(raw_pitch)
                roll = self.max_roll * torch.tanh(raw_roll)
            candidates.append(
                {
                    "start": [float(pitch0), float(roll0)],
                    "betas": betas.detach().cpu().numpy().astype(np.float64),
                    "ankle_pitch_degrees": float(np.rad2deg(pitch.item())),
                    "ankle_roll_degrees": float(np.rad2deg(roll.item())),
                    "history": history,
                }
            )
        return candidates
