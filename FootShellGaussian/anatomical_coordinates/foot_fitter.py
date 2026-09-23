"""Staged Adam fitting of SUPR shape, pitch and translation inside a shoe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from foot_prior.alignment import make_supr_to_shoe_axis_remap, neutral_length_scale
from foot_prior.mesh import TriangleMesh
from foot_prior.supr_foot import build_supr_mesh_subdivision

from .config import FitConfig
from .losses import (
    SHOE_FUNCTIONAL_LENGTH_MM,
    beta_prior,
    containment_loss,
    heel_seating_loss,
    length_residual,
    pose_prior,
    support_loss,
)
from .cavity_field import ShoeCavityField, sample_open_above
from .supr_torch import TorchSuprFoot


@dataclass
class FitResult:
    """Fitted parameters plus everything needed to hand off downstream."""

    vertices: np.ndarray  # (B, V, 3) in the normalized shoe frame
    betas: np.ndarray
    pose: np.ndarray
    ankle_degrees: np.ndarray
    midfoot_degrees: np.ndarray
    translation: np.ndarray
    scale: float
    history: list[dict[str, Any]]
    final_losses: dict[str, float]
    toe_allowance_mm: np.ndarray
    seconds: float
    peak_memory_bytes: int

    def transform(self, index: int = 0) -> np.ndarray:
        """Return the 4x4 posed-SUPR -> normalized-shoe matrix."""

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * make_supr_to_shoe_axis_remap()[:3, :3]
        matrix[:3, 3] = self.translation[index]
        return matrix


class ShoeFootFitter:
    """Holds the fixed geometry; ``fit`` runs the staged Adam schedule."""

    def __init__(
        self,
        model: TorchSuprFoot,
        neutral_foot: TriangleMesh,
        config: FitConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or FitConfig()
        self.device = model.device
        # The anchored SUPR->shoe scale is derived from the neutral template
        # alone and is deliberately NOT a free variable: foot length must stay
        # an output of the betas rather than something normalized away.
        self.scale = float(neutral_length_scale(neutral_foot))
        remap = make_supr_to_shoe_axis_remap()[:3, :3]
        self.remap = torch.as_tensor(
            remap, dtype=torch.float32, device=self.device
        )
        # Containment is scored on a deterministically subdivided copy. The
        # exact judge tests triangle intersections, and a 515-face foot has
        # edges long enough that a face can straddle a thin shoe wall while
        # every one of its sample points is comfortably clear of the surface.
        # Subdivision is a fixed linear map on the source vertices, so it costs
        # one matmul and stays differentiable; the represented surface is
        # unchanged. ``foot_prior.supr_foot`` already builds the recipe.
        levels = int(self.config.containment_subdivision_levels)
        subdivision = build_supr_mesh_subdivision(
            np.asarray(model.faces.cpu().numpy(), dtype=np.int64),
            len(neutral_foot.vertices),
            levels,
        )
        self.subdivision = subdivision
        self.faces = torch.as_tensor(
            subdivision.faces, dtype=torch.long, device=self.device
        )
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
        self.subdivide = torch.as_tensor(weights, device=self.device)
        # Per-vertex incident faces, padded to the maximum degree. Accumulating
        # vertex areas with scatter_add_ instead uses CUDA atomics, whose
        # summation order varies between runs; that was measured making two
        # identical invocations diverge to visibly different betas, because the
        # tiny weight differences amplify through 750 Adam steps of a
        # non-convex objective. A padded gather is deterministic.
        incidence: list[list[int]] = [
            [] for _ in range(subdivision.vertex_count)
        ]
        for face_index, face in enumerate(subdivision.faces):
            for vertex_index in face:
                incidence[int(vertex_index)].append(face_index)
        degree = max((len(item) for item in incidence), default=0)
        padded = np.zeros((subdivision.vertex_count, max(1, degree)), dtype=np.int64)
        mask = np.zeros_like(padded, dtype=np.float32)
        for vertex_index, faces_here in enumerate(incidence):
            if faces_here:
                padded[vertex_index, : len(faces_here)] = faces_here
                mask[vertex_index, : len(faces_here)] = 1.0
        self.incident_faces = torch.as_tensor(padded, device=self.device)
        self.incident_mask = torch.as_tensor(mask, device=self.device)

    def place(
        self,
        supr_vertices: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """Map raw SUPR vertices into the normalized shoe frame."""

        remapped = supr_vertices @ self.remap.T
        return remapped * self.scale + translation[:, None, :]

    def _parameters(
        self,
        batch: int,
        initial_translation: np.ndarray,
        initial_ankle_degrees: np.ndarray,
        initial_midfoot_degrees: np.ndarray,
        initial_betas: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        limit = np.deg2rad(config.max_abs_pitch_degrees)

        def inverse_tanh(value: np.ndarray, bound: float) -> torch.Tensor:
            ratio = np.clip(np.asarray(value) / bound, -0.999, 0.999)
            return torch.as_tensor(
                np.arctanh(ratio), dtype=torch.float32, device=self.device
            )

        return {
            "translation": torch.as_tensor(
                np.asarray(initial_translation, dtype=np.float32),
                device=self.device,
            ).clone().requires_grad_(True),
            "ankle_raw": inverse_tanh(
                np.deg2rad(initial_ankle_degrees), limit
            ).clone().requires_grad_(True),
            "midfoot_raw": inverse_tanh(
                np.deg2rad(initial_midfoot_degrees), limit
            ).clone().requires_grad_(True),
            "betas_raw": inverse_tanh(
                np.asarray(initial_betas, dtype=np.float64), config.max_abs_beta
            ).clone().requires_grad_(True),
        }

    def _decode(self, parameters: dict[str, torch.Tensor]):
        """Soft bounds via tanh, so the old hard clamps stay differentiable."""

        config = self.config
        limit = float(np.deg2rad(config.max_abs_pitch_degrees))
        ankle = limit * torch.tanh(parameters["ankle_raw"])
        midfoot = limit * torch.tanh(parameters["midfoot_raw"])
        betas = config.max_abs_beta * torch.tanh(parameters["betas_raw"])
        # Project onto the anatomy budget rather than penalizing departures
        # from it. A hinge was tried first and the containment term simply
        # outbid it: ||beta|| moved 8.37 -> 8.25 against a cap of 5.0. The
        # budget is a constraint, not a preference, because the cage deformation stage cannot deform
        # the canonical volume onto anatomy that far from canonical without
        # inverting an element. Scaling the whole vector keeps the shape
        # direction the optimizer chose and only limits how far it goes.
        if config.max_beta_norm > 0.0:
            norm = betas.norm(dim=1, keepdim=True).clamp(min=1e-6)
            betas = betas * torch.clamp(config.max_beta_norm / norm, max=1.0)
        return ankle, midfoot, betas

    def forward(
        self, parameters: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ankle, midfoot, betas = self._decode(parameters)
        pose = self.model.pose_from_pitches(ankle, midfoot)
        output = self.model(pose, betas, None)
        vertices = self.place(output.vertices, parameters["translation"])
        joints = self.place(output.joints, parameters["translation"])
        return vertices, joints, ankle, midfoot, betas

    def containment_samples(
        self, vertices: torch.Tensor, field: ShoeCavityField
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vertices plus face centroids, with the collar opening masked out.

        The exact judge tests triangles, so centroids are included alongside
        vertices exactly as ``foot_prior.cavity`` samples both.

        The exemption is keyed on the shoe, not on the foot. ``_ankle_signed
        _exemptions`` uses an axis-aligned box - above the ankle joint and
        behind the midfoot joint - and so did this method until it was measured:
        that box covers the entire rear-upper quadrant, 12-22 % of the foot,
        including the back of the heel, and inside it the containment sign is
        dropped. 64-86 % of heel containment violations sat inside it, which is
        to say the foot could reverse straight through the heel counter at zero
        cost.

        What the exemption is actually for is the place where the leg leaves
        through the collar, and the shoe already says where that is: a footbed
        column with no surface above it. Behind the heel counter and beyond the
        toe there is no footbed, so those samples can never be exempt however
        far out of the shoe they travel. No anatomical landmark and no tuned
        threshold is involved, so the rule is the same for every shoe.
        """

        dense = torch.einsum("sv,bvc->bsc", self.subdivide, vertices)
        triangles = dense[:, self.faces]
        centroids = triangles.mean(dim=2)
        samples = torch.cat((dense, centroids), dim=1)
        with torch.no_grad():
            # Detached on purpose: areas weight the barrier, they must never
            # become something the optimizer can shrink to cheapen a violation.
            face_area = 0.5 * torch.linalg.cross(
                triangles[:, :, 1] - triangles[:, :, 0],
                triangles[:, :, 2] - triangles[:, :, 0],
                dim=-1,
            ).norm(dim=-1)
            gathered = face_area[:, self.incident_faces] * self.incident_mask
            vertex_area = gathered.sum(dim=2) / 3.0
            area = torch.cat((vertex_area, face_area), dim=1)
            area = area / area.sum(dim=1, keepdim=True).clamp(min=1e-12)
        with torch.no_grad():
            # Each sample is judged by its own column, centroids included, so
            # the mask needs no separate per-face rule.
            mask = sample_open_above(field, samples[:, :, (0, 2)])
        return samples, mask, area

    def fit(
        self,
        field: ShoeCavityField,
        initial_translation: np.ndarray,
        initial_ankle_degrees: np.ndarray,
        initial_midfoot_degrees: np.ndarray,
        initial_betas: np.ndarray,
        contact_regions: dict,
        plantar_vertex_indices: np.ndarray,
        compression_allowance: float,
        stages: tuple | None = None,
        record_every: int = 25,
        lbfgs_steps: int = 0,
    ) -> FitResult:
        import time

        config = self.config
        torch.manual_seed(config.seed)
        batch = int(np.atleast_1d(initial_ankle_degrees).shape[0])
        parameters = self._parameters(
            batch,
            np.atleast_2d(initial_translation),
            np.atleast_1d(initial_ankle_degrees),
            np.atleast_1d(initial_midfoot_degrees),
            np.atleast_2d(initial_betas),
        )
        ankle_initial = (
            config.max_abs_pitch_degrees
            * torch.tanh(parameters["ankle_raw"]).detach()
            * float(np.pi / 180.0)
        )
        midfoot_initial = (
            config.max_abs_pitch_degrees
            * torch.tanh(parameters["midfoot_raw"]).detach()
            * float(np.pi / 180.0)
        )
        contact = {
            name: torch.as_tensor(
                np.asarray(indices, dtype=np.int64), device=self.device
            )
            for name, indices in contact_regions.items()
        }
        plantar = torch.as_tensor(
            np.asarray(plantar_vertex_indices, dtype=np.int64), device=self.device
        )
        groups = {
            "translation": [parameters["translation"]],
            "pose": [parameters["ankle_raw"], parameters["midfoot_raw"]],
            "betas": [parameters["betas_raw"]],
        }

        torch.cuda.reset_peak_memory_stats(self.device)
        history: list[dict[str, Any]] = []
        started = time.time()
        schedule = stages if stages is not None else config.stages
        for stage in schedule:
            for name, tensors in groups.items():
                for tensor in tensors:
                    tensor.requires_grad_(name in stage.active)
            optimizer = torch.optim.Adam(
                [
                    {"params": groups[name], "lr": stage.learning_rates[name]}
                    for name in stage.active
                ]
            )
            for step in range(stage.steps):
                optimizer.zero_grad(set_to_none=True)
                terms = self.evaluate(
                    parameters, field, contact, plantar, compression_allowance,
                    ankle_initial, midfoot_initial,
                )
                terms["total"].backward()
                optimizer.step()
                if step % record_every == 0 or step == stage.steps - 1:
                    history.append(
                        {
                            "stage": stage.name,
                            "step": step,
                            **{
                                key: float(value.detach())
                                for key, value in terms.items()
                            },
                        }
                    )
        if lbfgs_steps > 0:
            # Optional second-order polish on everything the last stage moved.
            # L-BFGS needs a closure because it re-evaluates the objective
            # several times per step.
            active = [
                tensor
                for name in schedule[-1].active
                for tensor in groups[name]
            ]
            for tensor in active:
                tensor.requires_grad_(True)
            # A softplus barrier is flat wherever the foot is clear of every
            # surface, and a strong-Wolfe line search extrapolates across that
            # flat region hard enough to produce non-finite vertices. A small
            # fixed step keeps it in the trust region the Adam stages left it
            # in, and the snapshot below makes the polish strictly optional.
            snapshot = {
                name: tensor.detach().clone()
                for name, tensor in parameters.items()
            }
            with torch.no_grad():
                before = float(
                    self.evaluate(
                        parameters, field, contact, plantar,
                        compression_allowance, ankle_initial, midfoot_initial,
                    )["total"]
                )
            polisher = torch.optim.LBFGS(
                active, lr=0.05, max_iter=lbfgs_steps, history_size=20
            )

            def closure() -> torch.Tensor:
                polisher.zero_grad(set_to_none=True)
                terms = self.evaluate(
                    parameters, field, contact, plantar, compression_allowance,
                    ankle_initial, midfoot_initial,
                )
                total = terms["total"]
                if torch.isfinite(total):
                    total.backward()
                return total

            polisher.step(closure)
            with torch.no_grad():
                terms = self.evaluate(
                    parameters, field, contact, plantar, compression_allowance,
                    ankle_initial, midfoot_initial,
                )
                after = float(terms["total"])
                healthy = (
                    np.isfinite(after)
                    and after <= before
                    and all(
                        bool(torch.isfinite(tensor).all())
                        for tensor in parameters.values()
                    )
                )
                if not healthy:
                    # Never hand back a polish that diverged or made it worse.
                    for name, tensor in parameters.items():
                        tensor.copy_(snapshot[name])
                    terms = self.evaluate(
                        parameters, field, contact, plantar,
                        compression_allowance, ankle_initial, midfoot_initial,
                    )
            history.append(
                {
                    "stage": "E_lbfgs",
                    "step": lbfgs_steps,
                    "accepted": bool(healthy),
                    "total_before": before,
                    **{key: float(value.detach()) for key, value in terms.items()},
                }
            )
        torch.cuda.synchronize(self.device)
        seconds = time.time() - started
        peak = int(torch.cuda.max_memory_allocated(self.device))

        with torch.no_grad():
            vertices, _, ankle, midfoot, betas = self.forward(parameters)
            terms = self.evaluate(
                parameters, field, contact, plantar, compression_allowance,
                ankle_initial, midfoot_initial,
            )
            allowance = (
                length_residual(vertices, 0.0).cpu().numpy()
            )
            pose = self.model.pose_from_pitches(ankle, midfoot)
        return FitResult(
            vertices=vertices.detach().cpu().numpy().astype(np.float64),
            betas=betas.detach().cpu().numpy().astype(np.float64),
            pose=pose.detach().cpu().numpy().astype(np.float64),
            ankle_degrees=np.rad2deg(ankle.detach().cpu().numpy()),
            midfoot_degrees=np.rad2deg(midfoot.detach().cpu().numpy()),
            translation=parameters["translation"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
            scale=self.scale,
            history=history,
            final_losses={
                key: float(value.detach()) for key, value in terms.items()
            },
            toe_allowance_mm=allowance,
            seconds=seconds,
            peak_memory_bytes=peak,
        )

    def evaluate(
        self,
        parameters: dict[str, torch.Tensor],
        field: ShoeCavityField,
        contact: dict,
        plantar: torch.Tensor,
        compression_allowance: float,
        ankle_initial: torch.Tensor,
        midfoot_initial: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        vertices, _joints, ankle, midfoot, betas = self.forward(parameters)
        samples, exempt_mask, area = self.containment_samples(vertices, field)
        containment, _ = containment_loss(
            samples,
            field,
            config.containment_margin_mm,
            config.containment_softness_mm,
            config.containment_scale_mm,
            exempt_mask if config.use_ankle_exemption else None,
            area,
        )
        contact_term, penetration, _ = support_loss(
            vertices,
            field,
            contact,
            plantar,
            compression_allowance,
            config.support_contact_scale_mm,
            config.support_penetration_scale_mm,
            safety_mm=config.support_safety_mm,
        )
        shape = beta_prior(betas)
        # Hinge on the anatomy budget: free inside it, steep outside.
        norm_excess = torch.nn.functional.relu(
            betas.norm(dim=1) - config.max_beta_norm
        ).square().mean()
        posture = pose_prior(
            ankle, midfoot, ankle_initial, midfoot_initial,
            config.pose_prior_scale_degrees,
        )
        length = (
            length_residual(vertices, config.target_toe_allowance_mm)
            / config.length_scale_mm
        ).square().mean()
        heel, _ = heel_seating_loss(
            vertices, plantar,
            config.heel_behind_scale_mm, config.heel_ahead_scale_mm,
        )
        total = (
            config.w_containment * containment
            + config.w_support * contact_term
            + config.w_penetration * penetration
            + config.w_beta * shape
            + config.w_beta_norm * norm_excess
            + config.w_pose * posture
            + config.w_length * length
            + config.w_heel * heel
        )
        return {
            "beta_norm_excess": norm_excess,
            "containment": containment,
            "support": contact_term,
            "penetration": penetration,
            "beta_prior": shape,
            "pose_prior": posture,
            "length": length,
            "heel": heel,
            "total": total,
        }
