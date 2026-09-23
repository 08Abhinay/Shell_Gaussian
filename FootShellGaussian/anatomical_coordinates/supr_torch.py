"""Differentiable, batched Torch interface to the official SUPR right foot.

The existing ``foot_prior.supr_foot.SuprFootModel`` deliberately severs the
autograd graph (``torch.no_grad`` plus ``.cpu().numpy()``). That wrapper is left
untouched; this module is a parallel entry point that keeps the graph alive.

The official model is already a ``torch.nn.Module`` whose forward pass is pure
Torch, but it registers its buffers with ``torch.cuda.FloatTensor``, so CUDA is
mandatory and the module is always float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from foot_prior.supr_foot import (
    SUPR_ANKLE_PITCH_INDEX,
    SUPR_MIDFOOT_PITCH_INDEX,
    SUPR_RIGHT_FOOT_FACE_COUNT,
    SUPR_RIGHT_FOOT_JOINT_COUNT,
    SUPR_RIGHT_FOOT_POSE_PARAMETER_COUNT,
    SUPR_RIGHT_FOOT_VERTEX_COUNT,
)


# The foot subset carries no ``axis_meta``, so SUPR runs its unconstrained
# kinematic tree: 13 joints x 3 axis-angle values. Pose index 3 is joint 1
# (ankle) rotating about SUPR X, and index 6 is joint 2 (midfoot) about the
# same axis. SUPR X is foot width, so both are pitch.
ANKLE_PITCH_INDEX = SUPR_ANKLE_PITCH_INDEX
MIDFOOT_PITCH_INDEX = SUPR_MIDFOOT_PITCH_INDEX


@dataclass(frozen=True)
class SuprFootOutput:
    """Vertices and posed joints that both retain their autograd history."""

    vertices: torch.Tensor  # (B, V, 3)
    joints: torch.Tensor  # (B, J, 3)


class TorchSuprFoot(torch.nn.Module):
    """Batched SUPR evaluation that preserves gradients.

    Gradients flow to ``betas``, to every pose entry, and to ``translation``.
    """

    def __init__(self, model_path: str | Path, num_betas: int = 10) -> None:
        super().__init__()
        source = Path(model_path).expanduser().resolve(strict=True)
        if source.suffix.lower() != ".npy":
            raise ValueError("SUPR model must be a .npy file")
        if int(num_betas) != num_betas or not 1 <= int(num_betas) <= 300:
            raise ValueError("num_betas must be an integer in [1, 300]")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the official SUPR implementation registers CUDA buffers, so a "
                "GPU is required"
            )
        from supr.pytorch.supr import SUPR

        model = SUPR(str(source), num_betas=int(num_betas))
        if (
            model.num_verts != SUPR_RIGHT_FOOT_VERTEX_COUNT
            or len(model.f) != SUPR_RIGHT_FOOT_FACE_COUNT
            or model.num_joints != SUPR_RIGHT_FOOT_JOINT_COUNT
            or model.num_pose != SUPR_RIGHT_FOOT_POSE_PARAMETER_COUNT
        ):
            raise ValueError(
                "expected the unconstrained 266-vertex, 13-joint SUPR right foot"
            )
        self.supr = model
        self.num_betas = int(num_betas)
        self.num_pose_parameters = int(model.num_pose)
        self.num_vertices = int(model.num_verts)
        self.register_buffer(
            "faces",
            torch.as_tensor(np.asarray(model.f, dtype=np.int64)),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self.supr.v_template.device

    def forward(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        translation: torch.Tensor | None = None,
    ) -> SuprFootOutput:
        """Evaluate a batch of poses and shapes, keeping the graph intact."""

        if pose.ndim != 2 or pose.shape[1] != self.num_pose_parameters:
            raise ValueError(
                f"pose must have shape (B, {self.num_pose_parameters})"
            )
        if betas.ndim != 2 or betas.shape[1] != self.num_betas:
            raise ValueError(f"betas must have shape (B, {self.num_betas})")
        if pose.shape[0] != betas.shape[0]:
            raise ValueError("pose and betas must share a batch size")
        device = self.device
        pose = pose.to(device=device, dtype=torch.float32)
        betas = betas.to(device=device, dtype=torch.float32)
        if translation is None:
            translation = torch.zeros(
                (pose.shape[0], 3), dtype=torch.float32, device=device
            )
        if translation.ndim != 2 or translation.shape != (pose.shape[0], 3):
            raise ValueError("translation must have shape (B, 3)")
        translation = translation.to(device=device, dtype=torch.float32)

        vertices = self.supr(pose, betas, translation)
        # SUPR stashes the posed joints on the returned tensor. Read it before
        # any further operation, because tensor attributes do not survive ops.
        joints = vertices.J_transformed
        return SuprFootOutput(vertices=torch.as_tensor(vertices), joints=joints)

    def pose_from_pitches(
        self,
        ankle_pitch: torch.Tensor,
        midfoot_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter two pitch angles (radians) into an otherwise neutral pose.

        Built with ``index_put`` on a zero tensor so the two active entries stay
        differentiable while every other pose value is a constant zero.
        """

        if ankle_pitch.ndim != 1 or midfoot_pitch.ndim != 1:
            raise ValueError("pitch tensors must have shape (B,)")
        if ankle_pitch.shape != midfoot_pitch.shape:
            raise ValueError("pitch tensors must share a batch size")
        batch = ankle_pitch.shape[0]
        columns = torch.stack((ankle_pitch, midfoot_pitch), dim=1)
        pose = torch.zeros(
            (batch, self.num_pose_parameters),
            dtype=columns.dtype,
            device=columns.device,
        )
        index = torch.as_tensor(
            [ANKLE_PITCH_INDEX, MIDFOOT_PITCH_INDEX],
            dtype=torch.long,
            device=columns.device,
        ).expand(batch, 2)
        return pose.scatter(1, index, columns)

    @torch.no_grad()
    def evaluate_numpy(
        self, pose: np.ndarray, betas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience NumPy path for visualization and cross-checks."""

        pose_array = np.atleast_2d(np.asarray(pose, dtype=np.float32))
        beta_array = np.atleast_2d(np.asarray(betas, dtype=np.float32))
        output = self.forward(
            torch.as_tensor(pose_array), torch.as_tensor(beta_array)
        )
        return (
            output.vertices.detach().cpu().numpy().astype(np.float64),
            output.joints.detach().cpu().numpy().astype(np.float64),
        )
