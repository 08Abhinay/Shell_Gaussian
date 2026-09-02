"""Neutral and articulated SUPR right-foot loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import TriangleMesh


SUPR_RIGHT_FOOT_VERTEX_COUNT = 266
SUPR_RIGHT_FOOT_FACE_COUNT = 515
SUPR_RIGHT_FOOT_JOINT_COUNT = 13
SUPR_RIGHT_FOOT_POSE_PARAMETER_COUNT = 39
SUPR_ANKLE_PITCH_INDEX = 3
SUPR_MIDFOOT_PITCH_INDEX = 6


class SuprFootModel:
    """Small NumPy-facing wrapper around the official CUDA SUPR model."""

    def __init__(self, model: Any, torch_module: Any, num_betas: int) -> None:
        self._model = model
        self._torch = torch_module
        self.num_betas = int(num_betas)
        self.num_pose_parameters = int(model.num_pose)
        self.faces = np.asarray(model.f, dtype=np.int64).copy()

    def evaluate(
        self,
        pose_parameters: np.ndarray,
        betas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one or a batch of poses, returning vertices and joints.

        One-dimensional inputs return arrays shaped ``(V, 3)`` and ``(J, 3)``.
        Batched inputs return ``(B, V, 3)`` and ``(B, J, 3)``.
        """

        pose = np.asarray(pose_parameters, dtype=np.float32)
        shape = np.asarray(betas, dtype=np.float32)
        single = pose.ndim == 1 and shape.ndim == 1
        if pose.ndim == 1:
            pose = pose[None, :]
        if shape.ndim == 1:
            shape = shape[None, :]
        if pose.ndim != 2 or pose.shape[1] != self.num_pose_parameters:
            raise ValueError(
                "pose_parameters must have shape (39,) or (B, 39)"
            )
        if shape.ndim != 2 or shape.shape[1] != self.num_betas:
            raise ValueError(
                f"betas must have shape ({self.num_betas},) or "
                f"(B, {self.num_betas})"
            )
        if not np.isfinite(pose).all() or not np.isfinite(shape).all():
            raise ValueError("SUPR pose parameters and betas must be finite")
        if len(pose) != len(shape):
            if len(pose) == 1:
                pose = np.repeat(pose, len(shape), axis=0)
                single = False
            elif len(shape) == 1:
                shape = np.repeat(shape, len(pose), axis=0)
                single = False
            else:
                raise ValueError("SUPR pose and beta batch sizes must agree")

        device = self._torch.device("cuda", self._torch.cuda.current_device())
        pose_tensor = self._torch.as_tensor(pose, device=device)
        beta_tensor = self._torch.as_tensor(shape, device=device)
        translation = self._torch.zeros(
            (len(pose), 3), dtype=self._torch.float32, device=device
        )
        with self._torch.no_grad():
            output = self._model(pose_tensor, beta_tensor, translation)
        vertices = output.detach().cpu().numpy().astype(np.float64, copy=False)
        joints = (
            output.J_transformed.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        if single:
            return vertices[0], joints[0]
        return vertices, joints


def load_neutral_supr_foot(model_path: str | Path) -> TriangleMesh:
    """Load the stored neutral right-foot template and faces from SUPR data."""

    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".npy":
        raise ValueError("SUPR model must be a .npy file")

    container = np.load(source, allow_pickle=True)
    if not isinstance(container, np.ndarray) or container.shape != ():
        raise ValueError("SUPR model must contain one scalar dictionary")
    payload: Any = container.item()
    if not isinstance(payload, Mapping):
        raise ValueError("SUPR model payload must be a mapping")
    missing = sorted({"v_template", "f"} - payload.keys())
    if missing:
        raise ValueError(f"SUPR model is missing required fields: {missing}")

    vertices = np.asarray(payload["v_template"])
    faces = np.asarray(payload["f"])
    expected_vertices = (SUPR_RIGHT_FOOT_VERTEX_COUNT, 3)
    expected_faces = (SUPR_RIGHT_FOOT_FACE_COUNT, 3)
    if vertices.shape != expected_vertices or faces.shape != expected_faces:
        raise ValueError(
            "expected the neutral SUPR right-foot subset with shapes "
            f"{expected_vertices} and {expected_faces}; received "
            f"{vertices.shape} and {faces.shape}"
        )
    return TriangleMesh(vertices, faces)


def load_posable_supr_foot(
    model_path: str | Path,
    num_betas: int = 10,
) -> SuprFootModel:
    """Load the official CUDA SUPR implementation for articulated evaluation."""

    source = Path(model_path).expanduser().resolve(strict=True)
    if source.suffix.lower() != ".npy":
        raise ValueError("SUPR model must be a .npy file")
    if int(num_betas) != num_betas or not 1 <= int(num_betas) <= 300:
        raise ValueError("num_betas must be an integer in [1, 300]")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "articulated SUPR fitting requires PyTorch; install the fitting extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "articulated SUPR fitting requires CUDA because the official SUPR "
            "implementation constructs CUDA buffers"
        )
    try:
        from supr.pytorch.supr import SUPR
    except ImportError as error:
        raise RuntimeError(
            "the official SUPR package is unavailable; install ../baselines/SUPR "
            "in editable mode"
        ) from error

    model = SUPR(str(source), num_betas=int(num_betas)).cuda().eval()
    if (
        model.num_verts != SUPR_RIGHT_FOOT_VERTEX_COUNT
        or len(model.f) != SUPR_RIGHT_FOOT_FACE_COUNT
        or model.num_joints != SUPR_RIGHT_FOOT_JOINT_COUNT
        or model.num_pose != SUPR_RIGHT_FOOT_POSE_PARAMETER_COUNT
    ):
        raise ValueError(
            "expected the unconstrained 266-vertex, 13-joint SUPR right-foot model"
        )
    return SuprFootModel(model, torch, int(num_betas))
