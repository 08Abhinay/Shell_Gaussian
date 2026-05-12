"""Minimal SUPR-Foot mesh loading helpers.

The first FootShellGaussian milestone only needs a neutral anatomical foot mesh
that can be converted into an SDF. This module reads the released SUPR-style
``.npy`` model dictionary directly and applies shape coefficients in template
space. When pose-dependent deformations are needed, ``load_supr_foot_posed``
delegates to the vendored official SUPR implementation under ``baselines/SUPR``.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class FootMesh:
    """Simple mesh container using NumPy arrays."""

    vertices: np.ndarray
    faces: np.ndarray
    metadata: Dict[str, Any]


def _as_float_betas(betas: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if betas is None:
        return None

    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.ndim != 1:
        raise ValueError("betas must be a 1D array after reshaping")
    return betas


def _apply_homogeneous_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")

    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    vertices_h = np.concatenate([vertices, ones], axis=1)
    return (vertices_h @ transform.T)[:, :3]


def _default_supr_repo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "baselines" / "SUPR"


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def load_supr_foot_template(
    model_path: str,
    betas: Optional[np.ndarray] = None,
    scale: float = 1.0,
    transform: Optional[np.ndarray] = None,
) -> FootMesh:
    """Load a neutral SUPR-Foot template mesh from a released ``.npy`` model.

    Args:
        model_path: Path to the SUPR-Foot ``.npy`` model dictionary.
        betas: Optional shape coefficients. When provided, they are applied with
            ``shapedirs`` in template space.
        scale: Uniform scale applied after shape offsets.
        transform: Optional 4x4 homogeneous transform applied after scaling.

    Returns:
        A ``FootMesh`` with ``vertices`` in float32 and ``faces`` in int64.
    """

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SUPR-Foot model not found: {path}")

    model = np.load(path, allow_pickle=True, encoding="latin1")[()]
    required_keys = {"v_template", "f"}
    missing_keys = sorted(required_keys.difference(model.keys()))
    if missing_keys:
        raise KeyError(f"SUPR-Foot model is missing keys: {missing_keys}")

    vertices = np.asarray(model["v_template"], dtype=np.float32).copy()
    faces = np.asarray(model["f"], dtype=np.int64).copy()

    betas = _as_float_betas(betas)
    if betas is not None:
        if "shapedirs" not in model:
            raise KeyError("Cannot apply betas because the model has no shapedirs")

        shapedirs = np.asarray(model["shapedirs"], dtype=np.float32)
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != vertices.shape:
            raise ValueError(
                "Expected shapedirs with shape "
                f"({vertices.shape[0]}, {vertices.shape[1]}, num_betas)"
            )
        if betas.shape[0] > shapedirs.shape[2]:
            raise ValueError(
                f"Got {betas.shape[0]} betas, but model only has {shapedirs.shape[2]}"
            )

        vertices = vertices + np.tensordot(
            shapedirs[:, :, : betas.shape[0]],
            betas,
            axes=([2], [0]),
        )

    vertices = vertices * np.float32(scale)
    if transform is not None:
        vertices = _apply_homogeneous_transform(vertices, transform)

    metadata = {
        "model_path": str(path),
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_betas_applied": 0 if betas is None else int(betas.shape[0]),
    }
    return FootMesh(vertices=vertices.astype(np.float32), faces=faces, metadata=metadata)


def load_supr_foot_posed(
    model_path: str,
    betas: Optional[np.ndarray] = None,
    pose: Optional[np.ndarray] = None,
    trans: Optional[np.ndarray] = None,
    scale: float = 1.0,
    transform: Optional[np.ndarray] = None,
    num_betas: int = 10,
    supr_repo_path: Optional[str] = None,
    device: str = "cuda",
) -> FootMesh:
    """Load a posed SUPR-Foot mesh through the official SUPR PyTorch model.

    The vendored SUPR implementation currently allocates buffers with
    ``torch.cuda.FloatTensor``. This wrapper therefore requires CUDA and is best
    used when we need pose-correct geometry. For neutral SDF generation,
    ``load_supr_foot_template`` remains simpler and CPU-friendly.
    """

    import torch

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SUPR-Foot model not found: {path}")

    repo_path = Path(supr_repo_path) if supr_repo_path is not None else _default_supr_repo_path()
    if not repo_path.exists():
        raise FileNotFoundError(f"SUPR repository not found: {repo_path}")

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("The official SUPR PyTorch loader requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Use load_supr_foot_template for CPU-only neutral meshes."
        )
    if torch_device.index is not None:
        torch.cuda.set_device(torch_device)

    with _temporary_sys_path(repo_path):
        from supr.pytorch.supr import SUPR

    model = SUPR(str(path), num_betas=num_betas)
    model.eval()

    betas = np.zeros(num_betas, dtype=np.float32) if betas is None else _as_float_betas(betas)
    if betas.shape[0] != num_betas:
        raise ValueError(f"Expected {num_betas} betas, got {betas.shape[0]}")

    pose_values = np.zeros(model.num_pose, dtype=np.float32) if pose is None else np.asarray(
        pose,
        dtype=np.float32,
    ).reshape(-1)
    if pose_values.shape[0] != model.num_pose:
        raise ValueError(f"Expected pose with {model.num_pose} values, got {pose_values.shape[0]}")

    trans_values = np.zeros(3, dtype=np.float32) if trans is None else np.asarray(
        trans,
        dtype=np.float32,
    ).reshape(-1)
    if trans_values.shape[0] != 3:
        raise ValueError(f"Expected trans with 3 values, got {trans_values.shape[0]}")

    pose_tensor = torch.as_tensor(pose_values[None], dtype=torch.float32, device=torch_device)
    beta_tensor = torch.as_tensor(betas[None], dtype=torch.float32, device=torch_device)
    trans_tensor = torch.as_tensor(trans_values[None], dtype=torch.float32, device=torch_device)

    with torch.no_grad():
        vertices_tensor = model.forward(pose_tensor, beta_tensor, trans_tensor)

    vertices = vertices_tensor.detach().cpu().numpy()[0].astype(np.float32)
    faces = np.asarray(model.f, dtype=np.int64).copy()

    vertices = vertices * np.float32(scale)
    if transform is not None:
        vertices = _apply_homogeneous_transform(vertices, transform)

    metadata = {
        "model_path": str(path),
        "supr_repo_path": str(repo_path),
        "source": "official_supr_pytorch",
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_betas_applied": int(num_betas),
        "num_pose": int(model.num_pose),
        "num_joints": int(model.num_joints),
    }
    return FootMesh(vertices=vertices, faces=faces, metadata=metadata)
