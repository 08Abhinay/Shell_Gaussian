"""Neutral SUPR foot loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import TriangleMesh


SUPR_RIGHT_FOOT_VERTEX_COUNT = 266
SUPR_RIGHT_FOOT_FACE_COUNT = 515


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
