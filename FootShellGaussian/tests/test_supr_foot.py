"""Tests for neutral SUPR loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foot_prior.supr_foot import (
    SUPR_ANKLE_PITCH_INDEX,
    SUPR_MIDFOOT_PITCH_INDEX,
    load_neutral_supr_foot,
    load_posable_supr_foot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"


def test_known_neutral_right_foot_shape_and_indices() -> None:
    mesh = load_neutral_supr_foot(SUPR_MODEL)
    assert mesh.vertices.shape == (266, 3)
    assert mesh.faces.shape == (515, 3)
    assert np.isfinite(mesh.vertices).all()
    assert np.issubdtype(mesh.faces.dtype, np.integer)
    assert mesh.faces.min() == 0
    assert mesh.faces.max() == 265


def test_loading_is_deterministic_and_uses_stored_neutral_template() -> None:
    first = load_neutral_supr_foot(SUPR_MODEL)
    second = load_neutral_supr_foot(SUPR_MODEL)
    stored = np.load(SUPR_MODEL, allow_pickle=True).item()
    np.testing.assert_array_equal(first.vertices, stored["v_template"])
    np.testing.assert_array_equal(first.faces, stored["f"])
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, second.faces)


def test_rejects_full_body_supr_model() -> None:
    full_body = REPOSITORY_ROOT / "baselines/SUPR/data/supr_neutral.npy"
    with pytest.raises(ValueError, match="right-foot subset"):
        load_neutral_supr_foot(full_body)


def test_rejects_missing_geometry_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, {"v_template": np.zeros((266, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="missing required fields"):
        load_neutral_supr_foot(path)


def test_cuda_posable_model_reproduces_neutral_and_batches_deterministically() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("articulated SUPR test requires CUDA")
    model = load_posable_supr_foot(SUPR_MODEL, num_betas=10)
    zero_pose = np.zeros(39, dtype=np.float32)
    zero_betas = np.zeros(10, dtype=np.float32)
    neutral_vertices, neutral_joints = model.evaluate(zero_pose, zero_betas)
    stored = np.load(SUPR_MODEL, allow_pickle=True).item()
    np.testing.assert_array_equal(neutral_vertices, stored["v_template"])
    assert neutral_joints.shape == (13, 3)

    poses = np.zeros((3, 39), dtype=np.float32)
    poses[:, SUPR_ANKLE_PITCH_INDEX] = np.deg2rad([-2.0, 0.0, 2.0])
    poses[:, SUPR_MIDFOOT_PITCH_INDEX] = np.deg2rad([1.0, 0.0, -1.0])
    first_vertices, first_joints = model.evaluate(poses, zero_betas)
    second_vertices, second_joints = model.evaluate(poses, zero_betas)
    np.testing.assert_array_equal(first_vertices, second_vertices)
    np.testing.assert_array_equal(first_joints, second_joints)
    np.testing.assert_array_equal(first_vertices[1], neutral_vertices)
    assert not np.array_equal(first_vertices[0], neutral_vertices)
