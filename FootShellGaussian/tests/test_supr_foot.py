"""Tests for neutral SUPR loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foot_prior.supr_foot import load_neutral_supr_foot


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
