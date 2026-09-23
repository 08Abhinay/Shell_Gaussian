"""HARD GATE 1: autograd through SUPR must match central finite differences."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anatomical_coordinates.supr_torch import (
    ANKLE_PITCH_INDEX,
    MIDFOOT_PITCH_INDEX,
    TorchSuprFoot,
)

MODEL_PATH = "/storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy"


@pytest.fixture(scope="module")
def model() -> TorchSuprFoot:
    if not torch.cuda.is_available():
        pytest.skip("SUPR requires CUDA")
    return TorchSuprFoot(MODEL_PATH, num_betas=10)


def _scalar(vertices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """A deterministic, non-symmetric scalar readout of the whole mesh."""

    return (vertices * weights).sum()


# SUPR runs in float32 (its buffers are ``torch.cuda.FloatTensor``), so the
# central difference is limited by round-off, not truncation: sweeping the step
# from 1e-3 to 3e-1 makes the agreement *improve* monotonically (36.3% -> 0.14%
# relative error on betas). The steps below therefore sit at the large end,
# where the differencing signal clears float32 noise while the function is still
# locally well approximated by its first-order term.
def _central_difference(function, value: torch.Tensor, index, step: float) -> float:
    plus = value.detach().clone()
    minus = value.detach().clone()
    plus[index] += step
    minus[index] -= step
    return float((function(plus) - function(minus)) / (2.0 * step))


def _finite_difference_check(model, parameter_name, step, tolerance, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = 2
    betas0 = (
        torch.randn(batch, model.num_betas, generator=generator) * 0.5
    ).double()
    ankle0 = (torch.randn(batch, generator=generator) * 0.1).double()
    midfoot0 = (torch.randn(batch, generator=generator) * 0.1).double()
    translation0 = (torch.randn(batch, 3, generator=generator) * 0.05).double()
    weights = torch.randn(
        batch, model.num_vertices, 3, generator=generator
    ).cuda()
    # SUPR evaluates in float32. A summed readout over 1596 terms loses the
    # finite-difference signal to cancellation, so scale the scalar down.
    weights = weights / weights.numel()

    def evaluate(betas, ankle, midfoot, translation):
        pose = model.pose_from_pitches(ankle.float().cuda(), midfoot.float().cuda())
        output = model(pose, betas.float().cuda(), translation.float().cuda())
        return _scalar(output.vertices, weights)

    values = {
        "betas": betas0,
        "ankle": ankle0,
        "midfoot": midfoot0,
        "translation": translation0,
    }
    target = values[parameter_name].clone().requires_grad_(True)
    live = dict(values)
    live[parameter_name] = target
    loss = evaluate(**live)
    loss.backward()
    analytic = target.grad.detach().cpu().numpy()
    assert np.isfinite(analytic).all(), f"{parameter_name}: non-finite gradient"

    flat = analytic.reshape(-1)
    # Probe the largest-magnitude entries plus a couple of arbitrary ones.
    probes = sorted(
        set(int(value) for value in np.argsort(-np.abs(flat))[:3])
        | {0, flat.size // 2}
    )
    numeric = np.zeros(len(probes))
    for slot, position in enumerate(probes):
        index = np.unravel_index(int(position), analytic.shape)

        def scalar_of(candidate, index=index):
            trial = dict(values)
            trial[parameter_name] = candidate
            with torch.no_grad():
                return float(evaluate(**trial).double())

        numeric[slot] = _central_difference(
            scalar_of, values[parameter_name], index, step
        )
    reference = flat[probes]
    scale = float(np.max(np.abs(flat)))
    assert scale > 0.0, f"{parameter_name}: gradient is identically zero"
    error = float(np.max(np.abs(numeric - reference)) / scale)
    assert error < tolerance, (
        f"{parameter_name}: relative gradient error {error:.3e} "
        f"analytic={reference} numeric={numeric}"
    )
    return error, float(np.max(np.abs(flat)))


def test_translation_gradient(model):
    error, magnitude = _finite_difference_check(model, "translation", 1e-2, 1e-3)
    assert magnitude > 0.0


def test_ankle_pitch_gradient(model):
    error, magnitude = _finite_difference_check(model, "ankle", 5e-2, 1e-2)
    assert magnitude > 0.0, "ankle pitch gradient is unexpectedly zero"


def test_midfoot_pitch_gradient(model):
    error, magnitude = _finite_difference_check(model, "midfoot", 5e-2, 1e-2)
    assert magnitude > 0.0, "midfoot pitch gradient is unexpectedly zero"


def test_beta_gradient(model):
    error, magnitude = _finite_difference_check(model, "betas", 1e-1, 1e-2)
    assert magnitude > 0.0


def test_pose_scatter_only_touches_two_indices(model):
    ankle = torch.full((3,), 0.25).cuda()
    midfoot = torch.full((3,), -0.125).cuda()
    pose = model.pose_from_pitches(ankle, midfoot)
    assert pose.shape == (3, model.num_pose_parameters)
    mask = torch.ones(model.num_pose_parameters, dtype=torch.bool)
    mask[[ANKLE_PITCH_INDEX, MIDFOOT_PITCH_INDEX]] = False
    assert torch.all(pose[:, mask] == 0.0)
    assert torch.allclose(pose[:, ANKLE_PITCH_INDEX], ankle)
    assert torch.allclose(pose[:, MIDFOOT_PITCH_INDEX], midfoot)


def test_batch_matches_single(model):
    generator = torch.Generator(device="cpu").manual_seed(7)
    betas = torch.randn(4, model.num_betas, generator=generator).cuda() * 0.4
    ankle = torch.randn(4, generator=generator).cuda() * 0.15
    midfoot = torch.randn(4, generator=generator).cuda() * 0.15
    translation = torch.randn(4, 3, generator=generator).cuda() * 0.05
    pose = model.pose_from_pitches(ankle, midfoot)
    batched = model(pose, betas, translation).vertices
    for item in range(4):
        single = model(
            pose[item : item + 1], betas[item : item + 1], translation[item : item + 1]
        ).vertices
        assert torch.allclose(batched[item], single[0], atol=1e-5)


def test_gradients_are_finite_at_exactly_zero_pose(model):
    """quat_feat divides by ||theta + 1e-8||; check the rest pose is safe."""

    betas = torch.zeros(1, model.num_betas, device="cuda", requires_grad=True)
    ankle = torch.zeros(1, device="cuda", requires_grad=True)
    midfoot = torch.zeros(1, device="cuda", requires_grad=True)
    translation = torch.zeros(1, 3, device="cuda", requires_grad=True)
    pose = model.pose_from_pitches(ankle, midfoot)
    output = model(pose, betas, translation)
    output.vertices.square().mean().backward()
    for name, tensor in (
        ("betas", betas),
        ("ankle", ankle),
        ("midfoot", midfoot),
        ("translation", translation),
    ):
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"
