"""Fit many shoes' velocity fields in one set of CUDA kernels.

Profiling the per-shoe fitter showed the work is launch-bound, not
FLOP-bound: a full training iteration costs 68 ms, of which 60 ms is the RK4
chain - 64 evaluations of a 128-wide MLP on 6,951 points. Each of those kernels
is far too small to occupy the device, so the GPU sat at 38 % and adding more
GPUs would only have produced more idle ones.

The fix is to give the device wider work rather than more devices. Every shoe
gets its own velocity field, but the fields are stored as stacked weights and
evaluated with ``baddbmm``, so S shoes cost one kernel instead of S. The maths
per shoe is identical to ``flow.VelocityField``; only the batching differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class BatchedVelocityFields(nn.Module):
    """S independent stationary velocity fields, evaluated together.

    Shapes throughout: points are ``(S, N, 3)``, weights ``(S, in, out)``.
    """

    def __init__(
        self,
        lower: torch.Tensor,     # (S, 3)
        upper: torch.Tensor,     # (S, 3)
        width: int = 128,
        depth: int = 3,
        fourier: int = 64,
        fourier_scale: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        shoes = lower.shape[0]
        self.shoes = shoes
        self.register_buffer("lower", lower[:, None, :])
        self.register_buffer("span", (upper - lower).clamp(min=1e-6)[:, None, :])
        generator = torch.Generator().manual_seed(seed)
        # One shared Fourier basis: the domains are already normalized, so the
        # frequencies mean the same thing for every shoe.
        self.register_buffer(
            "basis", torch.randn((3, fourier), generator=generator) * fourier_scale
        )
        sizes = [3 + 2 * fourier] + [width] * (depth) + [3]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for index in range(len(sizes) - 1):
            fan_in, fan_out = sizes[index], sizes[index + 1]
            if index == len(sizes) - 2:
                weight = torch.zeros((shoes, fan_in, fan_out))
                bias = torch.zeros((shoes, 1, fan_out))
            else:
                bound = float(np.sqrt(1.0 / fan_in))
                weight = (
                    torch.rand((shoes, fan_in, fan_out), generator=generator) * 2 - 1
                ) * bound
                bias = (
                    torch.rand((shoes, 1, fan_out), generator=generator) * 2 - 1
                ) * bound
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))
        self.activation = nn.Softplus(beta=8.0)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        unit = 2.0 * (points - self.lower) / self.span - 1.0
        projected = 2.0 * np.pi * (unit @ self.basis)
        hidden = torch.cat((unit, projected.sin(), projected.cos()), dim=-1)
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.baddbmm(bias, hidden, weight)
            if index != last:
                hidden = self.activation(hidden)
        return hidden


def integrate_batched(
    fields: BatchedVelocityFields,
    points: torch.Tensor,
    steps: int = 8,
    direction: float = 1.0,
) -> torch.Tensor:
    dt = direction / float(steps)
    current = points
    for _ in range(steps):
        k1 = fields(current)
        k2 = fields(current + 0.5 * dt * k1)
        k3 = fields(current + 0.5 * dt * k2)
        k4 = fields(current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def smoothness_batched(
    fields: BatchedVelocityFields, points: torch.Tensor, probes: int = 1
) -> torch.Tensor:
    """Hutchinson estimate of the mean squared spatial gradient.

    The exact version needs one backward pass per output axis. A random probe
    gives an unbiased estimate of the same Frobenius norm in a single pass,
    which matters here because the term is evaluated every iteration.
    """

    points = points.detach().requires_grad_(True)
    velocity = fields(points)
    total = points.new_zeros(())
    for _ in range(probes):
        noise = torch.randn_like(velocity)
        grad = torch.autograd.grad(
            (velocity * noise).sum(), points, create_graph=True
        )[0]
        total = total + grad.square().sum(dim=-1).mean()
    return total / probes


@dataclass
class BatchedResult:
    names: list[str]
    vertex_rms_mm: list[float]
    vertex_max_mm: list[float]
    seconds: float


def fit_batch(
    names: list[str],
    canonical: np.ndarray,          # (N, 3) shared canonical surface
    instances: np.ndarray,          # (S, N, 3)
    device: torch.device,
    iterations: int = 3000,
    steps: int = 8,
    weight_smooth: float = 1.0e-3,
    learning_rate: float = 3.0e-3,
    margin: float = 0.25,
    seed: int = 0,
) -> tuple[BatchedVelocityFields, BatchedResult]:
    import time

    torch.manual_seed(seed)
    shoes = instances.shape[0]
    lower = np.stack([
        np.minimum(canonical.min(0), instances[i].min(0)) - margin
        for i in range(shoes)
    ])
    upper = np.stack([
        np.maximum(canonical.max(0), instances[i].max(0)) + margin
        for i in range(shoes)
    ])
    fields = BatchedVelocityFields(
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
        seed=seed,
    ).to(device)

    source = torch.as_tensor(canonical, dtype=torch.float32, device=device)
    source = source[None].expand(shoes, -1, -1).contiguous()
    target = torch.as_tensor(instances, dtype=torch.float32, device=device)
    box_lower = torch.as_tensor(lower, dtype=torch.float32, device=device)[:, None, :]
    box_span = torch.as_tensor(upper - lower, dtype=torch.float32, device=device)[:, None, :]

    optimizer = torch.optim.Adam(fields.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    started = time.time()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        flowed = integrate_batched(fields, source, steps, 1.0)
        data = (flowed - target).square().sum(dim=-1).mean()
        probe = box_lower + box_span * torch.rand(
            (shoes, 1024, 3), device=device
        )
        (data + weight_smooth * smoothness_batched(fields, probe)).backward()
        optimizer.step()
        schedule.step()
    torch.cuda.synchronize(device)
    seconds = time.time() - started

    with torch.no_grad():
        flowed = integrate_batched(fields, source, steps, 1.0)
        residual = (flowed - target).norm(dim=-1)
    return fields, BatchedResult(
        names=list(names),
        vertex_rms_mm=[float(r.square().mean().sqrt()) * 262.5 for r in residual],
        vertex_max_mm=[float(r.max()) * 262.5 for r in residual],
        seconds=seconds,
    )
