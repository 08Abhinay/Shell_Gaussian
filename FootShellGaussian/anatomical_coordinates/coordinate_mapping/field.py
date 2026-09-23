"""Diffeomorphic anatomical coordinates by a stationary velocity field.

The existing the cage deformation stage carries the canonical tetrahedral volume onto each
fitted anatomy and calls the result bijective when no tetrahedron inverts. That
ties injectivity to a *fixed mesh connectivity*, and it is why the deformation
stalls with ``minimum_step_after_jacobian_determinant`` on the harder shoes: the
target is reachable as a map, just not as a piecewise-affine map over those
particular tetrahedra. The anatomy budget ladder we added is a workaround for
that representational limit, not a fact about feet.

This module takes the other standard route. Define the map as the flow of a
velocity field:

    chi(x) = phi_1(x),   d/dt phi_t(x) = v(phi_t(x)),   phi_0 = identity

The flow of a Lipschitz field is a diffeomorphism for *any* integration time, so
the map is injective by construction and has an exact inverse - integrate the
same field backwards. There is no mesh to invert. This is the stationary
velocity field formulation used throughout diffeomorphic registration (LDDMM,
scaling-and-squaring, and the neural-ODE shape-correspondence work that maps
instances onto a template).

Two properties matter for this project beyond injectivity:

* ``chi`` is smooth and differentiable *with respect to the foot parameters*,
  which a barycentric lookup through a tetrahedral index is not. Section 1.2 of
  the representation document conditions its clearance prior on F_i, so the
  coordinate map has to carry gradients for that to be learnable end to end.
* the bi-Lipschitz bound the design document asks for is controlled directly by
  the field's spatial gradient, so it can be regularized rather than discovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fixed random Fourier features, so the field can represent detail.

    A plain coordinate MLP is heavily biased toward low frequencies and cannot
    reproduce a 30 mm displacement that varies over the toes without also
    smearing it across the shank.
    """

    def __init__(self, count: int = 64, scale: float = 4.0, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        basis = torch.randn((3, count), generator=generator) * scale
        self.register_buffer("basis", basis)

    @property
    def out_features(self) -> int:
        return 3 + 2 * self.basis.shape[1]

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * np.pi * points @ self.basis
        return torch.cat(
            (points, torch.sin(projected), torch.cos(projected)), dim=-1
        )


class VelocityField(nn.Module):
    """A stationary velocity field on the normalized canonical domain."""

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        width: int = 128,
        depth: int = 3,
        fourier: int = 64,
        fourier_scale: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.register_buffer("lower", lower)
        self.register_buffer("span", (upper - lower).clamp(min=1e-6))
        self.encode = FourierFeatures(fourier, fourier_scale, seed)
        layers: list[nn.Module] = [nn.Linear(self.encode.out_features, width), nn.Softplus(beta=8.0)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Softplus(beta=8.0)]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)
        # Start at the identity flow: a zero field means chi = id, which is the
        # correct place to begin for an anatomy already close to canonical.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        return 2.0 * (points - self.lower) / self.span - 1.0

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(self.normalize(points)))


def integrate(
    field: VelocityField,
    points: torch.Tensor,
    steps: int = 16,
    direction: float = 1.0,
) -> torch.Tensor:
    """Classical RK4 on the stationary field.

    ``direction = -1`` integrates the same field backwards, which is the exact
    inverse map up to integration error. That error is measured rather than
    assumed; see ``round_trip_error``.
    """

    dt = direction / float(steps)
    current = points
    for _ in range(steps):
        k1 = field(current)
        k2 = field(current + 0.5 * dt * k1)
        k3 = field(current + 0.5 * dt * k2)
        k4 = field(current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def round_trip_error(
    field: VelocityField, points: torch.Tensor, steps: int = 16
) -> torch.Tensor:
    forward = integrate(field, points, steps, 1.0)
    back = integrate(field, forward, steps, -1.0)
    return (back - points).norm(dim=-1)


def flow_jacobian(
    field: VelocityField, points: torch.Tensor, steps: int = 16, direction: float = 1.0
) -> torch.Tensor:
    """Jacobian of the whole flow map at each point, shape (N, 3, 3).

    This is the quantity the cage deformation stage checks per tetrahedron. Here it is a property of
    the map itself rather than of a discretization, so a negative determinant
    would mean the integration step is too coarse for the field's Lipschitz
    constant - not that the anatomy is unreachable.
    """

    def single(point: torch.Tensor) -> torch.Tensor:
        return integrate(field, point[None], steps, direction)[0]

    return torch.vmap(torch.func.jacrev(single))(points)


def jacobian_determinants(
    field: VelocityField, points: torch.Tensor, steps: int = 16, chunk: int = 512
) -> torch.Tensor:
    out = []
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        out.append(torch.linalg.det(flow_jacobian(field, block, steps)))
    return torch.cat(out)


def smoothness(field: VelocityField, points: torch.Tensor) -> torch.Tensor:
    """Mean squared spatial gradient of the field.

    The flow is guaranteed injective when the per-step map is a contraction of
    the identity, which this controls directly. It is also the bi-Lipschitz
    handle the representation document asks for: bounding ``||grad v||`` bounds
    the distortion constants c and C.
    """

    points = points.detach().requires_grad_(True)
    velocity = field(points)
    total = points.new_zeros(())
    for axis in range(3):
        grad = torch.autograd.grad(
            velocity[:, axis].sum(), points, create_graph=True
        )[0]
        total = total + grad.square().sum(dim=-1).mean()
    return total / 3.0


@dataclass
class FlowFitReport:
    vertex_rms: float
    vertex_max: float
    round_trip_max: float
    jacobian_minimum: float
    jacobian_median: float
    smoothness: float
    steps: int
    iterations: int
