"""Build one coordinate map, and the geometry helpers the others share.

The comparison is deliberately like for like:

  boundary fidelity  the cage deformation stage reports how far the computational boundary had to be
                     corrected away from the fitted anatomy. Here it is the
                     residual between the flowed canonical surface and the
                     fitted surface, on the same 6,951 corresponding vertices.
  injectivity        the cage deformation stage requires every tetrahedron to keep a positive
                     determinant. Here it is the determinant of the flow map
                     itself, sampled densely through the volume.
  invertibility      the older map lookup measures a round trip through the forward and inverse
                     maps and reports ~1e-15. Here the inverse is the same field
                     integrated backwards, so the round trip measures the
                     integrator rather than a lookup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from foot_prior.mesh import load_triangle_mesh

from .field import (
    FlowFitReport,
    VelocityField,
    integrate,
    jacobian_determinants,
    round_trip_error,
    smoothness,
)


PIPELINE_ROOT = Path("/home/ab5298/Outputs/FootShellGaussian/pipeline27")
EXTENDED_VERTEX_COUNT = 6951


def load_pair(root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical and fitted surfaces; they share vertex ids and faces."""

    reference = load_triangle_mesh(
        root / "extended_anatomical_surface" / "reference" / "neutral_foot_lower_leg.ply"
    )
    instance = load_triangle_mesh(
        root / "extended_anatomical_surface" / name / "foot_lower_leg.ply"
    )
    if reference.vertices.shape != (EXTENDED_VERTEX_COUNT, 3):
        raise ValueError("canonical extended surface is not the expected topology")
    if instance.vertices.shape != reference.vertices.shape:
        raise ValueError(f"{name}: instance surface topology differs")
    return reference.vertices, instance.vertices, reference.faces


def domain_box(canonical: np.ndarray, instance: np.ndarray, margin: float = 0.25):
    """A box holding both surfaces with room for the surrounding volume."""

    both = np.concatenate((canonical, instance), axis=0)
    lower = both.min(axis=0) - margin
    upper = both.max(axis=0) + margin
    return lower, upper


def fit_one(
    canonical: np.ndarray,
    instance: np.ndarray,
    device: torch.device,
    iterations: int = 1500,
    steps: int = 16,
    weight_smooth: float = 2.0e-3,
    learning_rate: float = 3.0e-3,
    seed: int = 0,
) -> tuple[VelocityField, FlowFitReport]:
    torch.manual_seed(seed)
    lower, upper = domain_box(canonical, instance)
    field = VelocityField(
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
        seed=seed,
    ).to(device)

    source = torch.as_tensor(canonical, dtype=torch.float32, device=device)
    target = torch.as_tensor(instance, dtype=torch.float32, device=device)
    box_lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
    box_span = torch.as_tensor(upper - lower, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(field.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    started = time.time()
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        flowed = integrate(field, source, steps, 1.0)
        data = (flowed - target).square().sum(dim=-1).mean()
        # Smoothness is sampled through the whole box, not just on the surface,
        # because the map has to stay well behaved in the surrounding volume
        # where the footwear material lives.
        probe = box_lower + box_span * torch.rand(
            (2048, 3), device=device
        )
        regular = smoothness(field, probe)
        (data + weight_smooth * regular).backward()
        optimizer.step()
        schedule.step()

    with torch.no_grad():
        flowed = integrate(field, source, steps, 1.0)
        residual = (flowed - target).norm(dim=-1)
    trip = round_trip_error(field, source, steps).detach()
    probe = box_lower + box_span * torch.rand((4096, 3), device=device)
    determinants = jacobian_determinants(field, probe, steps).detach()
    report = FlowFitReport(
        vertex_rms=float(residual.square().mean().sqrt()),
        vertex_max=float(residual.max()),
        round_trip_max=float(trip.max()),
        jacobian_minimum=float(determinants.min()),
        jacobian_median=float(determinants.median()),
        smoothness=float(smoothness(field, probe).detach()),
        steps=steps,
        iterations=iterations,
    )
    report_seconds = time.time() - started
    setattr(report, "seconds", report_seconds)
    return field, report
