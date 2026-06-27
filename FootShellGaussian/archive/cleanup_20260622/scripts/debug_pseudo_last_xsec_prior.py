#!/usr/bin/env python3
"""Visualize pseudo-last cross-section labels used by the training prior."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from foot_prior.pseudo_last_loss import PseudoLastPriorConfig, PseudoLastPriorLoss


def _slice_points(
    prior: PseudoLastPriorLoss,
    x_value: torch.Tensor,
    y_samples: int,
    z_samples: int,
) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    cfg = prior.config
    device = prior.section_x.device
    x = x_value.reshape(1)
    center = prior._interp_1d(x, prior.center_z)[0]
    support_half = prior._interp_1d(x, prior.support_half_width)[0].clamp(min=1e-6)
    height = prior._interp_1d(x, prior.section_height)[0].clamp(min=1e-6)

    lateral = torch.linspace(
        -1.0 - float(cfg.xsec_support_lateral_pad),
        1.0 + float(cfg.xsec_support_lateral_pad),
        z_samples,
        device=device,
    )
    vertical = torch.linspace(-1.0, float(cfg.xsec_ignore_high_h_ratio), y_samples, device=device)
    z = center + lateral * support_half

    xz_points = torch.stack(
        [
            x.expand(z_samples),
            torch.zeros(z_samples, dtype=x.dtype, device=device),
            z,
        ],
        dim=-1,
    )
    half_width = prior._interp_1d(xz_points[:, 0], prior.half_width).clamp(min=1e-6)
    bottom_y = prior._interp_bottom_y(
        xz_points,
        center=center.expand(z_samples),
        half_width=half_width,
    )
    h_positive = vertical.clamp(min=0.0)[:, None] * height
    h_negative = vertical.clamp(max=0.0)[:, None] * float(cfg.xsec_sole_depth)
    y = bottom_y[None, :] - (h_positive + h_negative)
    points = torch.stack(
        [
            x.expand(y_samples, z_samples),
            y,
            z[None, :].expand(y_samples, z_samples),
        ],
        dim=-1,
    ).reshape(-1, 3)
    extent = (
        float(z.min().detach().cpu()),
        float(z.max().detach().cpu()),
        float(y.max().detach().cpu()),
        float(y.min().detach().cpu()),
    )
    return points, extent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-last-sdf", required=True)
    parser.add_argument("--pseudo-last-sections", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--y-samples", type=int, default=160)
    parser.add_argument("--z-samples", type=int, default=160)
    args = parser.parse_args()

    prior = PseudoLastPriorLoss(
        PseudoLastPriorConfig(
            sdf_path=args.pseudo_last_sdf,
            sections_path=args.pseudo_last_sections,
            xsec_x_slices=max(args.num_slices, 2),
            xsec_y_samples=32,
            xsec_z_samples=32,
        ),
        device=torch.device("cpu"),
    )

    x_values = torch.linspace(prior.section_x[0], prior.section_x[-1], args.num_slices)
    cmap = ListedColormap(["#d6d6d6", "#111111", "#3f91ff", "#ffb000"])
    legend_handles = [
        Patch(color="#d6d6d6", label="ignore"),
        Patch(color="#111111", label="solid material"),
        Patch(color="#3f91ff", label="empty cavity"),
        Patch(color="#ffb000", label="last surface band"),
    ]
    fig, axes = plt.subplots(1, args.num_slices, figsize=(3.2 * args.num_slices, 3.4), squeeze=False)
    for ax, x_value in zip(axes[0], x_values):
        points, extent = _slice_points(prior, x_value, args.y_samples, args.z_samples)
        with torch.no_grad():
            last_cavity = prior._query_last_cavity_field_chunked(points)
            labels = prior._classify_points(points, last_cavity)
            image = torch.zeros(points.shape[0], dtype=torch.float32)
            image[labels["material_mask"].cpu()] = 1.0
            image[labels["empty_mask"].cpu()] = 2.0
            surface = last_cavity.abs() <= float(prior.config.xsec_surface_band)
            image[surface.cpu()] = 3.0
        image_np = image.reshape(args.y_samples, args.z_samples).numpy()
        ax.imshow(image_np, origin="upper", extent=extent, aspect="auto", vmin=0, vmax=3, cmap=cmap)
        ax.set_title(f"x={float(x_value):.3f}")
        ax.set_xlabel("z")
        ax.set_ylabel("y")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
