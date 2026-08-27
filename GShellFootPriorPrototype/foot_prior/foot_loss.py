"""Training losses that turn the neutral SUPR foot prior into constraints.

The SDF grid is stored in raw SUPR foot coordinates. During training, GShell
geometry lives in shoe coordinates, so this module first maps shoe-space points
back into foot space with the saved alignment transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .foot_alignment import FootAlignment, axis_sign
from .foot_sdf import FootSDFConfig, FootSDFGrid


def _resolve_path(path: str) -> Path:
    """Resolve config paths from either cwd or the FootShellGaussian root."""

    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    footshell_root = Path(__file__).resolve().parents[1]
    return footshell_root / candidate


@dataclass(frozen=True)
class FootPriorLossConfig:
    """Runtime settings for neutral-foot training constraints."""

    sdf_path: str
    alignment_path: str
    clearance: float = 0.005
    clearance_weight: float = 2.0
    msdf_close_weight: float = 0.001
    msdf_close_margin: float = 0.001
    plantar_sdf_band: float = 0.035
    start_iter: int = 500
    warmup_iter: int = 1000
    max_surface_points: int = 50000
    max_watertight_points: int = 50000

    @classmethod
    def from_flags(cls, flags: object) -> "FootPriorLossConfig":
        return cls(
            sdf_path=str(getattr(flags, "foot_prior_sdf_path", "")),
            alignment_path=str(getattr(flags, "foot_prior_alignment_path", "")),
            clearance=float(getattr(flags, "foot_prior_clearance", 0.005)),
            clearance_weight=float(getattr(flags, "foot_prior_clearance_weight", 2.0)),
            msdf_close_weight=float(getattr(flags, "foot_prior_msdf_close_weight", 0.001)),
            msdf_close_margin=float(getattr(flags, "foot_prior_msdf_close_margin", 0.001)),
            plantar_sdf_band=float(getattr(flags, "foot_prior_plantar_sdf_band", 0.035)),
            start_iter=int(getattr(flags, "foot_prior_start_iter", 500)),
            warmup_iter=int(getattr(flags, "foot_prior_warmup_iter", 1000)),
            max_surface_points=int(getattr(flags, "foot_prior_max_surface_points", 50000)),
            max_watertight_points=int(getattr(flags, "foot_prior_max_watertight_points", 50000)),
        )


class FootPriorLoss(nn.Module):
    """Differentiable neutral-foot prior for GShell shoe coordinates."""

    def __init__(
        self,
        config: FootPriorLossConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config

        sdf_path = _resolve_path(config.sdf_path)
        alignment_path = _resolve_path(config.alignment_path)
        if not sdf_path.exists():
            raise FileNotFoundError(f"Foot SDF file not found: {sdf_path}")
        if not alignment_path.exists():
            raise FileNotFoundError(f"Foot alignment file not found: {alignment_path}")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.foot_sdf = FootSDFGrid.from_npz(
            str(sdf_path),
            config=FootSDFConfig(clearance=config.clearance),
            device=device,
        )
        alignment = FootAlignment.from_json(alignment_path)
        self.alignment = alignment

        self.register_buffer(
            "shoe_to_foot",
            torch.as_tensor(alignment.shoe_to_foot, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "plantar_coordinate",
            torch.tensor(float(alignment.plantar_z), dtype=torch.float32, device=device),
        )
        self.shoe_up_axis = int(alignment.config.shoe_up_axis)
        self.shoe_up_sign = float(axis_sign(alignment.config.shoe_up_sign))
        self.plantar_band = float(alignment.config.plantar_band)

    def schedule_weight(self, iteration: int) -> float:
        """Ramp the prior in after the base shape has started to form."""

        if iteration < self.config.start_iter:
            return 0.0
        if self.config.warmup_iter <= 0:
            return 1.0
        return min(1.0, float(iteration - self.config.start_iter) / float(self.config.warmup_iter))

    def transform_shoe_to_foot(self, points_shoe: torch.Tensor) -> torch.Tensor:
        """Map points from GShell shoe coordinates into raw SUPR foot coordinates."""

        if points_shoe.shape[-1] != 3:
            raise ValueError("points_shoe must have last dimension 3")

        matrix = self.shoe_to_foot.to(device=points_shoe.device, dtype=points_shoe.dtype)
        flat = points_shoe.reshape(-1, 3)
        ones = torch.ones((flat.shape[0], 1), dtype=flat.dtype, device=flat.device)
        homogeneous = torch.cat([flat, ones], dim=-1)
        transformed = homogeneous @ matrix.transpose(0, 1)
        return transformed[:, :3].reshape(points_shoe.shape)

    def query_shoe(self, points_shoe: torch.Tensor) -> torch.Tensor:
        """Query the foot SDF at points expressed in shoe coordinates."""

        points_foot = self.transform_shoe_to_foot(points_shoe)
        return self.foot_sdf.query(points_foot)

    def clearance_loss(self, points_shoe: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Penalize shell-surface points that enter or approach the foot."""

        points_shoe = self._subsample(points_shoe, self.config.max_surface_points)
        sdf_values = self.query_shoe(points_shoe)
        loss = torch.relu(self.config.clearance - sdf_values).square().mean()
        stats = self._sdf_stats(sdf_values, prefix="clearance")
        stats["clearance_loss_raw"] = float(loss.detach().cpu())
        return loss, stats

    def plantar_msdf_close_loss(
        self,
        points_shoe: torch.Tensor,
        msdf_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Push mSDF values positive near the plantar support region.

        Positive mSDF is the same direction used by the existing GShell
        close-regularizer. Here we apply it only under/near the aligned foot so
        the sole-side material is less likely to be cut away.
        """

        points_shoe, msdf_values = self._subsample_pair(
            points_shoe,
            msdf_values.reshape(-1),
            self.config.max_watertight_points,
        )
        with torch.no_grad():
            sdf_values = self.query_shoe(points_shoe)
            mask = self._plantar_region_mask(points_shoe, sdf_values)
        stats = {
            "plantar_candidate_count": int(mask.sum().detach().cpu()),
            "plantar_candidate_fraction": float(mask.float().mean().detach().cpu()) if mask.numel() else 0.0,
        }
        if not bool(mask.any()):
            return msdf_values.new_zeros(()), stats

        selected_msdf = msdf_values[mask]
        margin = float(self.config.msdf_close_margin)
        target = torch.full_like(selected_msdf, margin)
        loss = F.huber_loss(selected_msdf.clamp(max=margin), target, reduction="mean")
        stats["plantar_msdf_close_loss_raw"] = float(loss.detach().cpu())
        stats["plantar_msdf_mean"] = float(selected_msdf.detach().mean().cpu())
        return loss, stats

    def forward(
        self,
        iteration: int,
        surface_points: Optional[torch.Tensor] = None,
        watertight_points: Optional[torch.Tensor] = None,
        watertight_msdf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the weighted foot-prior loss and JSON-friendly stats."""

        device = self.shoe_to_foot.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        schedule_weight = self.schedule_weight(iteration)
        stats: Dict[str, float] = {
            "active": float(schedule_weight > 0.0),
            "schedule_weight": float(schedule_weight),
            "total": 0.0,
        }
        if schedule_weight <= 0.0:
            return total, stats

        if surface_points is not None and surface_points.numel() > 0 and self.config.clearance_weight > 0.0:
            raw_loss, loss_stats = self.clearance_loss(surface_points)
            weighted = raw_loss * self.config.clearance_weight * schedule_weight
            total = total + weighted
            stats.update(loss_stats)
            stats["clearance_loss_weighted"] = float(weighted.detach().cpu())

        if (
            watertight_points is not None
            and watertight_msdf is not None
            and watertight_points.numel() > 0
            and watertight_msdf.numel() > 0
            and self.config.msdf_close_weight > 0.0
        ):
            raw_loss, loss_stats = self.plantar_msdf_close_loss(watertight_points, watertight_msdf)
            weighted = raw_loss * self.config.msdf_close_weight * schedule_weight
            total = total + weighted
            stats.update(loss_stats)
            stats["plantar_msdf_close_loss_weighted"] = float(weighted.detach().cpu())

        stats["total"] = float(total.detach().cpu())
        return total, stats

    def _plantar_region_mask(self, points_shoe: torch.Tensor, sdf_values: torch.Tensor) -> torch.Tensor:
        signed_up = self.shoe_up_sign * points_shoe[:, self.shoe_up_axis]
        signed_plantar = self.shoe_up_sign * self.plantar_coordinate.to(
            device=points_shoe.device,
            dtype=points_shoe.dtype,
        )
        below_plantar = signed_up <= signed_plantar + self.plantar_band
        near_footprint = sdf_values.detach() <= float(self.config.plantar_sdf_band)
        return below_plantar & near_footprint

    @staticmethod
    def _subsample(points: torch.Tensor, max_points: int) -> torch.Tensor:
        if max_points <= 0 or points.shape[0] <= max_points:
            return points
        indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
        return points[indices]

    @staticmethod
    def _subsample_pair(
        points: torch.Tensor,
        values: torch.Tensor,
        max_points: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.shape[0] != values.shape[0]:
            raise ValueError("points and values must have matching first dimensions")
        if max_points <= 0 or points.shape[0] <= max_points:
            return points, values
        indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
        return points[indices], values[indices]

    def _sdf_stats(self, sdf_values: torch.Tensor, prefix: str) -> Dict[str, float]:
        detached = sdf_values.detach()
        return {
            f"{prefix}_sdf_mean": float(detached.mean().cpu()),
            f"{prefix}_sdf_min": float(detached.min().cpu()),
            f"{prefix}_inside_fraction": float((detached < 0.0).float().mean().cpu()),
            f"{prefix}_violation_fraction": float(
                (detached < self.config.clearance).float().mean().cpu()
            ),
        }
