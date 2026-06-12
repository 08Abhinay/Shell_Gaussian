"""Cross-section pseudo-last supervision for GShell training.

The pseudo-last is treated as approximate interior pseudo ground truth:
inside the last is empty cavity, the lower support region around and below it
is shoe material, and uncertain high-upper regions are ignored.

The stored pseudo-last SDF file uses the standard geometry convention
``negative = inside``. Internally this module immediately flips that query into
a GShell-style cavity field where ``positive = inside the represented volume``.
For the pseudo-last, the represented volume is the empty foot cavity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .foot_sdf import FootSDFConfig, FootSDFGrid


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    footshell_root = Path(__file__).resolve().parents[1]
    return footshell_root / candidate


@dataclass(frozen=True)
class PseudoLastPriorConfig:
    """Runtime settings for pseudo-last cross-section constraints."""

    sdf_path: str
    sections_path: str
    prior_mode: str = "cross_section"
    xsec_start_iter: int = 0
    xsec_warmup_iter: int = 250
    xsec_material_weight: float = 20.0
    xsec_empty_weight: float = 10.0
    xsec_surface_weight: float = 2.0
    xsec_msdf_keep_weight: float = 0.01
    xsec_x_slices: int = 64
    xsec_y_samples: int = 96
    xsec_z_samples: int = 96
    xsec_max_points: int = 49152
    xsec_grid_max_points: int = 0
    xsec_material_margin: float = 0.005
    xsec_empty_margin: float = 0.005
    xsec_surface_band: float = 0.003
    xsec_sole_depth: float = 0.025
    xsec_plantar_h_ratio: float = 0.12
    xsec_lower_wall_h_ratio: float = 0.45
    xsec_ignore_high_h_ratio: float = 0.85
    xsec_support_lateral_pad: float = 0.05
    xsec_msdf_margin: float = 0.005
    xsec_last_surface_points: int = 8192
    xsec_sole_sample_fraction: float = 0.40
    xsec_plantar_sample_fraction: float = 0.30
    xsec_sidewall_sample_fraction: float = 0.30

    # Compatibility with previous experimental configs. These fields are not
    # used by the cross-section prior, but accepting them avoids breaking old
    # command lines while we migrate configs.
    collision_weight: float = 0.0
    msdf_keep_weight: float = 0.0
    grid_msdf_keep_weight: float = 0.0
    containment_weight: float = 0.0
    use_field_conditioning: bool = False
    msdf_bias_strength: float = 0.0
    sdf_material_weight: float = 0.0

    @classmethod
    def from_flags(cls, flags: object) -> "PseudoLastPriorConfig":
        return cls(
            sdf_path=str(getattr(flags, "pseudo_last_sdf_path", "")),
            sections_path=str(getattr(flags, "pseudo_last_sections_path", "")),
            prior_mode=str(getattr(flags, "pseudo_last_prior_mode", "cross_section")),
            xsec_start_iter=int(getattr(flags, "pseudo_last_xsec_start_iter", 0)),
            xsec_warmup_iter=int(getattr(flags, "pseudo_last_xsec_warmup_iter", 250)),
            xsec_material_weight=float(getattr(flags, "pseudo_last_xsec_material_weight", 20.0)),
            xsec_empty_weight=float(getattr(flags, "pseudo_last_xsec_empty_weight", 10.0)),
            xsec_surface_weight=float(getattr(flags, "pseudo_last_xsec_surface_weight", 2.0)),
            xsec_msdf_keep_weight=float(getattr(flags, "pseudo_last_xsec_msdf_keep_weight", 0.01)),
            xsec_x_slices=int(getattr(flags, "pseudo_last_xsec_x_slices", 64)),
            xsec_y_samples=int(getattr(flags, "pseudo_last_xsec_y_samples", 96)),
            xsec_z_samples=int(getattr(flags, "pseudo_last_xsec_z_samples", 96)),
            xsec_max_points=int(getattr(flags, "pseudo_last_xsec_max_points", 49152)),
            xsec_grid_max_points=int(getattr(flags, "pseudo_last_xsec_grid_max_points", 0)),
            xsec_material_margin=float(getattr(flags, "pseudo_last_xsec_material_margin", 0.005)),
            xsec_empty_margin=float(getattr(flags, "pseudo_last_xsec_empty_margin", 0.005)),
            xsec_surface_band=float(getattr(flags, "pseudo_last_xsec_surface_band", 0.003)),
            xsec_sole_depth=float(getattr(flags, "pseudo_last_xsec_sole_depth", 0.025)),
            xsec_plantar_h_ratio=float(getattr(flags, "pseudo_last_xsec_plantar_h_ratio", 0.12)),
            xsec_lower_wall_h_ratio=float(getattr(flags, "pseudo_last_xsec_lower_wall_h_ratio", 0.45)),
            xsec_ignore_high_h_ratio=float(getattr(flags, "pseudo_last_xsec_ignore_high_h_ratio", 0.85)),
            xsec_support_lateral_pad=float(getattr(flags, "pseudo_last_xsec_support_lateral_pad", 0.05)),
            xsec_msdf_margin=float(getattr(flags, "pseudo_last_xsec_msdf_margin", 0.005)),
            xsec_last_surface_points=int(getattr(flags, "pseudo_last_xsec_last_surface_points", 8192)),
            xsec_sole_sample_fraction=float(getattr(flags, "pseudo_last_xsec_sole_sample_fraction", 0.40)),
            xsec_plantar_sample_fraction=float(getattr(flags, "pseudo_last_xsec_plantar_sample_fraction", 0.30)),
            xsec_sidewall_sample_fraction=float(getattr(flags, "pseudo_last_xsec_sidewall_sample_fraction", 0.30)),
            collision_weight=float(getattr(flags, "pseudo_last_collision_weight", 0.0)),
            msdf_keep_weight=float(getattr(flags, "pseudo_last_msdf_keep_weight", 0.0)),
            grid_msdf_keep_weight=float(getattr(flags, "pseudo_last_grid_msdf_keep_weight", 0.0)),
            containment_weight=float(getattr(flags, "pseudo_last_containment_weight", 0.0)),
            use_field_conditioning=bool(getattr(flags, "pseudo_last_use_field_conditioning", False)),
            msdf_bias_strength=float(getattr(flags, "pseudo_last_msdf_bias_strength", 0.0)),
            sdf_material_weight=float(getattr(flags, "pseudo_last_sdf_material_weight", 0.0)),
        )


class PseudoLastPriorLoss(nn.Module):
    """Dense cross-section prior for hidden shoe interior topology."""

    def __init__(
        self,
        config: PseudoLastPriorConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.prior_mode != "cross_section":
            raise ValueError(
                "Only pseudo_last_prior_mode='cross_section' is supported in the cleaned prior"
            )

        sdf_path = _resolve_path(config.sdf_path)
        sections_path = _resolve_path(config.sections_path)
        if not sdf_path.exists():
            raise FileNotFoundError(f"Pseudo-last SDF file not found: {sdf_path}")
        if not sections_path.exists():
            raise FileNotFoundError(f"Pseudo-last sections file not found: {sections_path}")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FootSDFGrid keeps the standard SDF sign from disk:
        #   negative = inside pseudo-last, positive = outside.
        # All classification below uses _query_last_cavity_field_chunked(), which
        # flips this into:
        #   positive = inside pseudo-last cavity, negative = outside.
        self.last_sdf = FootSDFGrid.from_npz(
            str(sdf_path),
            config=FootSDFConfig(clearance=config.xsec_surface_band),
            device=device,
        )
        self._load_sections(sections_path, device)

        self.register_buffer("empty_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("material_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("sole_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("plantar_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("sidewall_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("surface_points", torch.empty(0, 3, dtype=torch.float32, device=device))
        self.register_buffer("material_grid_indices", torch.empty(0, dtype=torch.long, device=device))
        self.register_buffer("material_grid_weights", torch.empty(0, dtype=torch.float32, device=device))

        self.label_stats: Dict[str, float] = {}
        self.grid_stats: Dict[str, float] = {}
        self._build_cross_section_pool()

    def _load_sections(self, sections_path: Path, device: torch.device) -> None:
        data = np.load(sections_path, allow_pickle=True)
        required_keys = {
            "sections",
            "bottom_sections",
            "x",
            "center_z",
            "left_z",
            "right_z",
            "height",
            "support_left_z",
            "support_right_z",
        }
        missing_keys = sorted(required_keys.difference(data.files))
        if missing_keys:
            raise KeyError(f"Pseudo-last sections file is missing keys: {missing_keys}")

        x = np.asarray(data["x"], dtype=np.float32)
        center_z = np.asarray(data["center_z"], dtype=np.float32)
        left_z = np.asarray(data["left_z"], dtype=np.float32)
        right_z = np.asarray(data["right_z"], dtype=np.float32)
        height = np.asarray(data["height"], dtype=np.float32)
        support_left_z = np.asarray(data["support_left_z"], dtype=np.float32)
        support_right_z = np.asarray(data["support_right_z"], dtype=np.float32)
        bottom_sections = np.asarray(data["bottom_sections"], dtype=np.float32)
        sections = np.asarray(data["sections"], dtype=np.float32)

        if x.ndim != 1 or x.shape[0] < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("Pseudo-last x grid must be strictly increasing")
        for name, values in {
            "center_z": center_z,
            "left_z": left_z,
            "right_z": right_z,
            "height": height,
            "support_left_z": support_left_z,
            "support_right_z": support_right_z,
        }.items():
            if values.shape != x.shape:
                raise ValueError(f"{name} must have the same shape as x")
        if bottom_sections.ndim != 3 or bottom_sections.shape[0] != x.shape[0] or bottom_sections.shape[2] != 3:
            raise ValueError("bottom_sections must have shape [Nx, Nu, 3]")
        if sections.ndim != 3 or sections.shape[0] != x.shape[0] or sections.shape[2] != 3:
            raise ValueError("sections must have shape [Nx, Nu, 3]")

        half_width = 0.5 * np.maximum(right_z - left_z, 1e-6).astype(np.float32)
        support_half_width = 0.5 * np.maximum(support_right_z - support_left_z, 1e-6).astype(np.float32)
        surface_points = sections.reshape(-1, 3)
        finite_mask = np.isfinite(surface_points).all(axis=1)
        surface_points = surface_points[finite_mask].astype(np.float32)
        if surface_points.size == 0:
            raise ValueError("Pseudo-last sections do not contain any finite surface points")

        self.register_buffer("section_x", torch.as_tensor(x, dtype=torch.float32, device=device))
        self.register_buffer("center_z", torch.as_tensor(center_z, dtype=torch.float32, device=device))
        self.register_buffer("half_width", torch.as_tensor(half_width, dtype=torch.float32, device=device))
        self.register_buffer("support_half_width", torch.as_tensor(support_half_width, dtype=torch.float32, device=device))
        self.register_buffer("section_height", torch.as_tensor(height, dtype=torch.float32, device=device))
        self.register_buffer(
            "bottom_y_grid",
            torch.as_tensor(bottom_sections[..., 1], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "all_last_surface_points",
            torch.as_tensor(surface_points, dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def _build_cross_section_pool(self) -> None:
        cfg = self.config
        nx = max(int(cfg.xsec_x_slices), 2)
        ny = max(int(cfg.xsec_y_samples), 8)
        nz = max(int(cfg.xsec_z_samples), 8)

        x_axis = torch.linspace(self.section_x[0], self.section_x[-1], nx, device=self.section_x.device)
        lateral_axis = torch.linspace(
            -1.0 - float(cfg.xsec_support_lateral_pad),
            1.0 + float(cfg.xsec_support_lateral_pad),
            nz,
            device=self.section_x.device,
        )
        vertical_axis = torch.linspace(-1.0, float(cfg.xsec_ignore_high_h_ratio), ny, device=self.section_x.device)

        center = self._interp_1d(x_axis, self.center_z)
        support_half = self._interp_1d(x_axis, self.support_half_width).clamp(min=1e-6)
        height = self._interp_1d(x_axis, self.section_height).clamp(min=1e-6)
        z_grid = center[:, None] + lateral_axis[None, :] * support_half[:, None]
        xz_points = torch.stack(
            [
                x_axis[:, None].expand(-1, nz),
                torch.zeros((nx, nz), dtype=x_axis.dtype, device=x_axis.device),
                z_grid,
            ],
            dim=-1,
        ).reshape(-1, 3)
        xz_center = center[:, None].expand(-1, nz).reshape(-1)
        xz_half = self._interp_1d(xz_points[:, 0], self.half_width).clamp(min=1e-6)
        bottom_y = self._interp_bottom_y(xz_points, center=xz_center, half_width=xz_half).reshape(nx, nz)

        h_positive = vertical_axis.clamp(min=0.0)[None, :, None] * height[:, None, None]
        h_negative = vertical_axis.clamp(max=0.0)[None, :, None] * float(cfg.xsec_sole_depth)
        h_abs = h_positive + h_negative
        y_grid = bottom_y[:, None, :] - h_abs
        x_grid = x_axis[:, None, None].expand(nx, ny, nz)
        z_grid_full = z_grid[:, None, :].expand(nx, ny, nz)
        points = torch.stack([x_grid, y_grid, z_grid_full], dim=-1).reshape(-1, 3)

        last_cavity = self._query_last_cavity_field_chunked(points)
        labels = self._classify_points(points, last_cavity)
        empty_points = points[labels["empty_mask"]]
        material_points = points[labels["material_mask"]]
        sole_points = points[labels["sole_mask"]]
        # The simple v1 regions can overlap. Keep the overlap for sampling so
        # bottom-adjacent points do not disappear when they are also near sides.
        plantar_points = points[labels["plantar_mask"]]
        sidewall_points = points[labels["sidewall_mask"]]
        surface_points = self.all_last_surface_points

        self.empty_points = empty_points.detach().to(dtype=torch.float32)
        self.material_points = material_points.detach().to(dtype=torch.float32)
        self.sole_points = sole_points.detach().to(dtype=torch.float32)
        self.plantar_points = plantar_points.detach().to(dtype=torch.float32)
        self.sidewall_points = sidewall_points.detach().to(dtype=torch.float32)
        self.surface_points = surface_points.detach().to(dtype=torch.float32)
        self.label_stats = {
            "xsec_pool_total_count": int(points.shape[0]),
            "xsec_empty_pool_count": int(empty_points.shape[0]),
            "xsec_material_pool_count": int(material_points.shape[0]),
            "xsec_surface_pool_count": int(surface_points.shape[0]),
            "xsec_sole_pool_count": int(labels["sole_mask"].sum().detach().cpu()),
            "xsec_plantar_pool_count": int(plantar_points.shape[0]),
            "xsec_sidewall_pool_count": int(sidewall_points.shape[0]),
        }
        if empty_points.numel() == 0:
            raise ValueError("Cross-section pseudo-last prior found no empty cavity points")
        if material_points.numel() == 0:
            raise ValueError("Cross-section pseudo-last prior found no lower material points")

    @torch.no_grad()
    def precompute_grid_keep_weights(
        self,
        grid_points: torch.Tensor,
        tet_edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        del tet_edges
        points = grid_points.reshape(-1, 3).to(device=self.section_x.device, dtype=self.section_x.dtype)
        last_cavity = self._query_last_cavity_field_chunked(points)
        labels = self._classify_points(points, last_cavity)
        weights = labels["material_weight"].detach().to(dtype=torch.float32)
        mask = weights > 0.0
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1).to(dtype=torch.long)
        self.material_grid_indices = indices
        self.material_grid_weights = weights[indices]
        coords = self.section_coordinates(points)
        selected_h = coords["h"][indices] if indices.numel() > 0 else torch.empty(0, device=points.device)
        selected_h_norm = (
            coords["h_norm"][indices] if indices.numel() > 0 else torch.empty(0, device=points.device)
        )
        far_below = selected_h < -float(self.config.xsec_sole_depth)
        self.grid_stats = {
            "xsec_grid_msdf_candidate_count": int(indices.numel()),
            "xsec_grid_msdf_candidate_fraction": float(mask.float().mean().detach().cpu()) if mask.numel() else 0.0,
            "xsec_grid_sole_candidate_count": int(labels["sole_mask"].sum().detach().cpu()),
            "xsec_grid_sidewall_candidate_count": int(labels["sidewall_mask"].sum().detach().cpu()),
            "xsec_grid_plantar_candidate_count": int(labels["plantar_mask"].sum().detach().cpu()),
            "xsec_grid_far_below_count": int(far_below.sum().detach().cpu()) if far_below.numel() else 0,
            "xsec_grid_h_min": self._stat_percentile(selected_h, 0),
            "xsec_grid_h_p50": self._stat_percentile(selected_h, 50),
            "xsec_grid_h_p99": self._stat_percentile(selected_h, 99),
            "xsec_grid_h_norm_p50": self._stat_percentile(selected_h_norm, 50),
        }
        stats = dict(self.label_stats)
        stats.update(self.grid_stats)
        return stats

    def schedule_weight(self, iteration: int) -> float:
        if iteration < self.config.xsec_start_iter:
            return 0.0
        if self.config.xsec_warmup_iter <= 0:
            return 1.0
        return min(
            1.0,
            float(iteration - self.config.xsec_start_iter) / float(self.config.xsec_warmup_iter),
        )

    def condition_grid_msdf(
        self,
        raw_msdf: torch.Tensor,
        iteration: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del iteration
        return raw_msdf, {"xsec_field_conditioning_disabled": 1.0}

    def forward(
        self,
        iteration: int,
        shell_sdf_query_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        grid_msdf: Optional[torch.Tensor] = None,
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = self.section_x.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        schedule = self.schedule_weight(iteration)
        stats: Dict[str, float] = {
            "active": float(schedule > 0.0),
            "schedule_weight": float(schedule),
            "total": 0.0,
            **self.label_stats,
            **self.grid_stats,
        }
        if schedule <= 0.0:
            return total, stats
        if shell_sdf_query_fn is None:
            raise ValueError("Cross-section pseudo-last prior requires shell_sdf_query_fn")

        sole_count, plantar_count, sidewall_count, empty_count, surface_count = self._sample_counts()
        sole_points = self._sample_rows(self.sole_points, sole_count)
        plantar_points = self._sample_rows(self.plantar_points, plantar_count)
        sidewall_points = self._sample_rows(self.sidewall_points, sidewall_count)
        material_groups = [
            points
            for points in (sole_points, plantar_points, sidewall_points)
            if points.numel() > 0
        ]
        material_points = (
            torch.cat(material_groups, dim=0)
            if material_groups
            else torch.empty(0, 3, dtype=torch.float32, device=device)
        )
        empty_points = self._sample_rows(self.empty_points, empty_count)
        surface_points = self._sample_rows(self.surface_points, surface_count)

        if material_points.numel() > 0:
            material_sdf = shell_sdf_query_fn(material_points).reshape(-1)
            material_loss = torch.relu(float(self.config.xsec_material_margin) - material_sdf).square().mean()
            total = total + material_loss * float(self.config.xsec_material_weight) * schedule
            mat_detached = material_sdf.detach()
            sole_detached, plantar_detached, sidewall_detached = self._split_material_sdf_detached(
                mat_detached,
                (sole_points.shape[0], plantar_points.shape[0], sidewall_points.shape[0]),
            )
        else:
            material_loss = total.new_zeros(())
            mat_detached = torch.empty(0, dtype=total.dtype, device=total.device)
            sole_detached = torch.empty(0, dtype=total.dtype, device=total.device)
            plantar_detached = torch.empty(0, dtype=total.dtype, device=total.device)
            sidewall_detached = torch.empty(0, dtype=total.dtype, device=total.device)

        empty_sdf = shell_sdf_query_fn(empty_points).reshape(-1)
        empty_loss = torch.relu(empty_sdf + float(self.config.xsec_empty_margin)).square().mean()
        total = total + empty_loss * float(self.config.xsec_empty_weight) * schedule
        empty_detached = empty_sdf.detach()

        if surface_points.numel() > 0 and float(self.config.xsec_surface_weight) > 0.0:
            surface_sdf = shell_sdf_query_fn(surface_points).reshape(-1)
            surface_loss = F.smooth_l1_loss(
                surface_sdf,
                torch.zeros_like(surface_sdf),
                beta=float(self.config.xsec_surface_band),
            )
            total = total + surface_loss * float(self.config.xsec_surface_weight) * schedule
            surface_detached = surface_sdf.detach()
        else:
            surface_loss = total.new_zeros(())
            surface_detached = torch.empty(0, dtype=total.dtype, device=total.device)

        msdf_loss = total.new_zeros(())
        msdf_count = 0
        msdf_positive = 0.0
        if (
            grid_msdf is not None
            and grid_msdf.numel() > 0
            and self.material_grid_indices.numel() > 0
            and float(self.config.xsec_msdf_keep_weight) > 0.0
        ):
            indices = self.material_grid_indices.to(device=grid_msdf.device)
            weights = self.material_grid_weights.to(device=grid_msdf.device, dtype=grid_msdf.dtype)
            if self.config.xsec_grid_max_points > 0 and indices.numel() > self.config.xsec_grid_max_points:
                choice = torch.randperm(indices.numel(), device=grid_msdf.device)[: self.config.xsec_grid_max_points]
                indices = indices[choice]
                weights = weights[choice]
            selected = grid_msdf.reshape(-1)[indices]
            per_point = torch.relu(float(self.config.xsec_msdf_margin) - selected).square()
            msdf_loss = self._weighted_mean(per_point, weights)
            total = total + msdf_loss * float(self.config.xsec_msdf_keep_weight) * schedule
            msdf_count = int(indices.numel())
            msdf_positive = float((selected.detach() > 0.0).float().mean().cpu()) if selected.numel() else 0.0

        stats.update(
            {
                "xsec_loss": float(total.detach().cpu()),
                "xsec_material_loss_raw": float(material_loss.detach().cpu()),
                "xsec_empty_loss_raw": float(empty_loss.detach().cpu()),
                "xsec_surface_loss_raw": float(surface_loss.detach().cpu()),
                "xsec_msdf_loss_raw": float(msdf_loss.detach().cpu()),
                "xsec_mat_pts": int(material_points.shape[0]),
                "xsec_sole_pts": int(sole_points.shape[0]),
                "xsec_plantar_pts": int(plantar_points.shape[0]),
                "xsec_sidewall_pts": int(sidewall_points.shape[0]),
                "xsec_empty_pts": int(empty_points.shape[0]),
                "xsec_surface_pts": int(surface_points.shape[0]),
                "xsec_mat_sdf_mean": self._stat_mean(mat_detached),
                "xsec_mat_sdf_pos": self._stat_fraction(mat_detached > 0.0),
                "xsec_mat_sdf_margin": self._stat_fraction(
                    mat_detached >= float(self.config.xsec_material_margin)
                ),
                "xsec_sole_sdf_pos": self._stat_fraction(sole_detached > 0.0),
                "xsec_plantar_sdf_pos": self._stat_fraction(plantar_detached > 0.0),
                "xsec_sidewall_sdf_pos": self._stat_fraction(sidewall_detached > 0.0),
                "xsec_empty_sdf_mean": float(empty_detached.mean().cpu()),
                "xsec_empty_sdf_neg": float((empty_detached < 0.0).float().mean().cpu()),
                "xsec_empty_sdf_margin": float(
                    (empty_detached <= -float(self.config.xsec_empty_margin)).float().mean().cpu()
                ),
                "xsec_surface_abs_sdf": float(surface_detached.abs().mean().cpu()) if surface_detached.numel() else 0.0,
                "xsec_grid_msdf_pts": msdf_count,
                "xsec_grid_msdf_pos": msdf_positive,
                "total": float(total.detach().cpu()),
            }
        )
        return total, stats

    def _sample_counts(self) -> Tuple[int, int, int, int, int]:
        max_points = int(self.config.xsec_max_points)
        material_capacities = [
            int(self.sole_points.shape[0]),
            int(self.plantar_points.shape[0]),
            int(self.sidewall_points.shape[0]),
        ]
        surface_cap = min(int(self.surface_points.shape[0]), int(self.config.xsec_last_surface_points))
        if max_points <= 0:
            return (
                material_capacities[0],
                material_capacities[1],
                material_capacities[2],
                int(self.empty_points.shape[0]),
                surface_cap,
            )
        mat_budget = min(sum(material_capacities), int(round(max_points * 0.50)))
        empty_count = min(int(self.empty_points.shape[0]), int(round(max_points * 0.35)))
        surface_budget = max_points - mat_budget - empty_count
        surface_count = min(surface_cap, max(surface_budget, 0))
        leftover = max_points - mat_budget - empty_count - surface_count
        if leftover > 0:
            add_empty = min(int(self.empty_points.shape[0]) - empty_count, leftover)
            empty_count += max(add_empty, 0)
            leftover -= max(add_empty, 0)
        if leftover > 0:
            add_mat = min(sum(material_capacities) - mat_budget, leftover)
            mat_budget += max(add_mat, 0)
        sole_count, plantar_count, sidewall_count = self._allocate_material_counts(
            material_capacities,
            mat_budget,
        )
        return sole_count, plantar_count, sidewall_count, empty_count, surface_count

    def _allocate_material_counts(
        self,
        capacities: Tuple[int, int, int] | list[int],
        budget: int,
    ) -> Tuple[int, int, int]:
        target = min(max(int(budget), 0), int(sum(capacities)))
        if target <= 0:
            return 0, 0, 0

        fractions = np.asarray(
            [
                max(float(self.config.xsec_sole_sample_fraction), 0.0),
                max(float(self.config.xsec_plantar_sample_fraction), 0.0),
                max(float(self.config.xsec_sidewall_sample_fraction), 0.0),
            ],
            dtype=np.float64,
        )
        capacities_array = np.asarray(capacities, dtype=np.int64)
        active = capacities_array > 0
        fractions = np.where(active, fractions, 0.0)
        if fractions.sum() <= 0.0:
            fractions = active.astype(np.float64)
        fractions = fractions / max(fractions.sum(), np.finfo(np.float64).eps)

        raw = fractions * float(target)
        counts = np.floor(raw).astype(np.int64)
        counts = np.minimum(counts, capacities_array)

        # Distribute leftover samples to groups with remaining capacity. The
        # fractional remainder sets the first pass, then larger target fractions
        # get priority. This keeps the sampling deterministic and balanced.
        while counts.sum() < target:
            remaining = capacities_array - counts
            candidates = np.nonzero(remaining > 0)[0]
            if candidates.size == 0:
                break
            scores = (raw - counts)[candidates] + fractions[candidates] * 1e-3
            chosen = int(candidates[int(np.argmax(scores))])
            counts[chosen] += 1
        return int(counts[0]), int(counts[1]), int(counts[2])

    @staticmethod
    def _split_material_sdf_detached(
        values: torch.Tensor,
        counts: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        chunks = []
        for count in counts:
            end = start + int(count)
            chunks.append(values[start:end])
            start = end
        return chunks[0], chunks[1], chunks[2]

    def _classify_points(
        self,
        points: torch.Tensor,
        last_cavity_field: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Classify query points using a positive-inside pseudo-last cavity field.

        ``last_cavity_field`` follows the same sign style as GShell's occupancy
        field: positive means inside the represented volume. Here, that volume is
        the foot/last cavity, not shoe material.
        """
        coords = self.section_coordinates(points)
        s_raw = coords["s_raw"]
        h = coords["h"]
        h_norm = coords["h_norm"]
        u_last_abs = coords["u_last"].abs()
        u_support_abs = coords["u_support"].abs()

        within_length = (s_raw >= 0.0) & (s_raw <= 1.0)
        within_support = u_support_abs <= (1.0 + float(self.config.xsec_support_lateral_pad))
        low_enough = h_norm <= float(self.config.xsec_ignore_high_h_ratio)
        inside_cavity = last_cavity_field > float(self.config.xsec_surface_band)
        outside_cavity = last_cavity_field < -float(self.config.xsec_surface_band)

        sole_mask = (
            within_length
            & within_support
            & low_enough
            & outside_cavity
            & (h < 0.0)
            & (h >= -float(self.config.xsec_sole_depth))
        )
        plantar_mask = (
            within_length
            & within_support
            & low_enough
            & outside_cavity
            & (h_norm >= 0.0)
            & (h_norm <= float(self.config.xsec_plantar_h_ratio))
            & (u_last_abs <= 1.15)
        )
        sidewall_mask = (
            within_length
            & within_support
            & low_enough
            & outside_cavity
            & (h_norm >= 0.0)
            & (h_norm <= float(self.config.xsec_lower_wall_h_ratio))
            & (u_last_abs >= 0.95)
        )
        material_mask = sole_mask | plantar_mask | sidewall_mask
        empty_mask = within_length & inside_cavity

        material_weight = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
        material_weight = torch.maximum(material_weight, sole_mask.to(points.dtype) * 1.0)
        material_weight = torch.maximum(material_weight, plantar_mask.to(points.dtype) * 0.9)
        material_weight = torch.maximum(material_weight, sidewall_mask.to(points.dtype) * 0.75)
        return {
            "empty_mask": empty_mask,
            "material_mask": material_mask,
            "material_weight": material_weight,
            "sole_mask": sole_mask,
            "plantar_mask": plantar_mask,
            "sidewall_mask": sidewall_mask,
        }

    def section_coordinates(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = points.reshape(-1, 3)
        x = flat[:, 0]
        y = flat[:, 1]
        z = flat[:, 2]
        x0 = self.section_x[0].to(device=flat.device, dtype=flat.dtype)
        x1 = self.section_x[-1].to(device=flat.device, dtype=flat.dtype)
        length = torch.clamp(x1 - x0, min=torch.finfo(flat.dtype).eps)
        s_raw = (x - x0) / length
        center = self._interp_1d(x, self.center_z)
        half_width = self._interp_1d(x, self.half_width).clamp(min=1e-6)
        support_half_width = self._interp_1d(x, self.support_half_width).clamp(min=1e-6)
        height = self._interp_1d(x, self.section_height).clamp(min=1e-6)
        bottom_y = self._interp_bottom_y(flat, center=center, half_width=half_width)
        h = bottom_y - y
        return {
            "s_raw": s_raw,
            "s": s_raw.clamp(0.0, 1.0),
            "u_last": (z - center) / half_width,
            "u_support": (z - center) / support_half_width,
            "h": h,
            "h_norm": h / height,
            "center_z": center,
            "half_width": half_width,
            "support_half_width": support_half_width,
            "height": height,
            "bottom_y": bottom_y,
        }

    def _query_last_standard_sdf_chunked(
        self,
        points: torch.Tensor,
        chunk_size: int = 262144,
    ) -> torch.Tensor:
        """Query the stored pseudo-last SDF using its on-disk standard sign."""
        chunks = []
        for start in range(0, points.shape[0], chunk_size):
            chunks.append(self.last_sdf.query(points[start : start + chunk_size]))
        return torch.cat(chunks, dim=0)

    def _query_last_cavity_field_chunked(
        self,
        points: torch.Tensor,
        chunk_size: int = 262144,
    ) -> torch.Tensor:
        """Query a GShell-style pseudo-last cavity field.

        Positive values are inside the pseudo-last cavity; negative values are
        outside the pseudo-last cavity.
        """
        return -self._query_last_standard_sdf_chunked(points, chunk_size=chunk_size)

    def _query_last_sdf_chunked(self, points: torch.Tensor, chunk_size: int = 262144) -> torch.Tensor:
        """Backward-compatible alias for the positive-inside cavity field.

        Older notebooks/scripts call this private helper. Keep them working, but
        return the cleaned sign convention used by this module.
        """
        return self._query_last_cavity_field_chunked(points, chunk_size=chunk_size)

    def _interp_1d(self, query_x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        section_x = self.section_x.to(device=query_x.device, dtype=query_x.dtype)
        values = values.to(device=query_x.device, dtype=query_x.dtype)
        idx0, idx1, t = self._x_interp_indices(query_x, section_x)
        return values[idx0] * (1.0 - t) + values[idx1] * t

    def _interp_bottom_y(
        self,
        points: torch.Tensor,
        center: torch.Tensor,
        half_width: torch.Tensor,
    ) -> torch.Tensor:
        x = points[:, 0]
        z = points[:, 2]
        section_x = self.section_x.to(device=points.device, dtype=points.dtype)
        bottom_y_grid = self.bottom_y_grid.to(device=points.device, dtype=points.dtype)
        idx0, idx1, tx = self._x_interp_indices(x, section_x)

        lateral = ((z - center) / half_width).clamp(-1.0, 1.0)
        nu = bottom_y_grid.shape[1]
        lateral_index = (lateral + 1.0) * 0.5 * float(nu - 1)
        j0 = torch.floor(lateral_index).long().clamp(0, nu - 1)
        j1 = (j0 + 1).clamp(0, nu - 1)
        tu = (lateral_index - j0.to(lateral_index.dtype)).clamp(0.0, 1.0)

        y00 = bottom_y_grid[idx0, j0]
        y01 = bottom_y_grid[idx0, j1]
        y10 = bottom_y_grid[idx1, j0]
        y11 = bottom_y_grid[idx1, j1]
        y0 = y00 * (1.0 - tu) + y01 * tu
        y1 = y10 * (1.0 - tu) + y11 * tu
        return y0 * (1.0 - tx) + y1 * tx

    @staticmethod
    def _x_interp_indices(
        query_x: torch.Tensor,
        section_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clamped_x = query_x.clamp(section_x[0], section_x[-1])
        idx1 = torch.searchsorted(section_x, clamped_x, right=False)
        idx1 = idx1.clamp(1, section_x.shape[0] - 1)
        idx0 = idx1 - 1
        x0 = section_x[idx0]
        x1 = section_x[idx1]
        t = (clamped_x - x0) / torch.clamp(x1 - x0, min=torch.finfo(query_x.dtype).eps)
        return idx0, idx1, t

    @staticmethod
    def _sample_rows(points: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or points.shape[0] <= count:
            return points
        indices = torch.randperm(points.shape[0], device=points.device)[:count]
        return points[indices]

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp(min=torch.finfo(values.dtype).eps)

    @staticmethod
    def _stat_mean(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        return float(values.detach().mean().cpu())

    @staticmethod
    def _stat_fraction(mask: torch.Tensor) -> float:
        if mask.numel() == 0:
            return 0.0
        return float(mask.detach().to(dtype=torch.float32).mean().cpu())

    @staticmethod
    def _stat_percentile(values: torch.Tensor, percentile: float) -> float:
        if values.numel() == 0:
            return 0.0
        return float(
            torch.quantile(
                values.detach().to(dtype=torch.float32),
                float(percentile) / 100.0,
            ).cpu()
        )
