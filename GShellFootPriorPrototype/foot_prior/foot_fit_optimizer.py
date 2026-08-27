"""Optimization-based SUPR foot fitting inside a pseudo shoe cavity.

This module implements the Section 1.4 fitting stage. It does not retrain
GShell and it does not change the shoe mesh. It refines an existing
``FootAlignment`` by fitting the already-aligned SUPR foot into a pseudo-cavity
estimated from the support-footprint JSON produced by ``support_footprint.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .foot_alignment import (
    FootAlignment,
    FootAlignmentConfig,
    MeshData,
    axis_sign,
    make_supr_to_shoe_axis_remap,
    mesh_bounds,
    plantar_coordinate,
    remap_supr_to_shoe_axes,
    transform_points,
)


@dataclass(frozen=True)
class PseudoCavityConfig:
    """Settings for turning the 2D support footprint into a 3D fitting cavity."""

    shoe_length_axis: int = 0
    shoe_up_axis: int = 1
    shoe_width_axis: int = 2
    shoe_up_sign: float = -1.0
    side_margin: float = 0.0025
    x_wall_margin_ratio: float = 0.010
    top_percentile: float = 3.0
    top_margin_ratio: float = 0.08


@dataclass(frozen=True)
class FootFitOptimizerConfig:
    """Optimizer knobs for Section 1.4 foot fitting."""

    device: str = "cuda"
    dtype: str = "float32"
    style_mode: str = "auto"
    boot_height_ratio_threshold: float = 0.85
    adam_steps: int = 160
    adam_lr: float = 0.035
    lbfgs_steps: int = 25
    max_yaw_degrees: float = 14.0
    max_pitch_degrees: float = 8.0
    max_roll_degrees: float = 6.0
    max_log_scale_delta: float = 0.16
    max_translation_x_ratio: float = 0.10
    max_translation_y_ratio: float = 0.12
    max_translation_z_ratio: float = 0.12
    footbed_offset_init_ratio: float = 0.000
    footbed_offset_min_ratio: float = 0.000
    footbed_offset_max_ratio: float = 0.000
    plantar_band_ratio: float = 0.080
    fit_height_fraction: float = 0.86
    foot_axis_slice_count: int = 28
    footbed_mask_threshold: float = 0.45
    wall_clearance: float = 0.0025
    heel_gap_min_ratio: float = 0.010
    heel_gap_max_ratio: float = 0.085
    toe_gap_min_ratio: float = 0.018
    toe_gap_max_ratio: float = 0.115
    plantar_contact_lower: float = 0.000
    plantar_contact_upper: float = 0.008
    arch_contact_lower: float = 0.002
    arch_contact_upper: float = 0.026
    toe_contact_lower: float = 0.001
    toe_contact_upper: float = 0.020
    cavity_weight: float = 90.0
    plantar_weight: float = 135.0
    gap_weight: float = 70.0
    axis_weight: float = 35.0
    footbed_mask_weight: float = 0.0
    width_slack_weight: float = 20.0
    side_balance_weight: float = 0.0
    side_total_clearance_weight: float = 0.0
    length_balance_weight: float = 0.0
    prior_weight: float = 10.0
    top_weight: float = 4.0
    side_total_clearance_min_ratio: float = 0.015
    side_total_clearance_max_ratio: float = 0.400
    side_region_edge_quantile: float = 0.08
    sole_yaw_prior_blend: float = 0.20
    yaw_prior_factor: float = 0.30
    translation_prior_factor: float = 0.0
    offset_prior_factor: float = 0.20
    multistart_yaw_degrees: Tuple[float, ...] = (-5.0, 0.0, 5.0)
    multistart_scale_delta: Tuple[float, ...] = (-0.035, 0.0, 0.035)
    multistart_z_ratio: Tuple[float, ...] = (-0.025, 0.0, 0.025)
    multistart_offset_ratio_delta: Tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class PseudoCavity:
    """A simple 3D cavity built from the shoe's support footprint."""

    centerline_x: np.ndarray
    centerline_z: np.ndarray
    left_boundary_z: np.ndarray
    right_boundary_z: np.ndarray
    floor_y: np.ndarray
    footbed_y: np.ndarray
    support_length: float
    top_y: float
    config: PseudoCavityConfig
    footbed_x_centers: Optional[np.ndarray] = None
    footbed_z_centers: Optional[np.ndarray] = None
    footbed_mask: Optional[np.ndarray] = None
    footbed_heightmap: Optional[np.ndarray] = None
    footbed_source: str = "support_axis_profile"
    sole_yaw_degrees: float = 0.0
    source_path: Optional[str] = None

    @property
    def x_min(self) -> float:
        return float(self.centerline_x.min())

    @property
    def x_max(self) -> float:
        return float(self.centerline_x.max())

    def to_npz(self, path: str | Path, *, metrics_json: Optional[Dict[str, object]] = None) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            centerline_x=self.centerline_x.astype(np.float32),
            centerline_z=self.centerline_z.astype(np.float32),
            left_boundary_z=self.left_boundary_z.astype(np.float32),
            right_boundary_z=self.right_boundary_z.astype(np.float32),
            floor_y=self.floor_y.astype(np.float32),
            footbed_y=self.footbed_y.astype(np.float32),
            support_length=np.asarray([self.support_length], dtype=np.float32),
            top_y=np.asarray([self.top_y], dtype=np.float32),
            sole_yaw_degrees=np.asarray([self.sole_yaw_degrees], dtype=np.float32),
            footbed_source=np.asarray(str(self.footbed_source)),
            config_json=np.asarray(json.dumps(asdict(self.config))),
            metrics_json=np.asarray(json.dumps(metrics_json or {})),
            source_path=np.asarray(str(self.source_path or "")),
        )
        if self.footbed_heightmap is not None:
            with np.load(output_path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays.update(
                footbed_x_centers=np.asarray(self.footbed_x_centers, dtype=np.float32),
                footbed_z_centers=np.asarray(self.footbed_z_centers, dtype=np.float32),
                footbed_mask=np.asarray(self.footbed_mask, dtype=bool),
                footbed_heightmap=np.asarray(self.footbed_heightmap, dtype=np.float32),
            )
            np.savez_compressed(output_path, **arrays)

    def sample_footbed_numpy(self, query_x: np.ndarray, query_z: np.ndarray) -> np.ndarray:
        if self.footbed_heightmap is None or self.footbed_x_centers is None or self.footbed_z_centers is None:
            return np.interp(
                np.clip(query_x, self.x_min, self.x_max),
                self.centerline_x,
                self.footbed_y,
            ).astype(np.float32)
        return _sample_grid_numpy(
            self.footbed_heightmap,
            self.footbed_x_centers,
            self.footbed_z_centers,
            query_x,
            query_z,
        )

    def sample_footbed_mask_numpy(self, query_x: np.ndarray, query_z: np.ndarray) -> np.ndarray:
        if self.footbed_mask is None or self.footbed_x_centers is None or self.footbed_z_centers is None:
            return np.ones_like(np.asarray(query_x, dtype=np.float32))
        return _sample_grid_numpy(
            self.footbed_mask.astype(np.float32),
            self.footbed_x_centers,
            self.footbed_z_centers,
            query_x,
            query_z,
        )


def _sample_grid_numpy(
    grid: np.ndarray,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    query_x: np.ndarray,
    query_z: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample an X-by-Z grid at shoe-frame X/Z coordinates."""

    values = np.asarray(grid, dtype=np.float32)
    xs = np.asarray(x_centers, dtype=np.float32)
    zs = np.asarray(z_centers, dtype=np.float32)
    qx = np.asarray(query_x, dtype=np.float32)
    qz = np.asarray(query_z, dtype=np.float32)
    original_shape = np.broadcast_shapes(qx.shape, qz.shape)
    qx = np.broadcast_to(qx, original_shape).reshape(-1)
    qz = np.broadcast_to(qz, original_shape).reshape(-1)

    qx = np.clip(qx, float(xs[0]), float(xs[-1]))
    qz = np.clip(qz, float(zs[0]), float(zs[-1]))
    ix1 = np.searchsorted(xs, qx, side="left")
    iz1 = np.searchsorted(zs, qz, side="left")
    ix1 = np.clip(ix1, 1, xs.size - 1)
    iz1 = np.clip(iz1, 1, zs.size - 1)
    ix0 = ix1 - 1
    iz0 = iz1 - 1
    x0 = xs[ix0]
    x1 = xs[ix1]
    z0 = zs[iz0]
    z1 = zs[iz1]
    tx = (qx - x0) / np.maximum(x1 - x0, 1e-8)
    tz = (qz - z0) / np.maximum(z1 - z0, 1e-8)

    v00 = values[ix0, iz0]
    v10 = values[ix1, iz0]
    v01 = values[ix0, iz1]
    v11 = values[ix1, iz1]
    sampled = (
        (1.0 - tx) * (1.0 - tz) * v00
        + tx * (1.0 - tz) * v10
        + (1.0 - tx) * tz * v01
        + tx * tz * v11
    )
    return sampled.reshape(original_shape).astype(np.float32)


@dataclass(frozen=True)
class FootFitResult:
    """Optimized alignment and diagnostic values."""

    alignment: FootAlignment
    baseline_alignment: FootAlignment
    aligned_vertices: np.ndarray
    baseline_vertices: np.ndarray
    cavity: PseudoCavity
    metrics: Dict[str, object]


class _TorchCavity:
    def __init__(self, cavity: PseudoCavity, device: torch.device, dtype: torch.dtype) -> None:
        self.cavity = cavity
        self.x = torch.as_tensor(cavity.centerline_x, dtype=dtype, device=device)
        self.center_z = torch.as_tensor(cavity.centerline_z, dtype=dtype, device=device)
        self.left_z = torch.as_tensor(cavity.left_boundary_z, dtype=dtype, device=device)
        self.right_z = torch.as_tensor(cavity.right_boundary_z, dtype=dtype, device=device)
        self.floor_y = torch.as_tensor(cavity.floor_y, dtype=dtype, device=device)
        self.footbed_y = torch.as_tensor(cavity.footbed_y, dtype=dtype, device=device)
        self.support_length = torch.tensor(float(cavity.support_length), dtype=dtype, device=device)
        self.x_min = torch.tensor(cavity.x_min, dtype=dtype, device=device)
        self.x_max = torch.tensor(cavity.x_max, dtype=dtype, device=device)
        self.top_y = torch.tensor(float(cavity.top_y), dtype=dtype, device=device)
        self.has_heightmap = (
            cavity.footbed_heightmap is not None
            and cavity.footbed_x_centers is not None
            and cavity.footbed_z_centers is not None
        )
        if self.has_heightmap:
            self.footbed_grid = torch.as_tensor(
                cavity.footbed_heightmap[None, None, :, :],
                dtype=dtype,
                device=device,
            )
            mask = (
                np.ones_like(cavity.footbed_heightmap, dtype=np.float32)
                if cavity.footbed_mask is None
                else cavity.footbed_mask.astype(np.float32)
            )
            self.footbed_mask_grid = torch.as_tensor(mask[None, None, :, :], dtype=dtype, device=device)
            self.grid_x_min = torch.tensor(float(cavity.footbed_x_centers[0]), dtype=dtype, device=device)
            self.grid_x_max = torch.tensor(float(cavity.footbed_x_centers[-1]), dtype=dtype, device=device)
            self.grid_z_min = torch.tensor(float(cavity.footbed_z_centers[0]), dtype=dtype, device=device)
            self.grid_z_max = torch.tensor(float(cavity.footbed_z_centers[-1]), dtype=dtype, device=device)
        else:
            self.footbed_grid = None
            self.footbed_mask_grid = None
            self.grid_x_min = self.x_min
            self.grid_x_max = self.x_max
            self.grid_z_min = torch.tensor(float(cavity.left_boundary_z.min()), dtype=dtype, device=device)
            self.grid_z_max = torch.tensor(float(cavity.right_boundary_z.max()), dtype=dtype, device=device)

    def interp(self, values: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        query = torch.clamp(query_x, float(self.cavity.x_min), float(self.cavity.x_max))
        idx = torch.searchsorted(self.x, query, right=False)
        idx1 = torch.clamp(idx, 1, self.x.shape[0] - 1)
        idx0 = idx1 - 1
        x0 = self.x[idx0]
        x1 = self.x[idx1]
        y0 = values[idx0]
        y1 = values[idx1]
        t = (query - x0) / torch.clamp(x1 - x0, min=1e-8)
        return y0 + t * (y1 - y0)

    def center(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.interp(self.center_z, query_x)

    def left(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.interp(self.left_z, query_x)

    def right(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.interp(self.right_z, query_x)

    def floor(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.interp(self.floor_y, query_x)

    def footbed(self, query_x: torch.Tensor, query_z: torch.Tensor) -> torch.Tensor:
        if not self.has_heightmap or self.footbed_grid is None:
            return self.interp(self.footbed_y, query_x)
        return self._sample_grid(self.footbed_grid, query_x, query_z, padding_mode="border")

    def footbed_mask_value(self, query_x: torch.Tensor, query_z: torch.Tensor) -> torch.Tensor:
        if not self.has_heightmap or self.footbed_mask_grid is None:
            return torch.ones_like(query_x)
        return self._sample_grid(self.footbed_mask_grid, query_x, query_z, padding_mode="zeros")

    def _sample_grid(
        self,
        grid_values: torch.Tensor,
        query_x: torch.Tensor,
        query_z: torch.Tensor,
        *,
        padding_mode: str,
    ) -> torch.Tensor:
        x_norm = 2.0 * (query_x - self.grid_x_min) / torch.clamp(self.grid_x_max - self.grid_x_min, min=1e-8) - 1.0
        z_norm = 2.0 * (query_z - self.grid_z_min) / torch.clamp(self.grid_z_max - self.grid_z_min, min=1e-8) - 1.0
        grid = torch.stack([z_norm, x_norm], dim=-1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            grid_values,
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        return sampled.reshape(-1)


class _BoundedParameters(torch.nn.Module):
    def __init__(
        self,
        cfg: FootFitOptimizerConfig,
        support_length: float,
        offset_min: float,
        offset_max: float,
        initial: Dict[str, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.support_length = float(support_length)
        self.offset_min = float(offset_min)
        self.offset_max = float(offset_max)

        def raw_tanh(value: float, bound: float) -> torch.Tensor:
            if bound <= 0.0:
                return torch.tensor(0.0, dtype=dtype, device=device)
            normalized = float(np.clip(value / bound, -0.98, 0.98))
            return torch.tensor(np.arctanh(normalized), dtype=dtype, device=device)

        def raw_sigmoid(value: float, low: float, high: float) -> torch.Tensor:
            if high <= low:
                return torch.tensor(0.0, dtype=dtype, device=device)
            normalized = float(np.clip((value - low) / (high - low), 1e-4, 1.0 - 1e-4))
            return torch.tensor(np.log(normalized / (1.0 - normalized)), dtype=dtype, device=device)

        self.raw_log_scale = torch.nn.Parameter(
            raw_tanh(float(initial.get("log_scale_delta", 0.0)), cfg.max_log_scale_delta)
        )
        self.raw_yaw = torch.nn.Parameter(
            raw_tanh(float(initial.get("yaw_degrees", 0.0)), cfg.max_yaw_degrees)
        )
        self.raw_pitch = torch.nn.Parameter(
            raw_tanh(float(initial.get("pitch_degrees", 0.0)), cfg.max_pitch_degrees)
        )
        self.raw_roll = torch.nn.Parameter(
            raw_tanh(float(initial.get("roll_degrees", 0.0)), cfg.max_roll_degrees)
        )
        self.raw_tx = torch.nn.Parameter(
            raw_tanh(float(initial.get("tx", 0.0)), cfg.max_translation_x_ratio * self.support_length)
        )
        self.raw_ty = torch.nn.Parameter(
            raw_tanh(float(initial.get("ty", 0.0)), cfg.max_translation_y_ratio * self.support_length)
        )
        self.raw_tz = torch.nn.Parameter(
            raw_tanh(float(initial.get("tz", 0.0)), cfg.max_translation_z_ratio * self.support_length)
        )
        self.raw_offset = torch.nn.Parameter(
            raw_sigmoid(float(initial.get("footbed_offset", 0.0)), self.offset_min, self.offset_max)
        )

    def values(self) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        length = self.support_length
        log_scale_delta = cfg.max_log_scale_delta * torch.tanh(self.raw_log_scale)
        return {
            "scale_delta": torch.exp(log_scale_delta),
            "log_scale_delta": log_scale_delta,
            "yaw_degrees": cfg.max_yaw_degrees * torch.tanh(self.raw_yaw),
            "pitch_degrees": cfg.max_pitch_degrees * torch.tanh(self.raw_pitch),
            "roll_degrees": cfg.max_roll_degrees * torch.tanh(self.raw_roll),
            "tx": cfg.max_translation_x_ratio * length * torch.tanh(self.raw_tx),
            "ty": cfg.max_translation_y_ratio * length * torch.tanh(self.raw_ty),
            "tz": cfg.max_translation_z_ratio * length * torch.tanh(self.raw_tz),
            "footbed_offset": self.offset_min + (self.offset_max - self.offset_min) * torch.sigmoid(self.raw_offset),
        }


def load_pseudo_cavity_from_support_json(
    support_json_path: str | Path,
    shoe_mesh: MeshData,
    footbed_npz_path: str | Path | None = None,
    use_sidecar_npz: bool = True,
    config: Optional[PseudoCavityConfig] = None,
) -> PseudoCavity:
    """Read ``support_footprint.json`` and build a pseudo-cavity."""

    cfg = config or PseudoCavityConfig()
    support_path = Path(support_json_path)
    with support_path.open("r") as f:
        payload = json.load(f)

    required = [
        "centerline_x",
        "centerline_z",
        "left_boundary_z",
        "right_boundary_z",
        "support_axis_profile",
        "length_extent",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing support-footprint fields in {support_json_path}: {missing}")

    centerline_x = np.asarray(payload["centerline_x"], dtype=np.float32)
    order = np.argsort(centerline_x)
    centerline_x = centerline_x[order]
    centerline_z = np.asarray(payload["centerline_z"], dtype=np.float32)[order]
    left_boundary_z = np.asarray(payload["left_boundary_z"], dtype=np.float32)[order]
    right_boundary_z = np.asarray(payload["right_boundary_z"], dtype=np.float32)[order]
    floor_y = np.asarray(payload["support_axis_profile"], dtype=np.float32)[order]
    footbed_y = np.asarray(payload.get("smooth_footbed_axis_profile", payload.get("footbed_axis_profile", floor_y)), dtype=np.float32)
    footbed_y = footbed_y[order]

    if centerline_x.size < 4:
        raise ValueError(f"Support profile is too short in {support_json_path}")

    footbed_x_centers = None
    footbed_z_centers = None
    footbed_mask = None
    footbed_heightmap = None
    footbed_source = "support_axis_profile"
    npz_path = (
        Path(footbed_npz_path)
        if footbed_npz_path is not None
        else support_path.parent / "pseudo_footbed_heightmap.npz"
        if use_sidecar_npz
        else None
    )
    if npz_path is not None and npz_path.exists():
        footbed_payload = _load_footbed_npz(npz_path)
        footbed_x_centers = footbed_payload["x_centers"]
        footbed_z_centers = footbed_payload["z_centers"]
        footbed_mask = footbed_payload["footbed_mask"]
        footbed_heightmap = footbed_payload["smooth_footbed_heightmap"]
        footbed_source = str(footbed_payload["smooth_footbed_source"])
        if "smooth_footbed_axis_profile" in footbed_payload:
            footbed_y = np.asarray(footbed_payload["smooth_footbed_axis_profile"], dtype=np.float32)
            if footbed_y.shape[0] == centerline_x.shape[0]:
                footbed_y = footbed_y[order]
            else:
                footbed_y = np.interp(centerline_x, footbed_payload["centerline_x"], footbed_y).astype(np.float32)

    vertices = np.asarray(shoe_mesh.vertices, dtype=np.float32)
    robust_opening_y = float(np.percentile(vertices[:, cfg.shoe_up_axis], cfg.top_percentile))
    support_length = float(payload["length_extent"])
    top_y = robust_opening_y - cfg.top_margin_ratio * max(support_length, 1e-6)
    sole_yaw = _sole_centerline_yaw_degrees(centerline_x, centerline_z)

    return PseudoCavity(
        centerline_x=centerline_x,
        centerline_z=centerline_z,
        left_boundary_z=np.minimum(left_boundary_z, right_boundary_z),
        right_boundary_z=np.maximum(left_boundary_z, right_boundary_z),
        floor_y=floor_y,
        footbed_y=footbed_y,
        support_length=support_length,
        top_y=top_y,
        config=cfg,
        footbed_x_centers=footbed_x_centers,
        footbed_z_centers=footbed_z_centers,
        footbed_mask=footbed_mask,
        footbed_heightmap=footbed_heightmap,
        footbed_source=footbed_source,
        sole_yaw_degrees=sole_yaw,
        source_path=str(support_path),
    )


def _load_footbed_npz(path: Path) -> Dict[str, object]:
    with np.load(path) as payload:
        required = ["x_centers", "z_centers", "footbed_mask", "smooth_footbed_heightmap"]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"Missing footbed heightmap fields in {path}: {missing}")
        x_centers = np.asarray(payload["x_centers"], dtype=np.float32)
        z_centers = np.asarray(payload["z_centers"], dtype=np.float32)
        mask = np.asarray(payload["footbed_mask"], dtype=bool)
        heightmap = np.asarray(payload["smooth_footbed_heightmap"], dtype=np.float32)
        profile = np.asarray(payload["smooth_footbed_axis_profile"], dtype=np.float32) if "smooth_footbed_axis_profile" in payload.files else None
        centerline_x = np.asarray(payload["centerline_x"], dtype=np.float32) if "centerline_x" in payload.files else x_centers
        source = str(payload["smooth_footbed_source"].item()) if "smooth_footbed_source" in payload.files else "smooth_footbed_heightmap"

    filled = _fill_footbed_heightmap(heightmap, mask, x_centers, centerline_x, profile)
    output: Dict[str, object] = {
        "x_centers": x_centers,
        "z_centers": z_centers,
        "footbed_mask": mask,
        "smooth_footbed_heightmap": filled,
        "centerline_x": centerline_x,
        "smooth_footbed_source": source,
    }
    if profile is not None:
        output["smooth_footbed_axis_profile"] = profile
    return output


def _fill_footbed_heightmap(
    heightmap: np.ndarray,
    mask: np.ndarray,
    x_centers: np.ndarray,
    centerline_x: np.ndarray,
    profile: Optional[np.ndarray],
) -> np.ndarray:
    values = np.asarray(heightmap, dtype=np.float32).copy()
    valid = np.isfinite(values) & np.asarray(mask, dtype=bool)
    if profile is not None and profile.size >= 2:
        fallback = np.interp(x_centers, centerline_x, np.asarray(profile, dtype=np.float32)).astype(np.float32)
    elif np.any(valid):
        fallback = np.full((x_centers.size,), float(np.nanmedian(values[valid])), dtype=np.float32)
    else:
        fallback = np.zeros((x_centers.size,), dtype=np.float32)
    fallback_grid = np.repeat(fallback[:, None], values.shape[1], axis=1)
    values[~np.isfinite(values)] = fallback_grid[~np.isfinite(values)]
    return values.astype(np.float32)


def _sole_centerline_yaw_degrees(centerline_x: np.ndarray, centerline_z: np.ndarray) -> float:
    x = np.asarray(centerline_x, dtype=np.float32)
    z = np.asarray(centerline_z, dtype=np.float32)
    if x.size < 4:
        return 0.0
    frac = (x - float(x.min())) / max(float(x.max() - x.min()), 1e-8)
    heel_mask = (frac >= 0.08) & (frac <= 0.28)
    ball_mask = (frac >= 0.55) & (frac <= 0.78)
    if int(heel_mask.sum()) < 2:
        heel_mask = frac <= 0.30
    if int(ball_mask.sum()) < 2:
        ball_mask = (frac >= 0.50) & (frac <= 0.85)
    if int(heel_mask.sum()) < 1 or int(ball_mask.sum()) < 1:
        return float(np.rad2deg(np.arctan2(z[-1] - z[0], x[-1] - x[0])))
    heel = np.asarray([float(x[heel_mask].mean()), float(z[heel_mask].mean())], dtype=np.float32)
    ball = np.asarray([float(x[ball_mask].mean()), float(z[ball_mask].mean())], dtype=np.float32)
    delta = ball - heel
    return float(np.rad2deg(np.arctan2(delta[1], delta[0])))


def _effective_config_for_shoe(
    cfg: FootFitOptimizerConfig,
    shoe_mesh: MeshData,
    cavity: PseudoCavity,
) -> Tuple[FootFitOptimizerConfig, Dict[str, object]]:
    _, _, shoe_size, _ = mesh_bounds(shoe_mesh.vertices)
    support_length = max(float(cavity.support_length), 1e-8)
    height_ratio = float(shoe_size[cavity.config.shoe_up_axis] / support_length)
    requested_style = str(cfg.style_mode).lower()
    if requested_style not in {"auto", "normal", "boot"}:
        raise ValueError("style_mode must be 'auto', 'normal', or 'boot'")
    style = "boot" if requested_style == "auto" and height_ratio >= cfg.boot_height_ratio_threshold else requested_style
    if style == "auto":
        style = "normal"

    if style == "boot":
        effective = replace(
            cfg,
            max_translation_x_ratio=0.080,
            max_translation_y_ratio=0.120,
            max_translation_z_ratio=0.120,
            footbed_offset_init_ratio=0.0,
            footbed_offset_min_ratio=0.0,
            footbed_offset_max_ratio=0.0,
            cavity_weight=22.0,
            plantar_weight=95.0,
            gap_weight=45.0,
            axis_weight=10.0,
            footbed_mask_weight=14.0,
            width_slack_weight=16.0,
            prior_weight=5.0,
            top_weight=0.05,
            sole_yaw_prior_blend=0.35,
            yaw_prior_factor=0.40,
            translation_prior_factor=0.18,
            multistart_yaw_degrees=(-5.0, 0.0, 5.0),
            multistart_scale_delta=(-0.035, 0.0, 0.035),
            multistart_z_ratio=(-0.025, 0.0, 0.025),
            multistart_offset_ratio_delta=(0.0,),
        )
    else:
        effective = replace(
            cfg,
            style_mode=style,
            max_translation_x_ratio=0.100,
            max_translation_y_ratio=0.120,
            max_translation_z_ratio=0.120,
            footbed_offset_init_ratio=0.0,
            footbed_offset_min_ratio=0.0,
            footbed_offset_max_ratio=0.0,
            cavity_weight=90.0,
            plantar_weight=135.0,
            gap_weight=70.0,
            axis_weight=35.0,
            footbed_mask_weight=0.0,
            width_slack_weight=20.0,
            side_balance_weight=0.0,
            side_total_clearance_weight=0.0,
            length_balance_weight=0.0,
            prior_weight=10.0,
            top_weight=4.0,
            sole_yaw_prior_blend=0.20,
            yaw_prior_factor=0.30,
            translation_prior_factor=0.0,
            offset_prior_factor=0.20,
            multistart_yaw_degrees=(-5.0, 0.0, 5.0),
            multistart_scale_delta=(-0.035, 0.0, 0.035),
            multistart_z_ratio=(-0.025, 0.0, 0.025),
            multistart_offset_ratio_delta=(0.0,),
        )

    style_info = {
        "mode": style,
        "requested_mode": requested_style,
        "height_ratio": height_ratio,
        "height": float(shoe_size[cavity.config.shoe_up_axis]),
        "support_length": support_length,
        "boot_height_ratio_threshold": float(cfg.boot_height_ratio_threshold),
        "multistart_count": int(
            len(effective.multistart_yaw_degrees)
            * len(effective.multistart_scale_delta)
            * len(effective.multistart_z_ratio)
            * len(effective.multistart_offset_ratio_delta)
        ),
    }
    return effective, style_info


def optimize_foot_fit(
    foot_mesh: MeshData,
    shoe_mesh: MeshData,
    baseline_alignment: FootAlignment,
    cavity: PseudoCavity,
    config: Optional[FootFitOptimizerConfig] = None,
) -> FootFitResult:
    """Optimize a baseline foot alignment inside a pseudo-cavity."""

    requested_cfg = config or FootFitOptimizerConfig()
    cfg, style_info = _effective_config_for_shoe(requested_cfg, shoe_mesh, cavity)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    tcavity = _TorchCavity(cavity, device, dtype)

    prepared = _prepare_foot_samples(foot_mesh, baseline_alignment, cfg)
    baseline_points = torch.as_tensor(prepared["baseline_points"], dtype=dtype, device=device)
    baseline_fit = torch.as_tensor(prepared["baseline_fit_points"], dtype=dtype, device=device)
    baseline_plantar = torch.as_tensor(prepared["baseline_plantar_points"], dtype=dtype, device=device)
    baseline_axis = torch.as_tensor(prepared["baseline_axis_points"], dtype=dtype, device=device)
    plantar_region = torch.as_tensor(prepared["plantar_region"], dtype=torch.int64, device=device)
    anchor = torch.as_tensor(prepared["anchor"], dtype=dtype, device=device)

    length = max(float(cavity.support_length), 1e-6)
    offset_init = cfg.footbed_offset_init_ratio * length
    offset_min = cfg.footbed_offset_min_ratio * length
    offset_max = cfg.footbed_offset_max_ratio * length
    baseline_yaw = float(baseline_alignment.auto_yaw_degrees + baseline_alignment.config.yaw_degrees)
    sole_delta_yaw = _wrap_degrees(float(cavity.sole_yaw_degrees) - baseline_yaw)
    sole_delta_yaw = float(np.clip(sole_delta_yaw, -cfg.max_yaw_degrees, cfg.max_yaw_degrees))
    yaw_prior_value = float(cfg.sole_yaw_prior_blend * sole_delta_yaw)
    yaw_prior = torch.tensor(yaw_prior_value, dtype=dtype, device=device)
    baseline_priors = {
        "log_scale_delta": torch.zeros((), dtype=dtype, device=device),
        "yaw_degrees": torch.zeros((), dtype=dtype, device=device),
        "pitch_degrees": torch.zeros((), dtype=dtype, device=device),
        "roll_degrees": torch.zeros((), dtype=dtype, device=device),
        "tx": torch.zeros((), dtype=dtype, device=device),
        "ty": torch.zeros((), dtype=dtype, device=device),
        "tz": torch.zeros((), dtype=dtype, device=device),
        "footbed_offset": torch.tensor(offset_init, dtype=dtype, device=device),
    }

    initial_metrics = _evaluate_candidate(
        baseline_points,
        baseline_fit,
        baseline_plantar,
        baseline_axis,
        plantar_region,
        tcavity,
        torch.tensor(offset_init, dtype=dtype, device=device),
        cfg,
        priors=baseline_priors,
        yaw_prior_degrees=yaw_prior,
    )

    starts = _make_multistart_initials(cfg, length, offset_init, offset_min, offset_max, yaw_prior_value)
    best: Optional[Dict[str, object]] = None
    for start_index, start in enumerate(starts):
        params = _BoundedParameters(cfg, length, offset_min, offset_max, start, device, dtype).to(device)
        initial_state = _parameter_values_as_float(params.values())
        history = _run_one_start(
            params,
            baseline_fit,
            baseline_plantar,
            baseline_axis,
            plantar_region,
            anchor,
            tcavity,
            cfg,
            yaw_prior,
        )
        values = params.values()
        final_fit = _apply_delta_transform(baseline_fit, anchor, values)
        final_plantar = _apply_delta_transform(baseline_plantar, anchor, values)
        final_axis = _apply_delta_transform(baseline_axis, anchor, values)
        final_metrics = _objective(
            final_fit,
            final_plantar,
            final_axis,
            plantar_region,
            tcavity,
            values["footbed_offset"],
            cfg,
            values,
            yaw_prior,
        )
        total = float(final_metrics["total"].detach().cpu())
        record = {
            "start_index": start_index,
            "initial": initial_state,
            "final_params": _parameter_values_as_float(values),
            "loss": total,
            "loss_components": _loss_dict_as_float(final_metrics),
            "history": history,
        }
        if best is None or total < float(best["loss"]):
            best = record

    if best is None:
        raise RuntimeError("No optimization starts were run")

    best_values_np = best["final_params"]
    best_values_torch = _values_to_torch(best_values_np, device, dtype)
    optimized_points = _apply_delta_transform(baseline_points, anchor, best_values_torch)
    optimized_points_np = optimized_points.detach().cpu().numpy().astype(np.float32)
    delta_matrix = _delta_matrix_from_values(
        prepared["anchor"],
        best_values_np,
    )
    optimized_foot_to_shoe = (delta_matrix @ baseline_alignment.foot_to_shoe).astype(np.float32)
    optimized_shoe_to_foot = np.linalg.inv(optimized_foot_to_shoe).astype(np.float32)

    final_eval = _evaluate_candidate(
        optimized_points,
        _apply_delta_transform(baseline_fit, anchor, best_values_torch),
        _apply_delta_transform(baseline_plantar, anchor, best_values_torch),
        _apply_delta_transform(baseline_axis, anchor, best_values_torch),
        plantar_region,
        tcavity,
        best_values_torch["footbed_offset"],
        cfg,
        priors=best_values_torch,
        yaw_prior_degrees=yaw_prior,
    )

    optimized_alignment = FootAlignment(
        foot_to_shoe=optimized_foot_to_shoe,
        shoe_to_foot=optimized_shoe_to_foot,
        scale=float(baseline_alignment.scale * best_values_np["scale_delta"]),
        plantar_z=plantar_coordinate(
            optimized_points_np,
            baseline_alignment.config.shoe_up_axis,
            baseline_alignment.config.shoe_up_sign,
        ),
        auto_yaw_degrees=baseline_alignment.auto_yaw_degrees,
        config=baseline_alignment.config,
        foot_anchor_remapped=baseline_alignment.foot_anchor_remapped,
        shoe_anchor=tuple(
            float(v)
            for v in transform_points(
                np.asarray([baseline_alignment.shoe_anchor], dtype=np.float32),
                delta_matrix,
            )[0]
        ),
        ankle_center_shoe=None
        if baseline_alignment.ankle_center_shoe is None
        else tuple(
            float(v)
            for v in transform_points(
                np.asarray([baseline_alignment.ankle_center_shoe], dtype=np.float32),
                delta_matrix,
            )[0]
        ),
        opening_center=baseline_alignment.opening_center,
        opening_component_index=baseline_alignment.opening_component_index,
        opening_component_size=baseline_alignment.opening_component_size,
    )

    transform_det = float(np.linalg.det(optimized_foot_to_shoe[:3, :3]))
    loss_initial = _loss_dict_as_float(initial_metrics)
    loss_final = _loss_dict_as_float(final_eval)
    metrics: Dict[str, object] = {
        "status": "ok",
        "optimizer_config": asdict(cfg),
        "requested_optimizer_config": asdict(requested_cfg),
        "shoe_style": style_info,
        "cavity_config": asdict(cavity.config),
        "baseline_loss": loss_initial,
        "optimized_loss": loss_final,
        "loss_improved": float(loss_final["total"]) < float(loss_initial["total"]),
        "selected_start": best,
        "optimized_params": best_values_np,
        "offset_bounds": {
            "min": offset_min,
            "init": offset_init,
            "max": offset_max,
        },
        "support_length": float(cavity.support_length),
        "sole_yaw": {
            "degrees": float(cavity.sole_yaw_degrees),
            "baseline_yaw_degrees": baseline_yaw,
            "initial_delta_degrees": sole_delta_yaw,
            "prior_delta_degrees": yaw_prior_value,
        },
        "footbed": {
            "source": cavity.footbed_source,
            "has_heightmap": bool(cavity.footbed_heightmap is not None),
        },
        "transform": {
            "determinant": transform_det,
            "invertible": bool(np.isfinite(optimized_shoe_to_foot).all() and abs(transform_det) > 1e-10),
            "no_nan": bool(np.isfinite(optimized_foot_to_shoe).all() and np.isfinite(optimized_points_np).all()),
        },
        "bounds": {
            "baseline_foot": _bounds_dict(prepared["baseline_points"]),
            "optimized_foot": _bounds_dict(optimized_points_np),
            "shoe": _bounds_dict(shoe_mesh.vertices),
        },
    }

    return FootFitResult(
        alignment=optimized_alignment,
        baseline_alignment=baseline_alignment,
        aligned_vertices=optimized_points_np,
        baseline_vertices=prepared["baseline_points"].astype(np.float32),
        cavity=cavity,
        metrics=metrics,
    )


def evaluate_fit_numpy(
    vertices_shoe: np.ndarray,
    cavity: PseudoCavity,
    footbed_offset: float,
    config: Optional[FootFitOptimizerConfig] = None,
) -> Dict[str, float]:
    """Compute simple cavity metrics for already transformed foot vertices."""

    cfg = config or FootFitOptimizerConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    tcavity = _TorchCavity(cavity, device, dtype)
    points = torch.as_tensor(vertices_shoe, dtype=dtype, device=device)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    left = tcavity.left(x) + cfg.wall_clearance
    right = tcavity.right(x) - cfg.wall_clearance
    footbed = tcavity.footbed(x, z) + axis_sign(cavity.config.shoe_up_sign) * float(footbed_offset)
    violations = torch.stack(
        [
            F.relu(tcavity.x_min - x),
            F.relu(x - tcavity.x_max),
            F.relu(left - z),
            F.relu(z - right),
            F.relu(y - footbed),
        ],
        dim=1,
    )
    max_violation = violations.max(dim=1).values
    return {
        "cavity_violation_fraction": float((max_violation > 1e-5).float().mean().cpu()),
        "cavity_violation_mean": float(max_violation.mean().cpu()),
        "cavity_violation_max": float(max_violation.max().cpu()),
    }


def _prepare_foot_samples(
    foot_mesh: MeshData,
    baseline_alignment: FootAlignment,
    cfg: FootFitOptimizerConfig,
) -> Dict[str, np.ndarray]:
    foot_vertices = np.asarray(foot_mesh.vertices, dtype=np.float32)
    align_cfg = baseline_alignment.config
    remapped = remap_supr_to_shoe_axes(
        foot_vertices,
        align_cfg.shoe_length_axis,
        align_cfg.shoe_up_axis,
        align_cfg.shoe_width_axis,
        align_cfg.shoe_length_sign,
        align_cfg.shoe_up_sign,
        align_cfg.shoe_width_sign,
    )
    baseline_points = baseline_alignment.transform_foot_to_shoe(foot_vertices).astype(np.float32)

    x = remapped[:, align_cfg.shoe_length_axis]
    y = remapped[:, align_cfg.shoe_up_axis]
    signed_up = axis_sign(align_cfg.shoe_up_sign) * y
    signed_min = float(np.min(signed_up))
    signed_max = float(np.max(signed_up))
    height_extent = max(signed_max - signed_min, 1e-8)
    height_fraction = (signed_up - signed_min) / height_extent
    fit_mask = height_fraction <= cfg.fit_height_fraction

    length_extent = max(float(np.max(x) - np.min(x)), 1e-8)
    plantar_y = plantar_coordinate(remapped, align_cfg.shoe_up_axis, align_cfg.shoe_up_sign)
    plantar_band = cfg.plantar_band_ratio * length_extent
    if axis_sign(align_cfg.shoe_up_sign) < 0.0:
        plantar_mask = y >= plantar_y - plantar_band
    else:
        plantar_mask = y <= plantar_y + plantar_band
    plantar_mask &= fit_mask

    if int(plantar_mask.sum()) < 20:
        threshold = np.percentile(y, 90.0) if axis_sign(align_cfg.shoe_up_sign) < 0.0 else np.percentile(y, 10.0)
        plantar_mask = y >= threshold if axis_sign(align_cfg.shoe_up_sign) < 0.0 else y <= threshold

    fit_indices = np.flatnonzero(fit_mask)
    plantar_indices = np.flatnonzero(plantar_mask)
    if fit_indices.size == 0 or plantar_indices.size == 0:
        raise ValueError("Unable to select enough SUPR foot samples for fitting")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    frac = (x - x_min) / max(x_max - x_min, 1e-8)
    plantar_region = np.zeros((plantar_indices.size,), dtype=np.int64)
    plantar_frac = frac[plantar_indices]
    plantar_region[(plantar_frac >= 0.25) & (plantar_frac <= 0.58)] = 1
    plantar_region[(plantar_frac >= 0.58) & (plantar_frac <= 0.80)] = 2
    plantar_region[plantar_frac > 0.80] = 3

    axis_raw_points = _build_foot_axis_points(foot_vertices, remapped, fit_mask, cfg.foot_axis_slice_count, align_cfg)
    baseline_axis_points = baseline_alignment.transform_foot_to_shoe(axis_raw_points).astype(np.float32)
    anchor = np.mean(baseline_points[fit_indices], axis=0).astype(np.float32)

    return {
        "baseline_points": baseline_points,
        "baseline_fit_points": baseline_points[fit_indices],
        "baseline_plantar_points": baseline_points[plantar_indices],
        "baseline_axis_points": baseline_axis_points,
        "plantar_region": plantar_region,
        "anchor": anchor,
        "fit_indices": fit_indices.astype(np.int64),
        "plantar_indices": plantar_indices.astype(np.int64),
    }


def _build_foot_axis_points(
    foot_vertices: np.ndarray,
    remapped: np.ndarray,
    fit_mask: np.ndarray,
    slice_count: int,
    config: FootAlignmentConfig,
) -> np.ndarray:
    x = remapped[:, config.shoe_length_axis]
    x_min = float(np.min(x[fit_mask]))
    x_max = float(np.max(x[fit_mask]))
    edges = np.linspace(x_min, x_max, max(slice_count, 4) + 1)
    axis_points = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = fit_mask & (x >= start) & (x <= end)
        if int(mask.sum()) < 3:
            continue
        points = foot_vertices[mask]
        axis_points.append(points.mean(axis=0))
    if len(axis_points) < 4:
        axis_points = [foot_vertices[fit_mask].mean(axis=0)]
    return np.asarray(axis_points, dtype=np.float32)


def _make_multistart_initials(
    cfg: FootFitOptimizerConfig,
    length: float,
    offset_init: float,
    offset_min: float,
    offset_max: float,
    yaw_center_degrees: float,
) -> list[Dict[str, float]]:
    starts: list[Dict[str, float]] = []
    for yaw in cfg.multistart_yaw_degrees:
        for scale_delta in cfg.multistart_scale_delta:
            for z_ratio in cfg.multistart_z_ratio:
                for offset_ratio_delta in cfg.multistart_offset_ratio_delta:
                    starts.append(
                        {
                            "yaw_degrees": float(yaw_center_degrees + yaw),
                            "log_scale_delta": float(scale_delta),
                            "tz": float(z_ratio * length),
                            "footbed_offset": float(
                                np.clip(offset_init + offset_ratio_delta * length, offset_min, offset_max)
                            ),
                        }
                    )
    return starts


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _run_one_start(
    params: _BoundedParameters,
    baseline_fit: torch.Tensor,
    baseline_plantar: torch.Tensor,
    baseline_axis: torch.Tensor,
    plantar_region: torch.Tensor,
    anchor: torch.Tensor,
    cavity: _TorchCavity,
    cfg: FootFitOptimizerConfig,
    yaw_prior_degrees: torch.Tensor,
) -> Dict[str, object]:
    history: Dict[str, object] = {"adam": [], "lbfgs": []}
    stage_plan = _adam_stage_plan(cfg.adam_steps)
    global_step = 0
    for stage_name, stage_steps, trainable_names, lr_scale in stage_plan:
        if stage_steps <= 0:
            continue
        _set_trainable_parameters(params, trainable_names)
        optimizer = torch.optim.Adam(
            [param for param in params.parameters() if param.requires_grad],
            lr=cfg.adam_lr * lr_scale,
        )
        for local_step in range(stage_steps):
            optimizer.zero_grad()
            values = params.values()
            fit = _apply_delta_transform(baseline_fit, anchor, values)
            plantar = _apply_delta_transform(baseline_plantar, anchor, values)
            axis = _apply_delta_transform(baseline_axis, anchor, values)
            losses = _objective(
                fit,
                plantar,
                axis,
                plantar_region,
                cavity,
                values["footbed_offset"],
                cfg,
                values,
                yaw_prior_degrees,
            )
            losses["total"].backward()
            optimizer.step()
            if local_step in (0, stage_steps // 2, stage_steps - 1):
                history["adam"].append(
                    {
                        "stage": stage_name,
                        "step": global_step,
                        "local_step": local_step,
                        **_loss_dict_as_float(losses),
                    }
                )
            global_step += 1

    _set_trainable_parameters(params, {"all"})
    lbfgs = torch.optim.LBFGS(
        params.parameters(),
        max_iter=cfg.lbfgs_steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    call_count = 0

    def closure() -> torch.Tensor:
        nonlocal call_count
        lbfgs.zero_grad()
        values = params.values()
        fit = _apply_delta_transform(baseline_fit, anchor, values)
        plantar = _apply_delta_transform(baseline_plantar, anchor, values)
        axis = _apply_delta_transform(baseline_axis, anchor, values)
        losses = _objective(
            fit,
            plantar,
            axis,
            plantar_region,
            cavity,
            values["footbed_offset"],
            cfg,
            values,
            yaw_prior_degrees,
        )
        losses["total"].backward()
        if call_count in (0, cfg.lbfgs_steps - 1):
            history["lbfgs"].append({"step": call_count, **_loss_dict_as_float(losses)})
        call_count += 1
        return losses["total"]

    try:
        lbfgs.step(closure)
    except RuntimeError:
        # Adam result is still useful for diagnostics; caller keeps the best start.
        history["lbfgs_error"] = "LBFGS failed; kept Adam result"
    return history


def _adam_stage_plan(total_steps: int) -> list[Tuple[str, int, set[str], float]]:
    stage_a = int(round(total_steps * 0.20))
    stage_b = int(round(total_steps * 0.40))
    stage_c = max(total_steps - stage_a - stage_b, 0)
    return [
        ("A_scale_yaw", stage_a, {"raw_log_scale", "raw_yaw"}, 1.00),
        (
            "B_rotation_xz",
            stage_b,
            {"raw_log_scale", "raw_yaw", "raw_pitch", "raw_roll", "raw_tx", "raw_tz"},
            0.75,
        ),
        (
            "C_seating",
            stage_c,
            {"raw_log_scale", "raw_yaw", "raw_pitch", "raw_roll", "raw_tx", "raw_ty", "raw_tz", "raw_offset"},
            0.50,
        ),
    ]


def _set_trainable_parameters(params: _BoundedParameters, names: set[str]) -> None:
    train_all = "all" in names
    for name, parameter in params.named_parameters():
        parameter.requires_grad_(train_all or name in names)


def _objective(
    fit_points: torch.Tensor,
    plantar_points: torch.Tensor,
    axis_points: torch.Tensor,
    plantar_region: torch.Tensor,
    cavity: _TorchCavity,
    footbed_offset: torch.Tensor,
    cfg: FootFitOptimizerConfig,
    priors: Optional[Dict[str, torch.Tensor]],
    yaw_prior_degrees: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    x = fit_points[:, 0]
    y = fit_points[:, 1]
    z = fit_points[:, 2]
    length = cavity.support_length
    x_margin = cavity.cavity.config.x_wall_margin_ratio * length
    left = cavity.left(x) + cfg.wall_clearance
    right = cavity.right(x) - cfg.wall_clearance
    footbed = cavity.footbed(x, z) + axis_sign(cavity.cavity.config.shoe_up_sign) * footbed_offset

    violations = torch.stack(
        [
            F.relu((cavity.x_min + x_margin) - x),
            F.relu(x - (cavity.x_max - x_margin)),
            F.relu(left - z),
            F.relu(z - right),
            F.relu(y - footbed),
        ],
        dim=1,
    )
    cavity_loss = _smooth_square(violations).mean()

    # A very weak top term prevents the optimizer from moving the whole foot far
    # above the shoe, but it should not fight boots or ankle openings.
    top_loss = _smooth_square(F.relu(cavity.top_y - y)).mean()

    px = plantar_points[:, 0]
    py = plantar_points[:, 1]
    pz = plantar_points[:, 2]
    pfootbed = cavity.footbed(px, pz) + axis_sign(cavity.cavity.config.shoe_up_sign) * footbed_offset
    clearance = pfootbed - py
    lower = torch.full_like(clearance, cfg.plantar_contact_lower)
    upper = torch.full_like(clearance, cfg.plantar_contact_upper)
    arch = plantar_region == 1
    toe = plantar_region == 3
    lower = torch.where(arch, torch.full_like(lower, cfg.arch_contact_lower), lower)
    upper = torch.where(arch, torch.full_like(upper, cfg.arch_contact_upper), upper)
    lower = torch.where(toe, torch.full_like(lower, cfg.toe_contact_lower), lower)
    upper = torch.where(toe, torch.full_like(upper, cfg.toe_contact_upper), upper)
    plantar_loss = (_smooth_square(F.relu(lower - clearance)) + _smooth_square(F.relu(clearance - upper))).mean()
    mask_values = cavity.footbed_mask_value(px, pz)
    footbed_mask_loss = _smooth_square(F.relu(cfg.footbed_mask_threshold - mask_values)).mean()

    foot_x_min = torch.min(fit_points[:, 0])
    foot_x_max = torch.max(fit_points[:, 0])
    heel_gap = foot_x_min - cavity.x_min
    toe_gap = cavity.x_max - foot_x_max
    heel_min = cfg.heel_gap_min_ratio * length
    heel_max = cfg.heel_gap_max_ratio * length
    toe_min = cfg.toe_gap_min_ratio * length
    toe_max = cfg.toe_gap_max_ratio * length
    gap_loss = (
        _interval_loss(heel_gap, heel_min, heel_max)
        + _interval_loss(toe_gap, toe_min, toe_max)
    )

    axis_x = axis_points[:, 0]
    axis_z = axis_points[:, 2]
    axis_target = cavity.center(axis_x)
    axis_loss = F.huber_loss(axis_z, axis_target, reduction="mean", delta=0.004)

    ball_mask = plantar_region == 2
    if bool(ball_mask.any()):
        ball_points = plantar_points[ball_mask]
    else:
        ball_points = plantar_points
    bx = ball_points[:, 0]
    bz = ball_points[:, 2]
    ball_left_slack = torch.min(bz - (cavity.left(bx) + cfg.wall_clearance))
    ball_right_slack = torch.min((cavity.right(bx) - cfg.wall_clearance) - bz)
    width_slack_loss = _smooth_square(F.relu(cfg.wall_clearance - ball_left_slack)) + _smooth_square(
        F.relu(cfg.wall_clearance - ball_right_slack)
    )
    side_terms = _side_gap_losses(plantar_points, plantar_region, cavity, cfg)
    length_balance_loss = (heel_gap - toe_gap).square()

    if priors is None:
        prior_loss = torch.zeros((), dtype=fit_points.dtype, device=fit_points.device)
    else:
        offset_target = cfg.footbed_offset_init_ratio * length
        yaw_target = torch.zeros((), dtype=fit_points.dtype, device=fit_points.device)
        if yaw_prior_degrees is not None:
            yaw_target = yaw_prior_degrees.to(dtype=fit_points.dtype, device=fit_points.device)
        offset_span = torch.clamp((cfg.footbed_offset_max_ratio - cfg.footbed_offset_min_ratio) * length, min=1e-6)
        tx_bound = torch.clamp(cfg.max_translation_x_ratio * length, min=1e-6)
        ty_bound = torch.clamp(cfg.max_translation_y_ratio * length, min=1e-6)
        tz_bound = torch.clamp(cfg.max_translation_z_ratio * length, min=1e-6)
        prior_loss = (
            (priors["log_scale_delta"] / max(cfg.max_log_scale_delta, 1e-6)).square()
            + cfg.yaw_prior_factor * ((priors["yaw_degrees"] - yaw_target) / max(cfg.max_yaw_degrees, 1e-6)).square()
            + 0.35 * (priors["pitch_degrees"] / max(cfg.max_pitch_degrees, 1e-6)).square()
            + 0.35 * (priors["roll_degrees"] / max(cfg.max_roll_degrees, 1e-6)).square()
            + cfg.translation_prior_factor * (priors["tx"] / tx_bound).square()
            + cfg.translation_prior_factor * (priors["ty"] / ty_bound).square()
            + cfg.translation_prior_factor * (priors["tz"] / tz_bound).square()
            + cfg.offset_prior_factor * ((priors["footbed_offset"] - offset_target) / offset_span).square()
        )

    total = (
        cfg.cavity_weight * cavity_loss
        + cfg.plantar_weight * plantar_loss
        + cfg.gap_weight * gap_loss
        + cfg.axis_weight * axis_loss
        + cfg.footbed_mask_weight * footbed_mask_loss
        + cfg.width_slack_weight * width_slack_loss
        + cfg.side_balance_weight * side_terms["side_balance"]
        + cfg.side_total_clearance_weight * side_terms["side_total_clearance"]
        + cfg.length_balance_weight * length_balance_loss
        + cfg.top_weight * top_loss
        + cfg.prior_weight * prior_loss
    )
    return {
        "total": total,
        "cavity": cavity_loss,
        "plantar": plantar_loss,
        "gap": gap_loss,
        "axis": axis_loss,
        "footbed_mask": footbed_mask_loss,
        "width_slack": width_slack_loss,
        "side_balance": side_terms["side_balance"],
        "side_total_clearance": side_terms["side_total_clearance"],
        "length_balance": length_balance_loss,
        "top": top_loss,
        "prior": prior_loss,
        "heel_gap": heel_gap.detach(),
        "toe_gap": toe_gap.detach(),
        "heel_toe_gap_delta": (heel_gap - toe_gap).detach(),
        "plantar_clearance_mean": clearance.detach().mean(),
        "plantar_clearance_min": clearance.detach().min(),
        "plantar_clearance_max": clearance.detach().max(),
        "side_gap_abs_mean": side_terms["side_gap_abs_mean"],
        "side_total_clearance_mean": side_terms["side_total_clearance_mean"],
        "side_left_gap_min": side_terms["side_left_gap_min"],
        "side_right_gap_min": side_terms["side_right_gap_min"],
        "side_region_count": side_terms["side_region_count"],
        "cavity_violation_fraction": (violations.max(dim=1).values.detach() > 1e-5).float().mean(),
        "cavity_violation_max": violations.max().detach(),
        "footbed_mask_mean": mask_values.detach().mean(),
        "footbed_mask_violation_fraction": (mask_values.detach() < cfg.footbed_mask_threshold).float().mean(),
    }


def _side_gap_losses(
    plantar_points: torch.Tensor,
    plantar_region: torch.Tensor,
    cavity: _TorchCavity,
    cfg: FootFitOptimizerConfig,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=plantar_points.dtype, device=plantar_points.device)
    empty = {
        "side_balance": zero,
        "side_total_clearance": zero,
        "side_gap_abs_mean": zero,
        "side_total_clearance_mean": zero,
        "side_left_gap_min": zero,
        "side_right_gap_min": zero,
        "side_region_count": zero,
    }
    if plantar_points.numel() == 0:
        return empty

    q = float(np.clip(cfg.side_region_edge_quantile, 0.0, 0.45))
    total_min = cfg.side_total_clearance_min_ratio * cavity.support_length
    total_max = cfg.side_total_clearance_max_ratio * cavity.support_length
    balance_terms = []
    total_terms = []
    abs_gaps = []
    total_clearances = []
    left_gaps = []
    right_gaps = []
    for region_id in (0, 2, 3):
        mask = plantar_region == region_id
        if int(mask.sum().detach().cpu()) < 3:
            continue
        points = plantar_points[mask]
        rx = points[:, 0]
        rz = points[:, 2]
        region_x = rx.mean()
        left_wall = cavity.left(region_x) + cfg.wall_clearance
        right_wall = cavity.right(region_x) - cfg.wall_clearance
        foot_left = torch.quantile(rz, q)
        foot_right = torch.quantile(rz, 1.0 - q)
        left_gap = foot_left - left_wall
        right_gap = right_wall - foot_right
        gap_delta = left_gap - right_gap
        total_gap = left_gap + right_gap
        balance_terms.append(gap_delta.square())
        total_terms.append(_interval_loss(total_gap, total_min, total_max))
        abs_gaps.append(gap_delta.detach().abs())
        total_clearances.append(total_gap.detach())
        left_gaps.append(left_gap.detach())
        right_gaps.append(right_gap.detach())

    if not balance_terms:
        return empty

    return {
        "side_balance": torch.stack(balance_terms).mean(),
        "side_total_clearance": torch.stack(total_terms).mean(),
        "side_gap_abs_mean": torch.stack(abs_gaps).mean(),
        "side_total_clearance_mean": torch.stack(total_clearances).mean(),
        "side_left_gap_min": torch.stack(left_gaps).min(),
        "side_right_gap_min": torch.stack(right_gaps).min(),
        "side_region_count": torch.tensor(
            float(len(balance_terms)),
            dtype=plantar_points.dtype,
            device=plantar_points.device,
        ),
    }


def _evaluate_candidate(
    all_points: torch.Tensor,
    fit_points: torch.Tensor,
    plantar_points: torch.Tensor,
    axis_points: torch.Tensor,
    plantar_region: torch.Tensor,
    cavity: _TorchCavity,
    footbed_offset: torch.Tensor,
    cfg: FootFitOptimizerConfig,
    priors: Optional[Dict[str, torch.Tensor]],
    yaw_prior_degrees: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    del all_points
    return _objective(
        fit_points,
        plantar_points,
        axis_points,
        plantar_region,
        cavity,
        footbed_offset,
        cfg,
        priors,
        yaw_prior_degrees,
    )


def _apply_delta_transform(points: torch.Tensor, anchor: torch.Tensor, values: Dict[str, torch.Tensor]) -> torch.Tensor:
    rotation = _torch_rotation_matrix(values["yaw_degrees"], values["pitch_degrees"], values["roll_degrees"], points)
    translation = torch.stack([values["tx"], values["ty"], values["tz"]])
    centered = (points - anchor) * values["scale_delta"]
    rotated = centered @ rotation.transpose(0, 1)
    return rotated + anchor + translation


def _torch_rotation_matrix(
    yaw_degrees: torch.Tensor,
    pitch_degrees: torch.Tensor,
    roll_degrees: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    dtype = reference.dtype
    device = reference.device
    deg_to_rad = math.pi / 180.0
    yaw = yaw_degrees * deg_to_rad
    pitch = pitch_degrees * deg_to_rad
    roll = roll_degrees * deg_to_rad

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # Axes follow the current shoe convention: yaw around Y, pitch around Z,
    # roll around X.
    ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ]
    ).to(device=device, dtype=dtype)
    rz = torch.stack(
        [
            torch.stack([cp, -sp, torch.zeros_like(cp)]),
            torch.stack([sp, cp, torch.zeros_like(cp)]),
            torch.stack([torch.zeros_like(cp), torch.zeros_like(cp), torch.ones_like(cp)]),
        ]
    ).to(device=device, dtype=dtype)
    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)]),
            torch.stack([torch.zeros_like(cr), cr, -sr]),
            torch.stack([torch.zeros_like(cr), sr, cr]),
        ]
    ).to(device=device, dtype=dtype)
    return ry @ rz @ rx


def _delta_matrix_from_values(anchor: np.ndarray, values: Dict[str, float]) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float32)
    scale = float(values["scale_delta"])
    yaw = math.radians(float(values["yaw_degrees"]))
    pitch = math.radians(float(values["pitch_degrees"]))
    roll = math.radians(float(values["roll_degrees"]))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz = np.asarray([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    linear = (ry @ rz @ rx) * scale

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = linear
    translation = np.asarray([values["tx"], values["ty"], values["tz"]], dtype=np.float32)
    matrix[:3, 3] = anchor + translation - linear @ anchor
    return matrix


def _values_to_torch(values: Dict[str, float], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(float(value), dtype=dtype, device=device) for key, value in values.items()}


def _parameter_values_as_float(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def _loss_dict_as_float(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def _bounds_dict(vertices: np.ndarray) -> Dict[str, object]:
    bounds_min, bounds_max, size, center = mesh_bounds(np.asarray(vertices, dtype=np.float32))
    return {
        "min": bounds_min.astype(float).tolist(),
        "max": bounds_max.astype(float).tolist(),
        "size": size.astype(float).tolist(),
        "center": center.astype(float).tolist(),
    }


def _smooth_square(value: torch.Tensor, beta: float = 0.002) -> torch.Tensor:
    return F.softplus(value / beta).mul(beta).square()


def _interval_loss(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return _smooth_square(F.relu(lower - value)) + _smooth_square(F.relu(value - upper))
