"""SUPR-derived pseudo-last construction.

The pseudo-last is a smooth, watertight interior prior built after foot-aware
alignment. It uses the aligned SUPR foot for anatomical proportions and the
support-footbed artifacts for the shoe-specific footprint and bottom surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .foot_alignment import MeshData, load_triangle_mesh, write_obj_mesh
from .foot_sdf import find_boundary_loops


@dataclass(frozen=True)
class PseudoLastConfig:
    """Settings for the SUPR-derived pseudo-last builder."""

    builder_mode: str = "section_loft"
    n_x: int = 128
    n_theta: int = 128
    eta_s: float = 0.93
    toe_merge_start: float = 0.74
    toe_merge_full: float = 0.88
    toe_allowance_ratio: float = 0.05
    arch_strength: float = 0.75
    arch_center: float = 0.45
    arch_sigma: float = 0.12
    heel_hold_strength: float = 0.05
    heel_hold_power: float = 2.0
    heel_hold_end: float = 0.25
    bottom_fixed_band_ratio: float = 0.045
    bottom_clearance_ratio: float = 0.003
    heel_clearance_ratio: float = 0.012
    midfoot_clearance_ratio: float = 0.020
    forefoot_clearance_ratio: float = 0.040
    vamp_clearance_ratio: float = 0.05
    toe_vertical_clearance_ratio: float = 0.04
    top_clearance_ratio: float = 0.012
    support_z_margin_ratio: float = 0.004
    toe_box_top_blend: float = 0.65
    toe_box_side_blend: float = 0.35
    superellipse_p: float = 3.5
    superellipse_q: float = 2.2
    smooth_lambda: float = 1e-3
    slice_half_width_ratio: float = 0.004
    min_slice_points: int = 12
    z_low_percentile: float = 5.0
    z_high_percentile: float = 95.0
    height_percentile: float = 95.0
    plantar_y_percentile: float = 95.0
    min_width_ratio: float = 0.35
    min_height_ratio: float = 0.08
    max_height_ratio: float = 0.45
    height_clearance: float = 0.04
    toe_height_clearance: float = 0.05
    toe_taper_min: float = 0.12
    smooth_iterations: int = 10
    smooth_step: float = 0.25
    foot_x_low_percentile: float = 1.0
    foot_x_high_percentile: float = 99.0
    section_mask_width: int = 96
    section_mask_height: int = 96
    section_point_radius_px: int = 2
    toe_close_min_radius_px: int = 1
    toe_close_max_radius_px: int = 7
    raw_section_debug_points: int = 96
    plantar_curve_smooth_lambda: float = 25.0
    plantar_curve_blend: float = 0.70
    plantar_curve_max_lift_ratio: float = 0.08


@dataclass(frozen=True)
class PseudoLastResult:
    """Generated pseudo-last mesh and diagnostic arrays."""

    mesh: MeshData
    bottom_mesh: MeshData
    sections: np.ndarray
    bottom_sections: np.ndarray
    x: np.ndarray
    center_z: np.ndarray
    left_z: np.ndarray
    right_z: np.ndarray
    height: np.ndarray
    support_left_z: np.ndarray
    support_right_z: np.ndarray
    inner_left_z: np.ndarray
    inner_right_z: np.ndarray
    metrics: Dict[str, object]
    config: PseudoLastConfig
    debug_arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def save_sections_npz(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            sections=self.sections.astype(np.float32),
            bottom_sections=self.bottom_sections.astype(np.float32),
            x=self.x.astype(np.float32),
            center_z=self.center_z.astype(np.float32),
            left_z=self.left_z.astype(np.float32),
            right_z=self.right_z.astype(np.float32),
            height=self.height.astype(np.float32),
            support_left_z=self.support_left_z.astype(np.float32),
            support_right_z=self.support_right_z.astype(np.float32),
            inner_left_z=self.inner_left_z.astype(np.float32),
            inner_right_z=self.inner_right_z.astype(np.float32),
            config_json=json.dumps(asdict(self.config)),
        )
        for key, value in self.debug_arrays.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind in {"b", "i", "u", "f"}:
                payload[key] = array.astype(np.float32) if array.dtype.kind == "f" else array
        np.savez_compressed(output_path, **payload)

    def save_metrics_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.metrics)
        payload["config"] = asdict(self.config)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


@dataclass(frozen=True)
class _SupportProfile:
    x: np.ndarray
    center_z: np.ndarray
    left_z: np.ndarray
    right_z: np.ndarray
    length_extent: float

    def center(self, query_x: np.ndarray | float) -> np.ndarray:
        return np.interp(query_x, self.x, self.center_z)

    def left(self, query_x: np.ndarray | float) -> np.ndarray:
        return np.interp(query_x, self.x, self.left_z)

    def right(self, query_x: np.ndarray | float) -> np.ndarray:
        return np.interp(query_x, self.x, self.right_z)


@dataclass(frozen=True)
class _FootbedSampler:
    x_centers: np.ndarray
    z_centers: np.ndarray
    heightmap: np.ndarray
    mask: Optional[np.ndarray]

    @classmethod
    def from_npz(cls, path: str | Path) -> "_FootbedSampler":
        payload = np.load(Path(path), allow_pickle=True)
        required = ["x_centers", "z_centers", "smooth_footbed_heightmap"]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"Missing footbed heightmap keys in {path}: {missing}")
        x_centers = np.asarray(payload["x_centers"], dtype=np.float32)
        z_centers = np.asarray(payload["z_centers"], dtype=np.float32)
        heightmap = np.asarray(payload["smooth_footbed_heightmap"], dtype=np.float32)
        mask = np.asarray(payload["footbed_mask"], dtype=bool) if "footbed_mask" in payload.files else None
        heightmap = _fill_invalid_heightmap(heightmap, x_centers, payload)
        return cls(x_centers=x_centers, z_centers=z_centers, heightmap=heightmap, mask=mask)

    def sample(self, query_x: np.ndarray | float, query_z: np.ndarray | float) -> np.ndarray:
        xq, zq = np.broadcast_arrays(np.asarray(query_x, dtype=np.float32), np.asarray(query_z, dtype=np.float32))
        flat_x = xq.reshape(-1)
        flat_z = zq.reshape(-1)

        ix = _fractional_index(flat_x, self.x_centers)
        iz = _fractional_index(flat_z, self.z_centers)
        ix0 = np.floor(ix).astype(np.int64)
        iz0 = np.floor(iz).astype(np.int64)
        ix1 = np.clip(ix0 + 1, 0, self.x_centers.size - 1)
        iz1 = np.clip(iz0 + 1, 0, self.z_centers.size - 1)
        tx = (ix - ix0).astype(np.float32)
        tz = (iz - iz0).astype(np.float32)

        h00 = self.heightmap[ix0, iz0]
        h10 = self.heightmap[ix1, iz0]
        h01 = self.heightmap[ix0, iz1]
        h11 = self.heightmap[ix1, iz1]
        h0 = h00 * (1.0 - tx) + h10 * tx
        h1 = h01 * (1.0 - tx) + h11 * tx
        sampled = h0 * (1.0 - tz) + h1 * tz
        return sampled.reshape(xq.shape)


def build_pseudo_last(
    foot_mesh: MeshData,
    support_json_path: str | Path,
    footbed_npz_path: str | Path,
    config: Optional[PseudoLastConfig] = None,
) -> PseudoLastResult:
    """Build a smooth pseudo-last from an aligned SUPR foot mesh."""

    cfg = config or PseudoLastConfig()
    _validate_config(cfg)
    support = _load_support_profile(support_json_path)
    sampler = _FootbedSampler.from_npz(footbed_npz_path)
    foot_vertices = np.asarray(foot_mesh.vertices, dtype=np.float32)
    if foot_vertices.ndim != 2 or foot_vertices.shape[1] != 3:
        raise ValueError("foot_mesh.vertices must have shape [V, 3]")

    base_x, x_heel, x_toe = _make_base_x_grid(foot_vertices, support, cfg)
    support_length = max(float(support.length_extent), float(support.x[-1] - support.x[0]), 1e-6)
    foot_length = max(float(x_toe - x_heel), 1e-6)

    if cfg.builder_mode == "surface_offset":
        raw = _estimate_base_profiles(foot_vertices, base_x, support, sampler, cfg, support_length)
        smoothed = _smooth_and_clamp_profiles(raw, base_x, support, cfg, support_length)
        template_sections, bottom_sections = _make_sections(base_x, smoothed, support, sampler, cfg, support_length)
        template_sections, bottom_sections, profile_payload = _append_toe_extension(
            template_sections,
            bottom_sections,
            base_x,
            smoothed,
            support,
            sampler,
            cfg,
            support_length,
            foot_length,
        )
        mesh = _build_surface_offset_mesh(foot_mesh, support, sampler, smoothed, base_x, cfg, support_length)
        sections = _sample_surface_debug_sections(mesh.vertices, profile_payload["x"], cfg.n_theta)
        debug_arrays: Dict[str, np.ndarray] = {}
        method = "surface_offset_supr_preserving"
    else:
        raw = _estimate_pdf_section_profiles(foot_mesh, base_x, support, sampler, cfg, support_length)
        smoothed = _smooth_and_clamp_pdf_profiles(raw, base_x, cfg, support_length)
        sections, bottom_sections = _make_sections(base_x, smoothed, support, sampler, cfg, support_length)
        sections, bottom_sections, profile_payload = _append_toe_extension(
            sections,
            bottom_sections,
            base_x,
            smoothed,
            support,
            sampler,
            cfg,
            support_length,
            foot_length,
        )
        mesh = _loft_sections(sections)
        mesh = _smooth_nonbottom_vertices(mesh, sections.shape[0], cfg.n_theta, cfg)
        mesh = _reclamp_section_loft_mesh(mesh, profile_payload, sampler, cfg, support_length)
        sections = mesh.vertices[: sections.shape[0] * cfg.n_theta].reshape(sections.shape[0], cfg.n_theta, 3).astype(np.float32)
        bottom_sections = sections[:, : cfg.n_theta // 2].astype(np.float32)
        debug_arrays = _make_section_loft_debug_arrays(raw, base_x, profile_payload, sections, bottom_sections, sampler, cfg)
        method = "section_loft_pdf_v1"

    bottom_mesh = _loft_bottom_sections(bottom_sections)

    boundary_edges = _count_boundary_edges(mesh.faces)
    width_violation = np.maximum(
        0.0,
        np.maximum(profile_payload["inner_left_z"] - profile_payload["left_z"], profile_payload["right_z"] - profile_payload["inner_right_z"]),
    )
    bottom_b = sampler.sample(bottom_sections[..., 0], bottom_sections[..., 2])
    bottom_distance = bottom_b - bottom_sections[..., 1]
    support_width = np.maximum(profile_payload["support_right_z"] - profile_payload["support_left_z"], 1e-6)
    target_width = np.maximum(profile_payload["right_z"] - profile_payload["left_z"], 0.0)
    support_conformity = target_width / support_width

    metrics: Dict[str, object] = {
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "bottom_vertex_count": int(bottom_mesh.vertices.shape[0]),
        "bottom_face_count": int(bottom_mesh.faces.shape[0]),
        "section_count": int(sections.shape[0]),
        "section_point_count": int(sections.shape[1]),
        "boundary_edge_count": int(boundary_edges),
        "support_length": float(support_length),
        "foot_length": float(foot_length),
        "toe_extension_length": float(profile_payload["x"][-1] - base_x[-1]),
        "max_width_violation": float(np.max(width_violation)) if width_violation.size else 0.0,
        "bottom_footbed_distance_mean": float(np.mean(bottom_distance)),
        "bottom_footbed_distance_p95_abs": float(np.percentile(np.abs(bottom_distance), 95)),
        "plantar_rmse_to_footbed": float(np.sqrt(np.mean(np.square(bottom_distance)))),
        "arch_lift_mean": float(np.mean(np.maximum(bottom_distance, 0.0))),
        "arch_lift_max": float(np.max(np.maximum(bottom_distance, 0.0))),
        "support_conformity_ratio_max": float(np.max(support_conformity)) if support_conformity.size else 0.0,
        "width_profile_roughness": _profile_roughness(target_width),
        "height_profile_roughness": _profile_roughness(profile_payload["height"]),
        "toe_component_count_after_s_box": int(_final_section_component_count_after_s_box(sections, profile_payload["x"], cfg)),
        "raw_toe_component_count_after_s_box": int(_toe_component_count_after_s_box(debug_arrays, profile_payload["x"], cfg)),
        "slice_fallback_count": int(np.sum(debug_arrays.get("slice_used_fallback", np.zeros((0,), dtype=np.int32)))),
        "sdf_cleanup_used": False,
        "method": method,
        "builder_mode": cfg.builder_mode,
        "x_range": [float(profile_payload["x"][0]), float(profile_payload["x"][-1])],
        "z_range": [float(np.min(sections[..., 2])), float(np.max(sections[..., 2]))],
        "y_range": [float(np.min(sections[..., 1])), float(np.max(sections[..., 1]))],
        "support_json": str(Path(support_json_path)),
        "footbed_npz": str(Path(footbed_npz_path)),
    }

    return PseudoLastResult(
        mesh=mesh,
        bottom_mesh=bottom_mesh,
        sections=sections.astype(np.float32),
        bottom_sections=bottom_sections.astype(np.float32),
        x=profile_payload["x"].astype(np.float32),
        center_z=profile_payload["center_z"].astype(np.float32),
        left_z=profile_payload["left_z"].astype(np.float32),
        right_z=profile_payload["right_z"].astype(np.float32),
        height=profile_payload["height"].astype(np.float32),
        support_left_z=profile_payload["support_left_z"].astype(np.float32),
        support_right_z=profile_payload["support_right_z"].astype(np.float32),
        inner_left_z=profile_payload["inner_left_z"].astype(np.float32),
        inner_right_z=profile_payload["inner_right_z"].astype(np.float32),
        metrics=metrics,
        config=cfg,
        debug_arrays=debug_arrays,
    )


def build_pseudo_last_from_paths(
    foot_obj_path: str | Path,
    support_json_path: str | Path,
    footbed_npz_path: str | Path,
    config: Optional[PseudoLastConfig] = None,
) -> PseudoLastResult:
    """Load inputs from disk and build a pseudo-last."""

    foot_mesh = load_triangle_mesh(foot_obj_path)
    return build_pseudo_last(foot_mesh, support_json_path, footbed_npz_path, config=config)


def save_pseudo_last_artifacts(
    result: PseudoLastResult,
    output_dir: str | Path,
    *,
    foot_mesh: Optional[MeshData] = None,
) -> Dict[str, str]:
    """Write OBJ/NPZ/JSON/PNG artifacts for a pseudo-last result."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "pseudo_last_obj": str(out / "pseudo_last.obj"),
        "pseudo_last_bottom_surface_obj": str(out / "pseudo_last_bottom_surface.obj"),
        "pseudo_last_sections_npz": str(out / "pseudo_last_sections.npz"),
        "pseudo_last_metrics_json": str(out / "pseudo_last_metrics.json"),
        "section_overlays_png": str(out / "section_overlays.png"),
        "pseudo_last_overlay_png": str(out / "pseudo_last_overlay.png"),
    }
    write_obj_mesh(artifacts["pseudo_last_obj"], result.mesh, comments=["SUPR-derived pseudo-last mesh."])
    write_obj_mesh(
        artifacts["pseudo_last_bottom_surface_obj"],
        result.bottom_mesh,
        comments=["Pseudo-last bottom surface sampled from the pseudo-footbed and SUPR arch."],
    )
    result.save_sections_npz(artifacts["pseudo_last_sections_npz"])
    result.save_metrics_json(artifacts["pseudo_last_metrics_json"])
    plot_section_overlays(result, artifacts["section_overlays_png"], foot_mesh=foot_mesh)
    plot_pseudo_last_overlay(result, artifacts["pseudo_last_overlay_png"], foot_mesh=foot_mesh)
    return artifacts


def plot_section_overlays(
    result: PseudoLastResult,
    output_path: str | Path,
    *,
    foot_mesh: Optional[MeshData] = None,
    section_count: int = 6,
) -> None:
    """Plot representative X slices of the generated last."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, result.sections.shape[0] - 1, min(section_count, result.sections.shape[0]), dtype=int)
    fig, axes = plt.subplots(2, int(np.ceil(indices.size / 2)), figsize=(14, 7), squeeze=False)
    axes_flat = axes.reshape(-1)
    foot_vertices = None if foot_mesh is None else np.asarray(foot_mesh.vertices, dtype=np.float32)
    slice_half_width = max(float(result.metrics["support_length"]) * 0.006, 1e-4)
    raw_points_debug = result.debug_arrays.get("raw_section_points")
    raw_counts_debug = result.debug_arrays.get("raw_section_counts")
    footbed_sections = result.debug_arrays.get("footbed_sections")

    for ax, idx in zip(axes_flat, indices):
        section = result.sections[idx]
        bottom = result.bottom_sections[idx]
        x = float(result.x[idx])
        ax.plot(section[:, 2], section[:, 1], color="tab:blue", linewidth=1.5, label="pseudo-last")
        ax.plot(bottom[:, 2], bottom[:, 1], color="tab:green", linewidth=1.2, label="bottom")
        if footbed_sections is not None and idx < footbed_sections.shape[0]:
            footbed = footbed_sections[idx]
            ax.plot(footbed[:, 2], footbed[:, 1], color="0.45", linestyle=":", linewidth=1.1, label="footbed")
        ax.axvline(float(result.inner_left_z[idx]), color="tab:red", linestyle="--", linewidth=0.9, label="inner support")
        ax.axvline(float(result.inner_right_z[idx]), color="tab:red", linestyle="--", linewidth=0.9)
        if raw_points_debug is not None and raw_counts_debug is not None and idx < raw_points_debug.shape[0]:
            count = int(raw_counts_debug[idx])
            pts = raw_points_debug[idx, :count]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.size:
                ax.scatter(pts[:, 2], pts[:, 1], s=7, color="0.20", alpha=0.45, label="raw SUPR section")
        elif foot_vertices is not None:
            mask = np.abs(foot_vertices[:, 0] - x) <= slice_half_width
            pts = foot_vertices[mask]
            if pts.size:
                ax.scatter(pts[:, 2], pts[:, 1], s=4, color="0.25", alpha=0.25, label="SUPR foot")
        ax.set_title(f"x={x:.3f}, s={(x - result.x[0]) / max(result.x[-1] - result.x[0], 1e-6):.2f}")
        ax.set_xlabel("Z width")
        ax.set_ylabel("Y bottom/opening")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[indices.size :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Pseudo-Last Section Overlays")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pseudo_last_overlay(
    result: PseudoLastResult,
    output_path: str | Path,
    *,
    foot_mesh: Optional[MeshData] = None,
) -> None:
    """Plot a lightweight 3D overlay of the foot and pseudo-last."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = result.mesh
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    face_step = max(1, faces.shape[0] // 2500)
    polys = vertices[faces[::face_step]]
    coll = Poly3DCollection(polys, facecolor=(0.25, 0.55, 0.95, 0.35), edgecolor=(0.1, 0.1, 0.1, 0.15), linewidth=0.2)
    ax.add_collection3d(coll)

    if foot_mesh is not None:
        pts = np.asarray(foot_mesh.vertices, dtype=np.float32)
        step = max(1, pts.shape[0] // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1], s=2, color="tab:orange", alpha=0.25)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] + radius, center[1] - radius)
    ax.set_xlabel("X length")
    ax.set_ylabel("Z width")
    ax.set_zlabel("Y bottom/opening")
    ax.set_title("Pseudo-Last Overlay")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _validate_config(cfg: PseudoLastConfig) -> None:
    if cfg.builder_mode not in {"section_loft", "surface_offset"}:
        raise ValueError("builder_mode must be 'section_loft' or 'surface_offset'")
    if cfg.n_x < 8:
        raise ValueError("n_x must be at least 8")
    if cfg.n_theta < 12 or cfg.n_theta % 2 != 0:
        raise ValueError("n_theta must be an even integer >= 12")
    if not (0.0 < cfg.eta_s <= 1.0):
        raise ValueError("eta_s must be in (0, 1]")
    if cfg.smooth_lambda < 0.0:
        raise ValueError("smooth_lambda must be non-negative")
    if cfg.toe_merge_full < cfg.toe_merge_start:
        raise ValueError("toe_merge_full must be >= toe_merge_start")
    if cfg.section_mask_width < 16 or cfg.section_mask_height < 16:
        raise ValueError("section mask resolution must be at least 16x16")
    if cfg.raw_section_debug_points <= 0:
        raise ValueError("raw_section_debug_points must be positive")
    if cfg.plantar_curve_smooth_lambda < 0.0:
        raise ValueError("plantar_curve_smooth_lambda must be non-negative")
    if not (0.0 <= cfg.plantar_curve_blend <= 1.0):
        raise ValueError("plantar_curve_blend must be in [0, 1]")
    if cfg.plantar_curve_max_lift_ratio <= 0.0:
        raise ValueError("plantar_curve_max_lift_ratio must be positive")


def _load_support_profile(path: str | Path) -> _SupportProfile:
    with Path(path).open("r") as f:
        payload = json.load(f)
    required = ["centerline_x", "centerline_z", "left_boundary_z", "right_boundary_z"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing support footprint fields in {path}: {missing}")
    x = np.asarray(payload["centerline_x"], dtype=np.float32)
    center = np.asarray(payload["centerline_z"], dtype=np.float32)
    left = np.asarray(payload["left_boundary_z"], dtype=np.float32)
    right = np.asarray(payload["right_boundary_z"], dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(center) & np.isfinite(left) & np.isfinite(right) & (right > left)
    if valid.sum() < 4:
        raise ValueError(f"Support profile has too few valid samples: {path}")
    order = np.argsort(x[valid])
    x = x[valid][order]
    center = center[valid][order]
    left = left[valid][order]
    right = right[valid][order]
    return _SupportProfile(
        x=x,
        center_z=center,
        left_z=left,
        right_z=right,
        length_extent=float(payload.get("length_extent", float(x[-1] - x[0]))),
    )


def _fill_invalid_heightmap(heightmap: np.ndarray, x_centers: np.ndarray, payload: np.lib.npyio.NpzFile) -> np.ndarray:
    values = np.asarray(heightmap, dtype=np.float32).copy()
    finite = np.isfinite(values)
    if finite.all():
        return values
    if "smooth_footbed_axis_profile" in payload.files and "centerline_x" in payload.files:
        profile = np.asarray(payload["smooth_footbed_axis_profile"], dtype=np.float32)
        centerline_x = np.asarray(payload["centerline_x"], dtype=np.float32)
        fallback = np.interp(x_centers, centerline_x, profile).astype(np.float32)
    elif finite.any():
        fallback = np.full((x_centers.size,), float(np.nanmedian(values[finite])), dtype=np.float32)
    else:
        fallback = np.zeros((x_centers.size,), dtype=np.float32)
    fallback_grid = np.repeat(fallback[:, None], values.shape[1], axis=1)
    values[~finite] = fallback_grid[~finite]
    return values


def _fractional_index(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.size < 2:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - centers[0]) / max(float(centers[-1] - centers[0]), 1e-8) * float(centers.size - 1)
    return np.clip(scaled, 0.0, float(centers.size - 1)).astype(np.float32)


def _make_base_x_grid(
    foot_vertices: np.ndarray,
    support: _SupportProfile,
    cfg: PseudoLastConfig,
) -> Tuple[np.ndarray, float, float]:
    foot_x0 = float(np.percentile(foot_vertices[:, 0], cfg.foot_x_low_percentile))
    foot_x1 = float(np.percentile(foot_vertices[:, 0], cfg.foot_x_high_percentile))
    support_x0 = float(support.x[0])
    support_x1 = float(support.x[-1])
    x_heel = max(foot_x0, support_x0)
    x_toe = min(foot_x1, support_x1)
    if x_toe <= x_heel + 0.25 * max(float(support_x1 - support_x0), 1e-6):
        x_heel = support_x0
        x_toe = support_x1
    return np.linspace(x_heel, x_toe, cfg.n_x, dtype=np.float32), float(x_heel), float(x_toe)


def _estimate_base_profiles(
    foot_vertices: np.ndarray,
    x_grid: np.ndarray,
    support: _SupportProfile,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Dict[str, np.ndarray]:
    support_left = support.left(x_grid).astype(np.float32)
    support_right = support.right(x_grid).astype(np.float32)
    support_center = support.center(x_grid).astype(np.float32)
    inner_left = support_center + cfg.eta_s * (support_left - support_center)
    inner_right = support_center + cfg.eta_s * (support_right - support_center)
    support_width = np.maximum(inner_right - inner_left, 1e-6)

    z_left = np.full_like(x_grid, np.nan, dtype=np.float32)
    z_right = np.full_like(x_grid, np.nan, dtype=np.float32)
    height = np.full_like(x_grid, np.nan, dtype=np.float32)
    plantar_lift = np.full_like(x_grid, np.nan, dtype=np.float32)
    slice_half_width = max(float(support_length * cfg.slice_half_width_ratio), float((x_grid[-1] - x_grid[0]) / max(x_grid.size - 1, 1)) * 0.7)
    height_cap = support_length * cfg.max_height_ratio * 1.4

    for index, x in enumerate(x_grid):
        mask = np.abs(foot_vertices[:, 0] - float(x)) <= slice_half_width
        if mask.sum() < cfg.min_slice_points:
            mask = np.abs(foot_vertices[:, 0] - float(x)) <= slice_half_width * 2.5
        pts = foot_vertices[mask]
        if pts.shape[0] < cfg.min_slice_points:
            continue
        footbed_y = sampler.sample(pts[:, 0], pts[:, 2])
        h = footbed_y - pts[:, 1]
        valid = np.isfinite(h) & (h >= -0.05 * support_length) & (h <= height_cap)
        pts = pts[valid]
        h = h[valid]
        if pts.shape[0] < cfg.min_slice_points:
            continue
        z_left[index] = float(np.percentile(pts[:, 2], cfg.z_low_percentile))
        z_right[index] = float(np.percentile(pts[:, 2], cfg.z_high_percentile))
        positive_h = h[h > 0.0]
        height[index] = float(np.percentile(positive_h if positive_h.size else h, cfg.height_percentile))
        plantar_y = float(np.percentile(pts[:, 1], cfg.plantar_y_percentile))
        center_b = float(sampler.sample(np.asarray([x], dtype=np.float32), np.asarray([support_center[index]], dtype=np.float32))[0])
        plantar_lift[index] = max(center_b - plantar_y, 0.0)

    z_left = _fill_nan_profile(x_grid, z_left)
    z_right = _fill_nan_profile(x_grid, z_right)
    height = _fill_nan_profile(x_grid, height)
    plantar_lift = _fill_nan_profile(x_grid, plantar_lift)

    foot_width = np.maximum(z_right - z_left, cfg.min_width_ratio * support_width)
    return {
        "support_left": support_left,
        "support_right": support_right,
        "support_center": support_center,
        "inner_left": inner_left.astype(np.float32),
        "inner_right": inner_right.astype(np.float32),
        "support_width": support_width.astype(np.float32),
        "foot_z_left": z_left.astype(np.float32),
        "foot_z_right": z_right.astype(np.float32),
        "foot_width": foot_width.astype(np.float32),
        "height": height.astype(np.float32),
        "plantar_lift": plantar_lift.astype(np.float32),
    }


def _smooth_and_clamp_profiles(
    raw: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    support: _SupportProfile,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Dict[str, np.ndarray]:
    s = (x_grid - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6)
    heel_w = 1.0 - _smoothstep(0.12, 0.32, s)
    fore_w = _smoothstep(0.55, 0.85, s)
    mid_w = np.clip(1.0 - heel_w - 0.35 * fore_w, 0.0, 1.0)
    delta_side = raw["foot_width"] * (0.015 * heel_w + 0.025 * mid_w + 0.045 * fore_w)

    desired_left = raw["foot_z_left"] - delta_side
    desired_right = raw["foot_z_right"] + delta_side
    left = np.maximum(raw["inner_left"], desired_left)
    right = np.minimum(raw["inner_right"], desired_right)
    min_width = np.minimum(raw["support_width"], np.maximum(cfg.min_width_ratio * raw["support_width"], raw["foot_width"] * 0.8))
    center_desired = 0.5 * (raw["foot_z_left"] + raw["foot_z_right"])

    too_narrow = (right - left) < min_width
    if np.any(too_narrow):
        width = min_width[too_narrow]
        center = np.clip(
            center_desired[too_narrow],
            raw["inner_left"][too_narrow] + 0.5 * width,
            raw["inner_right"][too_narrow] - 0.5 * width,
        )
        left[too_narrow] = center - 0.5 * width
        right[too_narrow] = center + 0.5 * width

    center_z = 0.5 * (left + right)
    height_clearance = cfg.height_clearance + cfg.toe_height_clearance * fore_w
    height = raw["height"] * (1.0 + height_clearance)
    height = np.clip(height, cfg.min_height_ratio * support_length, cfg.max_height_ratio * support_length)

    left = _smooth_profile(left.astype(np.float32), cfg.smooth_lambda)
    right = _smooth_profile(right.astype(np.float32), cfg.smooth_lambda)
    height = _smooth_profile(height.astype(np.float32), cfg.smooth_lambda)
    plantar_lift = _smooth_profile(raw["plantar_lift"].astype(np.float32), cfg.smooth_lambda)

    left = np.maximum(left, raw["inner_left"])
    right = np.minimum(right, raw["inner_right"])
    invalid = right <= left + 1e-5
    if np.any(invalid):
        center = raw["support_center"][invalid]
        width = np.maximum(raw["support_width"][invalid] * cfg.min_width_ratio, 1e-4)
        left[invalid] = center - 0.5 * width
        right[invalid] = center + 0.5 * width

    return {
        "support_left": raw["support_left"],
        "support_right": raw["support_right"],
        "inner_left": raw["inner_left"],
        "inner_right": raw["inner_right"],
        "left": left.astype(np.float32),
        "right": right.astype(np.float32),
        "center": center_z.astype(np.float32),
        "height": height.astype(np.float32),
        "plantar_lift": np.maximum(plantar_lift, 0.0).astype(np.float32),
    }


def _make_sections(
    x_grid: np.ndarray,
    profiles: Dict[str, np.ndarray],
    support: _SupportProfile,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    del support
    sections = []
    bottom_sections = []
    for i, x in enumerate(x_grid):
        s = float((x - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6))
        section, bottom = _make_single_section(
            float(x),
            s,
            float(profiles["left"][i]),
            float(profiles["right"][i]),
            float(profiles["height"][i]),
            float(profiles["plantar_lift"][i]),
            None if "plantar_lift_curve" not in profiles else profiles["plantar_lift_curve"][i],
            sampler,
            cfg,
            support_length,
        )
        sections.append(section)
        bottom_sections.append(bottom)
    return np.stack(sections, axis=0), np.stack(bottom_sections, axis=0)


def _append_toe_extension(
    sections: np.ndarray,
    bottom_sections: np.ndarray,
    base_x: np.ndarray,
    profiles: Dict[str, np.ndarray],
    support: _SupportProfile,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
    foot_length: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    x_toe = float(base_x[-1])
    x_end = min(x_toe + cfg.toe_allowance_ratio * foot_length, float(support.x[-1]))
    extra_sections = []
    extra_bottoms = []
    extra_x = []
    extra_left = []
    extra_right = []
    extra_height = []
    extra_center = []
    extra_support_left = []
    extra_support_right = []
    extra_inner_left = []
    extra_inner_right = []
    extra_plantar_lift_curve = []
    plantar_lift_curve = profiles.get("plantar_lift_curve")

    if x_end > x_toe + 1e-5:
        ext_count = max(4, int(round(cfg.n_x * cfg.toe_allowance_ratio)))
        toe_width = max(float(profiles["right"][-1] - profiles["left"][-1]), 1e-5)
        toe_center = float(0.5 * (profiles["right"][-1] + profiles["left"][-1]))
        toe_height = float(profiles["height"][-1])
        for x in np.linspace(x_toe, x_end, ext_count + 1, dtype=np.float32)[1:]:
            u = float((x - x_toe) / max(x_end - x_toe, 1e-6))
            taper = max(float(np.sqrt(max(1.0 - u * u, 0.0))), cfg.toe_taper_min)
            support_left = float(support.left(float(x)))
            support_right = float(support.right(float(x)))
            support_center = float(support.center(float(x)))
            inner_left = support_center + cfg.eta_s * (support_left - support_center)
            inner_right = support_center + cfg.eta_s * (support_right - support_center)
            width = min(toe_width * taper, max(inner_right - inner_left, 1e-5))
            center = float(np.clip(toe_center, inner_left + 0.5 * width, inner_right - 0.5 * width))
            left = center - 0.5 * width
            right = center + 0.5 * width
            height = max(toe_height * taper, cfg.min_height_ratio * support_length)
            section, bottom = _make_single_section(
                float(x),
                1.0,
                left,
                right,
                height,
                float(profiles["plantar_lift"][-1]) * taper,
                None if plantar_lift_curve is None else plantar_lift_curve[-1] * taper,
                sampler,
                cfg,
                support_length,
            )
            extra_sections.append(section)
            extra_bottoms.append(bottom)
            extra_x.append(float(x))
            extra_left.append(left)
            extra_right.append(right)
            extra_height.append(height)
            extra_center.append(center)
            extra_support_left.append(support_left)
            extra_support_right.append(support_right)
            extra_inner_left.append(inner_left)
            extra_inner_right.append(inner_right)
            if plantar_lift_curve is not None:
                extra_plantar_lift_curve.append(plantar_lift_curve[-1] * taper)

    if extra_sections:
        sections = np.concatenate([sections, np.stack(extra_sections, axis=0)], axis=0)
        bottom_sections = np.concatenate([bottom_sections, np.stack(extra_bottoms, axis=0)], axis=0)
        x = np.concatenate([base_x, np.asarray(extra_x, dtype=np.float32)])
        left = np.concatenate([profiles["left"], np.asarray(extra_left, dtype=np.float32)])
        right = np.concatenate([profiles["right"], np.asarray(extra_right, dtype=np.float32)])
        height = np.concatenate([profiles["height"], np.asarray(extra_height, dtype=np.float32)])
        center = np.concatenate([0.5 * (profiles["left"] + profiles["right"]), np.asarray(extra_center, dtype=np.float32)])
        support_left = np.concatenate([profiles["support_left"], np.asarray(extra_support_left, dtype=np.float32)])
        support_right = np.concatenate([profiles["support_right"], np.asarray(extra_support_right, dtype=np.float32)])
        inner_left = np.concatenate([profiles["inner_left"], np.asarray(extra_inner_left, dtype=np.float32)])
        inner_right = np.concatenate([profiles["inner_right"], np.asarray(extra_inner_right, dtype=np.float32)])
        if plantar_lift_curve is not None:
            plantar_lift_curve = np.concatenate([plantar_lift_curve, np.stack(extra_plantar_lift_curve, axis=0).astype(np.float32)], axis=0)
    else:
        x = base_x
        left = profiles["left"]
        right = profiles["right"]
        height = profiles["height"]
        center = 0.5 * (left + right)
        support_left = profiles["support_left"]
        support_right = profiles["support_right"]
        inner_left = profiles["inner_left"]
        inner_right = profiles["inner_right"]
        plantar_lift_curve = profiles.get("plantar_lift_curve")

    payload = {
        "x": x.astype(np.float32),
        "left_z": left.astype(np.float32),
        "right_z": right.astype(np.float32),
        "center_z": center.astype(np.float32),
        "height": height.astype(np.float32),
        "support_left_z": support_left.astype(np.float32),
        "support_right_z": support_right.astype(np.float32),
        "inner_left_z": inner_left.astype(np.float32),
        "inner_right_z": inner_right.astype(np.float32),
    }
    if plantar_lift_curve is not None:
        payload["plantar_lift_curve"] = plantar_lift_curve.astype(np.float32)

    return (
        sections.astype(np.float32),
        bottom_sections.astype(np.float32),
        payload,
    )


def _estimate_pdf_section_profiles(
    foot_mesh: MeshData,
    x_grid: np.ndarray,
    support: _SupportProfile,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Dict[str, np.ndarray]:
    vertices = np.asarray(foot_mesh.vertices, dtype=np.float32)
    faces = np.asarray(foot_mesh.faces, dtype=np.int64)
    support_left = support.left(x_grid).astype(np.float32)
    support_right = support.right(x_grid).astype(np.float32)
    support_center = support.center(x_grid).astype(np.float32)
    support_width = np.maximum(support_right - support_left, 1e-6).astype(np.float32)

    raw_width = np.full_like(x_grid, np.nan, dtype=np.float32)
    raw_height = np.full_like(x_grid, np.nan, dtype=np.float32)
    raw_center_offset = np.full_like(x_grid, np.nan, dtype=np.float32)
    plantar_lift = np.full_like(x_grid, np.nan, dtype=np.float32)
    plantar_lift_curve = np.full((x_grid.size, cfg.n_theta // 2), np.nan, dtype=np.float32)
    component_count = np.zeros_like(x_grid, dtype=np.int32)
    used_fallback = np.zeros_like(x_grid, dtype=np.int32)
    raw_section_points = np.full(
        (x_grid.size, max(1, int(cfg.raw_section_debug_points)), 3),
        np.nan,
        dtype=np.float32,
    )
    raw_section_counts = np.zeros((x_grid.size,), dtype=np.int32)

    dx = float(x_grid[-1] - x_grid[0]) / max(x_grid.size - 1, 1)
    slice_half_width = max(float(support_length * cfg.slice_half_width_ratio), dx * 0.75, 1e-5)
    height_cap = support_length * cfg.max_height_ratio * 1.5

    for index, x in enumerate(x_grid):
        s = float((x - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6))
        section_points, fallback = _extract_x_section_points(vertices, faces, float(x), slice_half_width, cfg.min_slice_points)
        used_fallback[index] = int(fallback)
        if section_points.shape[0] < 3:
            continue

        footbed_y = sampler.sample(section_points[:, 0], section_points[:, 2])
        height_above_bottom = footbed_y - section_points[:, 1]
        z_margin = 0.25 * float(support_width[index])
        valid = (
            np.isfinite(height_above_bottom)
            & (height_above_bottom >= -0.04 * support_length)
            & (height_above_bottom <= height_cap)
            & (section_points[:, 2] >= support_left[index] - z_margin)
            & (section_points[:, 2] <= support_right[index] + z_margin)
        )
        section_points = section_points[valid]
        height_above_bottom = height_above_bottom[valid]
        if section_points.shape[0] < 3:
            continue

        raw_section_counts[index] = min(section_points.shape[0], raw_section_points.shape[1])
        raw_section_points[index, : raw_section_counts[index]] = _sample_debug_points(section_points, raw_section_points.shape[1])

        xi = section_points[:, 2] - float(support_center[index])
        local = np.stack([xi, np.maximum(height_above_bottom, 0.0)], axis=1).astype(np.float32)
        mask, xi_axis, h_axis = _rasterize_pdf_section_mask(local, float(support_width[index]), support_length, cfg)
        if s >= cfg.toe_merge_start:
            mask = _close_toe_mask(mask, s, cfg)
        width, height, center_offset = _section_mask_stats(mask, xi_axis, h_axis)
        if not np.isfinite(width) or width <= 1e-6:
            width = float(np.percentile(section_points[:, 2], cfg.z_high_percentile) - np.percentile(section_points[:, 2], cfg.z_low_percentile))
            center_offset = float(0.5 * (np.percentile(xi, cfg.z_low_percentile) + np.percentile(xi, cfg.z_high_percentile)))
        if not np.isfinite(height) or height <= 1e-6:
            positive_h = local[:, 1][local[:, 1] > 0.0]
            height = float(np.percentile(positive_h if positive_h.size else local[:, 1], cfg.height_percentile))

        raw_width[index] = max(float(width), 1e-6)
        raw_height[index] = max(float(height), 1e-6)
        raw_center_offset[index] = float(center_offset)
        plantar_y = float(np.percentile(section_points[:, 1], cfg.plantar_y_percentile))
        center_b = float(sampler.sample(np.asarray([x], dtype=np.float32), np.asarray([support_center[index]], dtype=np.float32))[0])
        plantar_lift[index] = max(center_b - plantar_y, 0.0)
        plantar_lift_curve[index] = _estimate_plantar_lift_curve(
            float(x),
            float(support_center[index]),
            0.5 * float(support_width[index]),
            section_points,
            sampler,
            cfg,
        )
        component_count[index] = _mask_component_count(mask)

    raw_width = _fill_nan_profile(x_grid, raw_width)
    raw_height = _fill_nan_profile(x_grid, raw_height)
    raw_center_offset = _fill_nan_profile(x_grid, raw_center_offset)
    plantar_lift = _fill_nan_profile(x_grid, plantar_lift)
    plantar_lift_curve = _fill_nan_lift_curves(x_grid, plantar_lift_curve)
    raw_width = np.maximum(raw_width, cfg.min_width_ratio * support_width)
    raw_height = np.clip(raw_height, cfg.min_height_ratio * support_length, cfg.max_height_ratio * support_length)

    return {
        "support_left": support_left,
        "support_right": support_right,
        "support_center": support_center,
        "support_width": support_width,
        "raw_width": raw_width.astype(np.float32),
        "raw_height": raw_height.astype(np.float32),
        "raw_center_offset": raw_center_offset.astype(np.float32),
        "plantar_lift": np.maximum(plantar_lift, 0.0).astype(np.float32),
        "plantar_lift_curve": np.maximum(plantar_lift_curve, 0.0).astype(np.float32),
        "raw_section_points": raw_section_points.astype(np.float32),
        "raw_section_counts": raw_section_counts.astype(np.int32),
        "slice_used_fallback": used_fallback.astype(np.int32),
        "closed_component_count": component_count.astype(np.int32),
    }


def _smooth_and_clamp_pdf_profiles(
    raw: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Dict[str, np.ndarray]:
    s = (x_grid - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6)
    support_center = _smooth_profile(raw["support_center"].astype(np.float32), cfg.smooth_lambda)
    support_width = _smooth_profile(raw["support_width"].astype(np.float32), cfg.smooth_lambda)
    support_width = np.maximum(support_width, 1e-6)
    support_left = support_center - 0.5 * support_width
    support_right = support_center + 0.5 * support_width
    inner_left = support_center - 0.5 * cfg.eta_s * support_width
    inner_right = support_center + 0.5 * cfg.eta_s * support_width

    raw_width = _smooth_profile(raw["raw_width"].astype(np.float32), cfg.smooth_lambda)
    raw_height = _smooth_profile(raw["raw_height"].astype(np.float32), cfg.smooth_lambda)
    plantar_lift = _smooth_profile(raw["plantar_lift"].astype(np.float32), cfg.smooth_lambda)
    plantar_lift_curve = _smooth_lift_curves(raw["plantar_lift_curve"].astype(np.float32), cfg.plantar_curve_smooth_lambda)
    max_lift = cfg.plantar_curve_max_lift_ratio * support_length
    plantar_lift_curve = np.clip(plantar_lift_curve, 0.0, max_lift)
    plantar_lift = np.clip(plantar_lift, 0.0, max_lift)

    heel_w = 1.0 - _smoothstep(0.08, 0.25, s)
    mid_w = _smoothstep(0.30, 0.55, s) * (1.0 - _smoothstep(0.55, 0.75, s))
    fore_w = _smoothstep(0.60, 0.85, s)
    side_clearance = raw_width * (
        cfg.heel_clearance_ratio * heel_w
        + cfg.midfoot_clearance_ratio * mid_w
        + cfg.forefoot_clearance_ratio * fore_w
    )
    target_half = np.minimum(0.5 * cfg.eta_s * support_width, 0.5 * raw_width + side_clearance)
    min_half = np.minimum(0.5 * cfg.eta_s * support_width, np.maximum(0.5 * cfg.min_width_ratio * support_width, 0.38 * raw_width))
    target_half = np.maximum(target_half, min_half)

    vamp_clearance = cfg.vamp_clearance_ratio * _smoothstep(0.35, 0.65, s)
    toe_clearance = cfg.toe_vertical_clearance_ratio * _smoothstep(0.72, 0.95, s)
    height = raw_height * (1.0 + vamp_clearance + toe_clearance)
    height = np.clip(height, cfg.min_height_ratio * support_length, cfg.max_height_ratio * support_length)

    left = support_center - target_half
    right = support_center + target_half
    left = np.maximum(left, inner_left)
    right = np.minimum(right, inner_right)
    invalid = right <= left + 1e-5
    if np.any(invalid):
        fallback_half = np.maximum(0.5 * cfg.min_width_ratio * support_width[invalid], 1e-5)
        left[invalid] = support_center[invalid] - fallback_half
        right[invalid] = support_center[invalid] + fallback_half

    return {
        "support_left": support_left.astype(np.float32),
        "support_right": support_right.astype(np.float32),
        "inner_left": inner_left.astype(np.float32),
        "inner_right": inner_right.astype(np.float32),
        "left": left.astype(np.float32),
        "right": right.astype(np.float32),
        "center": support_center.astype(np.float32),
        "height": height.astype(np.float32),
        "plantar_lift": np.maximum(plantar_lift, 0.0).astype(np.float32),
        "plantar_lift_curve": np.maximum(plantar_lift_curve, 0.0).astype(np.float32),
        "raw_width_smoothed": raw_width.astype(np.float32),
        "raw_height_smoothed": raw_height.astype(np.float32),
    }


def _extract_x_section_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    x: float,
    slice_half_width: float,
    min_points: int,
) -> Tuple[np.ndarray, bool]:
    eps = 1e-7
    points = []
    for tri in np.asarray(faces, dtype=np.int64):
        tri_vertices = vertices[tri]
        xs = tri_vertices[:, 0]
        if x < float(np.min(xs)) - eps or x > float(np.max(xs)) + eps:
            continue
        tri_points = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p0 = tri_vertices[a]
            p1 = tri_vertices[b]
            d0 = float(p0[0] - x)
            d1 = float(p1[0] - x)
            if abs(d0) <= eps and abs(d1) <= eps:
                tri_points.extend([p0, p1])
            elif abs(d0) <= eps:
                tri_points.append(p0)
            elif abs(d1) <= eps:
                tri_points.append(p1)
            elif d0 * d1 < 0.0:
                t = (x - float(p0[0])) / max(float(p1[0] - p0[0]), 1e-12)
                tri_points.append(p0 + np.float32(t) * (p1 - p0))
        if tri_points:
            points.extend(tri_points)

    exact = _unique_points(np.asarray(points, dtype=np.float32)) if points else np.zeros((0, 3), dtype=np.float32)
    if exact.shape[0] >= min_points:
        return exact, False

    mask = np.abs(vertices[:, 0] - x) <= slice_half_width
    fallback = vertices[mask]
    if fallback.shape[0] < min_points:
        mask = np.abs(vertices[:, 0] - x) <= 2.5 * slice_half_width
        fallback = vertices[mask]
    if fallback.shape[0] < min_points:
        order = np.argsort(np.abs(vertices[:, 0] - x))
        fallback = vertices[order[: max(min_points, min(48, vertices.shape[0]))]]
    if exact.shape[0] > 0:
        fallback = np.concatenate([exact, fallback.astype(np.float32)], axis=0)
    return _unique_points(fallback.astype(np.float32)), True


def _unique_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    rounded = np.round(points.astype(np.float64), 7)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_indices)].astype(np.float32)


def _sample_debug_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.astype(np.float32)
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices].astype(np.float32)


def _estimate_plantar_lift_curve(
    x: float,
    center_z: float,
    support_half_width: float,
    section_points: np.ndarray,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
) -> np.ndarray:
    bin_count = cfg.n_theta // 2
    t_axis = np.linspace(-1.0, 1.0, bin_count, dtype=np.float32)
    z_axis = center_z + t_axis * max(float(support_half_width), 1e-6)
    footbed_y = sampler.sample(np.full_like(z_axis, x, dtype=np.float32), z_axis)

    points = np.asarray(section_points, dtype=np.float32)
    local_t = (points[:, 2] - float(center_z)) / max(float(support_half_width), 1e-6)
    y_values = points[:, 1]
    bin_radius = max(2.5 / max(bin_count - 1, 1), 0.045)
    plantar_y = np.full((bin_count,), np.nan, dtype=np.float32)
    for index, t in enumerate(t_axis):
        mask = np.abs(local_t - float(t)) <= bin_radius
        if not np.any(mask):
            continue
        plantar_y[index] = float(np.percentile(y_values[mask], cfg.plantar_y_percentile))

    plantar_y = _fill_nan_profile(t_axis, plantar_y)
    lift = np.maximum(footbed_y - plantar_y, 0.0)
    lift = _smooth_profile(lift.astype(np.float32), cfg.plantar_curve_smooth_lambda)
    return np.maximum(lift, 0.0).astype(np.float32)


def _fill_nan_lift_curves(x_grid: np.ndarray, curves: np.ndarray) -> np.ndarray:
    values = np.asarray(curves, dtype=np.float32).copy()
    if values.ndim != 2:
        raise ValueError("plantar lift curves must have shape [N, B]")
    t_axis = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    for index in range(values.shape[0]):
        values[index] = _fill_nan_profile(t_axis, values[index])
    for column in range(values.shape[1]):
        values[:, column] = _fill_nan_profile(x_grid, values[:, column])
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _smooth_lift_curves(curves: np.ndarray, smooth_lambda: float) -> np.ndarray:
    values = np.asarray(curves, dtype=np.float32).copy()
    if values.ndim != 2:
        raise ValueError("plantar lift curves must have shape [N, B]")
    if smooth_lambda <= 0.0:
        return np.maximum(values, 0.0).astype(np.float32)
    for column in range(values.shape[1]):
        values[:, column] = _smooth_profile(values[:, column], smooth_lambda)
    for row in range(values.shape[0]):
        values[row] = _smooth_profile(values[row], smooth_lambda)
    return np.maximum(values, 0.0).astype(np.float32)


def _sample_plantar_lift_curve(
    plantar_lift_curve: Optional[np.ndarray],
    z: np.ndarray,
    center: float,
    half_width: float,
) -> np.ndarray:
    if plantar_lift_curve is None:
        return np.zeros_like(z, dtype=np.float32)
    curve = np.asarray(plantar_lift_curve, dtype=np.float32).reshape(-1)
    if curve.size < 2 or not np.isfinite(curve).any():
        return np.zeros_like(z, dtype=np.float32)
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
    t_axis = np.linspace(-1.0, 1.0, curve.size, dtype=np.float32)
    query_t = np.clip((np.asarray(z, dtype=np.float32) - float(center)) / max(float(half_width), 1e-6), -1.0, 1.0)
    return np.interp(query_t, t_axis, curve).astype(np.float32)


def _rasterize_pdf_section_mask(
    local_points: np.ndarray,
    support_width: float,
    support_length: float,
    cfg: PseudoLastConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(int(cfg.section_mask_width), 16)
    height = max(int(cfg.section_mask_height), 16)
    points = np.asarray(local_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        return np.zeros((height, width), dtype=bool), np.linspace(-1.0, 1.0, width), np.linspace(0.0, 1.0, height)

    xi_abs = float(np.percentile(np.abs(points[:, 0]), 98)) if points.shape[0] else 0.0
    h_max_raw = float(np.percentile(points[:, 1], 98)) if points.shape[0] else 0.0
    xi_half = max(0.55 * support_width, 1.20 * xi_abs, cfg.min_width_ratio * support_width, 1e-5)
    h_max = max(1.20 * h_max_raw, cfg.min_height_ratio * support_length, 1e-5)
    xi_axis = np.linspace(-xi_half, xi_half, width, dtype=np.float32)
    h_axis = np.linspace(0.0, h_max, height, dtype=np.float32)
    xi_grid, h_grid = np.meshgrid(xi_axis, h_axis)
    grid_points = np.stack([xi_grid.reshape(-1), h_grid.reshape(-1)], axis=1)

    mask = np.zeros((height, width), dtype=bool)
    try:
        from matplotlib.path import Path as MplPath

        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        polygon = points[np.argsort(angles)]
        mask = MplPath(polygon).contains_points(grid_points).reshape(height, width)
    except Exception:
        mask = np.zeros((height, width), dtype=bool)

    point_mask = np.zeros_like(mask)
    cols = np.clip(np.round((points[:, 0] - xi_axis[0]) / max(float(xi_axis[-1] - xi_axis[0]), 1e-8) * (width - 1)).astype(np.int64), 0, width - 1)
    rows = np.clip(np.round(points[:, 1] / max(float(h_axis[-1]), 1e-8) * (height - 1)).astype(np.int64), 0, height - 1)
    point_mask[rows, cols] = True
    ndimage = _scipy_ndimage()
    if ndimage is not None and cfg.section_point_radius_px > 0:
        kernel = _ellipse_kernel(cfg.section_point_radius_px, cfg.section_point_radius_px)
        point_mask = ndimage.binary_dilation(point_mask, structure=kernel)
    mask = mask | point_mask
    if ndimage is not None:
        mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool), xi_axis, h_axis


def _close_toe_mask(mask: np.ndarray, s: float, cfg: PseudoLastConfig) -> np.ndarray:
    ndimage = _scipy_ndimage()
    if ndimage is None or not np.any(mask):
        return mask.astype(bool)
    mu = float(_smoothstep(cfg.toe_merge_start, cfg.toe_merge_full, np.asarray([s], dtype=np.float32))[0])
    radius_x = int(round(cfg.toe_close_min_radius_px + mu * (cfg.toe_close_max_radius_px - cfg.toe_close_min_radius_px)))
    radius_y = max(1, int(round(0.55 * radius_x)))
    closed = ndimage.binary_closing(mask, structure=_ellipse_kernel(radius_x, radius_y))
    closed = ndimage.binary_fill_holes(closed)
    return closed.astype(bool)


def _ellipse_kernel(radius_x: int, radius_y: int) -> np.ndarray:
    rx = max(1, int(radius_x))
    ry = max(1, int(radius_y))
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    return (xx * xx / float(rx * rx) + yy * yy / float(ry * ry)) <= 1.0


def _scipy_ndimage():
    try:
        from scipy import ndimage

        return ndimage
    except Exception:
        return None


def _section_mask_stats(mask: np.ndarray, xi_axis: np.ndarray, h_axis: np.ndarray) -> Tuple[float, float, float]:
    if not np.any(mask):
        return float("nan"), float("nan"), 0.0
    rows, cols = np.nonzero(mask)
    width = float(xi_axis[int(cols.max())] - xi_axis[int(cols.min())])
    height = float(h_axis[int(rows.max())])
    center_offset = float(0.5 * (xi_axis[int(cols.max())] + xi_axis[int(cols.min())]))
    return width, height, center_offset


def _mask_component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    ndimage = _scipy_ndimage()
    if ndimage is None:
        return 1
    _, count = ndimage.label(mask)
    return int(count)


def _reclamp_section_loft_mesh(
    mesh: MeshData,
    profiles: Dict[str, np.ndarray],
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
) -> MeshData:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    x = np.clip(vertices[:, 0], float(profiles["x"][0]), float(profiles["x"][-1]))
    inner_left = np.interp(x, profiles["x"], profiles["inner_left_z"]).astype(np.float32)
    inner_right = np.interp(x, profiles["x"], profiles["inner_right_z"]).astype(np.float32)
    margin = cfg.support_z_margin_ratio * support_length
    if np.all(inner_right - inner_left > 2.0 * margin):
        vertices[:, 2] = np.clip(vertices[:, 2], inner_left + margin, inner_right - margin)
    else:
        vertices[:, 2] = np.clip(vertices[:, 2], inner_left, inner_right)
    footbed_y = sampler.sample(vertices[:, 0], vertices[:, 2])
    vertices[:, 1] = np.minimum(vertices[:, 1], footbed_y)
    return MeshData(vertices=vertices.astype(np.float32), faces=faces)


def _make_section_loft_debug_arrays(
    raw: Dict[str, np.ndarray],
    base_x: np.ndarray,
    profiles: Dict[str, np.ndarray],
    sections: np.ndarray,
    bottom_sections: np.ndarray,
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
) -> Dict[str, np.ndarray]:
    del cfg
    full_x = profiles["x"].astype(np.float32)
    extra_count = max(0, full_x.size - base_x.size)

    def pad_1d(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        values = np.asarray(values)
        if extra_count <= 0:
            return values
        padding = np.full((extra_count,), fill, dtype=values.dtype)
        return np.concatenate([values, padding], axis=0)

    raw_points = raw["raw_section_points"]
    if extra_count > 0:
        raw_points = np.concatenate(
            [
                raw_points,
                np.full((extra_count, raw_points.shape[1], 3), np.nan, dtype=np.float32),
            ],
            axis=0,
        )

    raw_curve = raw.get("plantar_lift_curve")
    smoothed_curve = profiles.get("plantar_lift_curve")
    if raw_curve is not None and extra_count > 0:
        raw_curve = np.concatenate(
            [
                raw_curve,
                np.full((extra_count, raw_curve.shape[1]), np.nan, dtype=np.float32),
            ],
            axis=0,
        )

    footbed_y = sampler.sample(bottom_sections[..., 0], bottom_sections[..., 2])
    footbed_sections = np.stack([bottom_sections[..., 0], footbed_y.astype(np.float32), bottom_sections[..., 2]], axis=-1)
    support_width = np.maximum(profiles["support_right_z"] - profiles["support_left_z"], 1e-6)
    target_width = np.maximum(profiles["right_z"] - profiles["left_z"], 0.0)
    arch_lift = np.maximum(footbed_y - bottom_sections[..., 1], 0.0)

    return {
        "raw_width_profile": pad_1d(raw["raw_width"].astype(np.float32)).astype(np.float32),
        "raw_height_profile": pad_1d(raw["raw_height"].astype(np.float32)).astype(np.float32),
        "raw_center_offset_profile": pad_1d(raw["raw_center_offset"].astype(np.float32)).astype(np.float32),
        "raw_plantar_lift_profile": pad_1d(raw["plantar_lift"].astype(np.float32)).astype(np.float32),
        "raw_plantar_lift_curve": np.asarray(raw_curve, dtype=np.float32) if raw_curve is not None else np.zeros((0, 0), dtype=np.float32),
        "smoothed_plantar_lift_curve": np.asarray(smoothed_curve, dtype=np.float32) if smoothed_curve is not None else np.zeros((0, 0), dtype=np.float32),
        "smoothed_width_profile": target_width.astype(np.float32),
        "smoothed_height_profile": profiles["height"].astype(np.float32),
        "support_width_profile": support_width.astype(np.float32),
        "support_conformity_ratio": (target_width / support_width).astype(np.float32),
        "arch_lift_profile": np.mean(arch_lift, axis=1).astype(np.float32),
        "footbed_sections": footbed_sections.astype(np.float32),
        "raw_section_points": raw_points.astype(np.float32),
        "raw_section_counts": pad_1d(raw["raw_section_counts"].astype(np.int32), fill=0).astype(np.int32),
        "slice_used_fallback": pad_1d(raw["slice_used_fallback"].astype(np.int32), fill=0).astype(np.int32),
        "closed_component_count": pad_1d(raw["closed_component_count"].astype(np.int32), fill=0).astype(np.int32),
    }


def _profile_roughness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return 0.0
    return float(np.sum(np.square(np.diff(values, n=2))))


def _toe_component_count_after_s_box(debug_arrays: Dict[str, np.ndarray], x: np.ndarray, cfg: PseudoLastConfig) -> int:
    counts = debug_arrays.get("closed_component_count")
    if counts is None or len(counts) == 0:
        return 0
    s = (x - x[0]) / max(float(x[-1] - x[0]), 1e-6)
    selected = np.asarray(counts)[(s >= cfg.toe_merge_full) & (np.asarray(counts) > 0)]
    if selected.size == 0:
        return 1
    return int(np.max(selected))


def _final_section_component_count_after_s_box(sections: np.ndarray, x: np.ndarray, cfg: PseudoLastConfig) -> int:
    if sections.size == 0 or x.size == 0:
        return 0
    s = (x - x[0]) / max(float(x[-1] - x[0]), 1e-6)
    selected = np.where(s >= cfg.toe_merge_full)[0]
    if selected.size == 0:
        return 1
    for index in selected:
        if _ordered_section_area_yz(sections[int(index)]) > 1e-10:
            return 1
    return 0


def _ordered_section_area_yz(section: np.ndarray) -> float:
    points = np.asarray(section, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3:
        return 0.0
    z = points[:, 2].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return float(abs(0.5 * np.sum(z * np.roll(y, -1) - np.roll(z, -1) * y)))


def _build_surface_offset_mesh(
    foot_mesh: MeshData,
    support: _SupportProfile,
    sampler: _FootbedSampler,
    profiles: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    cfg: PseudoLastConfig,
    support_length: float,
) -> MeshData:
    """Inflate the aligned SUPR surface while keeping it constrained by the shoe footprint."""

    vertices = np.asarray(foot_mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(foot_mesh.faces, dtype=np.int64).copy()
    normals = _oriented_vertex_normals(MeshData(vertices=vertices, faces=faces))

    s = np.clip((vertices[:, 0] - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6), 0.0, 1.0)
    footbed_y = sampler.sample(vertices[:, 0], vertices[:, 2])
    h = footbed_y - vertices[:, 1]

    heel_w = 1.0 - _smoothstep(0.12, 0.32, s)
    fore_w = _smoothstep(0.55, 0.85, s)
    mid_w = np.clip(1.0 - heel_w - 0.35 * fore_w, 0.0, 1.0)
    body_clearance_ratio = (
        cfg.heel_clearance_ratio * heel_w
        + cfg.midfoot_clearance_ratio * mid_w
        + cfg.forefoot_clearance_ratio * fore_w
    )
    height_weight = _smoothstep(0.025 * support_length, 0.090 * support_length, h)
    clearance = support_length * (
        cfg.bottom_clearance_ratio
        + (body_clearance_ratio - cfg.bottom_clearance_ratio) * height_weight
        + cfg.top_clearance_ratio * _smoothstep(0.20 * support_length, 0.38 * support_length, h)
    )

    deformed = vertices + normals * clearance[:, None].astype(np.float32)

    toe_mu = _smoothstep(cfg.toe_merge_start, 1.0, s)
    deformed[:, 0] += (toe_mu * toe_mu * cfg.toe_allowance_ratio * support_length).astype(np.float32)

    deformed = _apply_support_clamp(deformed, support, cfg, support_length)
    deformed = _apply_footbed_bottom_clamp(deformed, sampler)
    deformed = _apply_toe_box_envelope(deformed, support, sampler, profiles, x_grid, cfg, support_length)
    deformed = _apply_support_clamp(deformed, support, cfg, support_length)
    deformed = _apply_footbed_bottom_clamp(deformed, sampler)

    bottom_mask = h <= cfg.bottom_fixed_band_ratio * support_length
    mesh = MeshData(vertices=deformed.astype(np.float32), faces=faces)
    mesh = _smooth_surface_vertices(mesh, bottom_mask, cfg)
    mesh = _cap_boundary_loops(mesh)
    return mesh


def _oriented_vertex_normals(mesh: MeshData) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = vertices.mean(axis=0)
    normals = np.zeros_like(vertices, dtype=np.float32)
    for tri in faces:
        v0, v1, v2 = vertices[tri]
        normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(normal) < 1e-12:
            continue
        centroid = (v0 + v1 + v2) / 3.0
        if float(np.dot(normal, centroid - center)) < 0.0:
            normal = -normal
        normals[tri] += normal.astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = vertices - center[None, :]
    fallback_lengths = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = fallback / np.maximum(fallback_lengths, 1e-8)
    normals = np.where(lengths > 1e-8, normals / np.maximum(lengths, 1e-8), fallback)
    outward = np.sum(normals * (vertices - center[None, :]), axis=1) < 0.0
    normals[outward] *= -1.0
    return normals.astype(np.float32)


def _apply_support_clamp(
    vertices: np.ndarray,
    support: _SupportProfile,
    cfg: PseudoLastConfig,
    support_length: float,
) -> np.ndarray:
    output = np.asarray(vertices, dtype=np.float32).copy()
    x = np.clip(output[:, 0], float(support.x[0]), float(support.x[-1]))
    center = support.center(x)
    left = support.left(x)
    right = support.right(x)
    inner_left = center + cfg.eta_s * (left - center)
    inner_right = center + cfg.eta_s * (right - center)
    margin = cfg.support_z_margin_ratio * support_length
    output[:, 2] = np.clip(output[:, 2], inner_left + margin, inner_right - margin)
    return output.astype(np.float32)


def _apply_footbed_bottom_clamp(vertices: np.ndarray, sampler: _FootbedSampler) -> np.ndarray:
    output = np.asarray(vertices, dtype=np.float32).copy()
    footbed_y = sampler.sample(output[:, 0], output[:, 2])
    output[:, 1] = np.minimum(output[:, 1], footbed_y)
    return output.astype(np.float32)


def _apply_toe_box_envelope(
    vertices: np.ndarray,
    support: _SupportProfile,
    sampler: _FootbedSampler,
    profiles: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    cfg: PseudoLastConfig,
    support_length: float,
) -> np.ndarray:
    output = np.asarray(vertices, dtype=np.float32).copy()
    s = np.clip((output[:, 0] - x_grid[0]) / max(float(x_grid[-1] - x_grid[0]), 1e-6), 0.0, 1.0)
    mu = _smoothstep(cfg.toe_merge_start, cfg.toe_merge_full, s) * cfg.toe_box_top_blend
    if np.max(mu) <= 0.0:
        return output

    x = np.clip(output[:, 0], float(support.x[0]), float(support.x[-1]))
    support_center = support.center(x)
    support_left = support.left(x)
    support_right = support.right(x)
    inner_left = support_center + cfg.eta_s * (support_left - support_center)
    inner_right = support_center + cfg.eta_s * (support_right - support_center)
    half = np.maximum(0.5 * (inner_right - inner_left), 1e-5)
    center = 0.5 * (inner_left + inner_right)

    height = np.interp(x, x_grid, profiles["height"]).astype(np.float32)
    footbed_y = sampler.sample(output[:, 0], output[:, 2])
    r = np.clip(np.abs((output[:, 2] - center) / half), 0.0, 1.0)
    top_height = height * np.power(np.maximum(1.0 - np.power(r, cfg.superellipse_p), 0.0), 1.0 / cfg.superellipse_q)
    envelope_y = footbed_y - top_height
    current_h = footbed_y - output[:, 1]
    top_region = _smoothstep(0.25 * support_length, 0.55 * support_length, current_h)
    blend = mu * top_region
    output[:, 1] = output[:, 1] * (1.0 - blend) + envelope_y * blend

    side_blend = _smoothstep(cfg.toe_merge_start, cfg.toe_merge_full, s) * cfg.toe_box_side_blend
    z_target = center + np.clip(output[:, 2] - center, -half, half)
    output[:, 2] = output[:, 2] * (1.0 - side_blend) + z_target * side_blend
    return output.astype(np.float32)


def _smooth_surface_vertices(mesh: MeshData, bottom_mask: np.ndarray, cfg: PseudoLastConfig) -> MeshData:
    if cfg.smooth_iterations <= 0 or cfg.smooth_step <= 0.0:
        return mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    adjacency = [[] for _ in range(vertices.shape[0])]
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
    movable = ~np.asarray(bottom_mask, dtype=bool)
    for _ in range(max(1, cfg.smooth_iterations // 2)):
        updated = vertices.copy()
        for idx, neighbors in enumerate(adjacency):
            if not movable[idx] or not neighbors:
                continue
            mean_neighbor = vertices[np.asarray(neighbors, dtype=np.int64)].mean(axis=0)
            updated[idx] = vertices[idx] * (1.0 - cfg.smooth_step) + mean_neighbor * cfg.smooth_step
        vertices = updated
    return MeshData(vertices=vertices.astype(np.float32), faces=faces)


def _cap_boundary_loops(mesh: MeshData) -> MeshData:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = find_boundary_loops(faces)
    if not loops:
        return mesh
    new_vertices = [vertices]
    new_faces = [faces]
    mesh_center = vertices.mean(axis=0)
    for loop in loops:
        if len(loop) < 3:
            continue
        loop_indices = np.asarray(loop, dtype=np.int64)
        loop_vertices = vertices[loop_indices]
        center = loop_vertices.mean(axis=0, keepdims=True).astype(np.float32)
        center_idx = vertices.shape[0] + sum(part.shape[0] for part in new_vertices[1:])
        cap_faces = []
        for i in range(loop_indices.size):
            a = int(loop_indices[i])
            b = int(loop_indices[(i + 1) % loop_indices.size])
            tri = [center_idx, a, b]
            normal = np.cross(vertices[a] - center[0], vertices[b] - center[0])
            if float(np.dot(normal, center[0] - mesh_center)) < 0.0:
                tri = [center_idx, b, a]
            cap_faces.append(tri)
        new_vertices.append(center)
        new_faces.append(np.asarray(cap_faces, dtype=np.int64))
    return MeshData(
        vertices=np.concatenate(new_vertices, axis=0).astype(np.float32),
        faces=np.concatenate(new_faces, axis=0).astype(np.int64),
    )


def _sample_surface_debug_sections(vertices: np.ndarray, x_grid: np.ndarray, n_theta: int) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32)
    sections = []
    half_width = max(float(x_grid[-1] - x_grid[0]) / max(x_grid.size - 1, 1) * 2.0, 1e-4)
    angles = np.linspace(-np.pi, np.pi, n_theta, endpoint=False, dtype=np.float32)
    for x in x_grid:
        mask = np.abs(points[:, 0] - float(x)) <= half_width
        pts = points[mask]
        if pts.shape[0] < 8:
            order = np.argsort(np.abs(points[:, 0] - float(x)))
            pts = points[order[: min(32, points.shape[0])]]
        yz = pts[:, [2, 1]]
        center = yz.mean(axis=0)
        local = yz - center[None, :]
        theta = np.arctan2(local[:, 1], local[:, 0])
        radius = np.linalg.norm(local, axis=1)
        section = []
        for a in angles:
            delta = np.angle(np.exp(1j * (theta - float(a))))
            weights = np.exp(-(delta * delta) / (2.0 * 0.35 * 0.35))
            r = float(np.sum(weights * radius) / max(float(np.sum(weights)), 1e-8))
            z = center[0] + r * np.cos(float(a))
            y = center[1] + r * np.sin(float(a))
            section.append([float(x), y, z])
        sections.append(np.asarray(section, dtype=np.float32))
    return np.stack(sections, axis=0).astype(np.float32)


def _make_single_section(
    x: float,
    s: float,
    left_z: float,
    right_z: float,
    height: float,
    plantar_lift: float,
    plantar_lift_curve: Optional[np.ndarray],
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
    support_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    half = cfg.n_theta // 2
    bottom_z = np.linspace(left_z, right_z, half, dtype=np.float32)
    top_z = np.linspace(right_z, left_z, cfg.n_theta - half + 2, dtype=np.float32)[1:-1]
    center = 0.5 * (left_z + right_z)
    half_width = max(0.5 * (right_z - left_z), 1e-5)

    bottom_y = _section_bottom_y(x, s, bottom_z, center, half_width, plantar_lift, plantar_lift_curve, sampler, cfg)
    bottom_points = np.stack([np.full_like(bottom_z, x), bottom_y, bottom_z], axis=1)

    top_bottom_y = _section_bottom_y(x, s, top_z, center, half_width, plantar_lift, plantar_lift_curve, sampler, cfg)
    r = np.clip(np.abs((top_z - center) / half_width), 0.0, 1.0)
    mu = _smoothstep(cfg.toe_merge_start, cfg.toe_merge_full, np.asarray([s], dtype=np.float32))[0]
    p = cfg.superellipse_p + 0.8 * float(mu)
    q = cfg.superellipse_q
    h_top = max(height, cfg.min_height_ratio * support_length) * np.power(np.maximum(1.0 - np.power(r, p), 0.0), 1.0 / q)
    top_y = top_bottom_y - h_top
    top_points = np.stack([np.full_like(top_z, x), top_y.astype(np.float32), top_z], axis=1)

    section = np.concatenate([bottom_points, top_points], axis=0)
    section = _apply_heel_hold(section, s, center, height, cfg)
    return section.astype(np.float32), bottom_points.astype(np.float32)


def _section_bottom_y(
    x: float,
    s: float,
    z: np.ndarray,
    center: float,
    half_width: float,
    plantar_lift: float,
    plantar_lift_curve: Optional[np.ndarray],
    sampler: _FootbedSampler,
    cfg: PseudoLastConfig,
) -> np.ndarray:
    footbed_y = sampler.sample(np.full_like(z, x, dtype=np.float32), z)
    lateral = np.maximum(1.0 - np.abs((z - center) / max(half_width, 1e-5)), 0.0) ** 1.5
    arch = cfg.arch_strength * np.exp(-((s - cfg.arch_center) ** 2) / (2.0 * cfg.arch_sigma * cfg.arch_sigma))
    scalar_lift = max(plantar_lift, 0.0) * lateral
    curve_lift = _sample_plantar_lift_curve(plantar_lift_curve, z, center, half_width)
    if plantar_lift_curve is None:
        lift = scalar_lift
    else:
        blend = float(np.clip(cfg.plantar_curve_blend, 0.0, 1.0))
        lift = blend * curve_lift + (1.0 - blend) * scalar_lift
    return (footbed_y - arch * np.maximum(lift, 0.0)).astype(np.float32)


def _apply_heel_hold(section: np.ndarray, s: float, center: float, height: float, cfg: PseudoLastConfig) -> np.ndarray:
    if s >= cfg.heel_hold_end or height <= 1e-6 or cfg.heel_hold_strength <= 0.0:
        return section
    output = section.copy()
    y_bottom = np.max(output[:, 1])
    tau = np.clip((y_bottom - output[:, 1]) / max(height, 1e-6), 0.0, 1.0)
    active = 1.0 - _smoothstep(0.0, cfg.heel_hold_end, np.asarray([s], dtype=np.float32))[0]
    shrink = cfg.heel_hold_strength * active * np.power(tau, cfg.heel_hold_power)
    output[:, 2] = center + (output[:, 2] - center) * (1.0 - shrink)
    return output.astype(np.float32)


def _loft_sections(sections: np.ndarray) -> MeshData:
    n_sections, n_theta, _ = sections.shape
    vertices = sections.reshape(-1, 3).astype(np.float32)
    faces = []
    for i in range(n_sections - 1):
        base0 = i * n_theta
        base1 = (i + 1) * n_theta
        for j in range(n_theta):
            a = base0 + j
            b = base0 + (j + 1) % n_theta
            c = base1 + j
            d = base1 + (j + 1) % n_theta
            faces.append([a, c, b])
            faces.append([b, c, d])

    heel_center = vertices.shape[0]
    toe_center = vertices.shape[0] + 1
    vertices = np.concatenate(
        [
            vertices,
            sections[0].mean(axis=0, keepdims=True),
            sections[-1].mean(axis=0, keepdims=True),
        ],
        axis=0,
    )
    for j in range(n_theta):
        faces.append([heel_center, (j + 1) % n_theta, j])
        toe_base = (n_sections - 1) * n_theta
        faces.append([toe_center, toe_base + j, toe_base + (j + 1) % n_theta])
    return MeshData(vertices=vertices.astype(np.float32), faces=np.asarray(faces, dtype=np.int64))


def _loft_bottom_sections(bottom_sections: np.ndarray) -> MeshData:
    n_sections, n_z, _ = bottom_sections.shape
    vertices = bottom_sections.reshape(-1, 3).astype(np.float32)
    faces = []
    for i in range(n_sections - 1):
        base0 = i * n_z
        base1 = (i + 1) * n_z
        for j in range(n_z - 1):
            a = base0 + j
            b = base0 + j + 1
            c = base1 + j
            d = base1 + j + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return MeshData(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _smooth_nonbottom_vertices(mesh: MeshData, n_sections: int, n_theta: int, cfg: PseudoLastConfig) -> MeshData:
    if cfg.smooth_iterations <= 0 or cfg.smooth_step <= 0.0:
        return mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    adjacency = [[] for _ in range(vertices.shape[0])]
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency[a].append(b)
            adjacency[b].append(a)
    bottom_count = n_theta // 2
    movable = np.ones((vertices.shape[0],), dtype=bool)
    for i in range(n_sections):
        movable[i * n_theta : i * n_theta + bottom_count] = False
    movable[n_sections * n_theta :] = False
    for _ in range(cfg.smooth_iterations):
        new_vertices = vertices.copy()
        for idx, neighbors in enumerate(adjacency):
            if not movable[idx] or not neighbors:
                continue
            mean_neighbor = vertices[np.asarray(neighbors, dtype=np.int64)].mean(axis=0)
            new_vertices[idx] = vertices[idx] * (1.0 - cfg.smooth_step) + mean_neighbor * cfg.smooth_step
        vertices = new_vertices
    return MeshData(vertices=vertices.astype(np.float32), faces=faces)


def _fill_nan_profile(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if valid.all():
        return values
    if valid.sum() >= 2:
        return np.interp(x, x[valid], values[valid]).astype(np.float32)
    if valid.sum() == 1:
        return np.full_like(values, float(values[valid][0]), dtype=np.float32)
    return np.zeros_like(values, dtype=np.float32)


def _smooth_profile(values: np.ndarray, smooth_lambda: float) -> np.ndarray:
    y = np.asarray(values, dtype=np.float32)
    if smooth_lambda <= 0.0 or y.size < 4:
        return y.copy()
    n = y.size
    d2 = np.zeros((n - 2, n), dtype=np.float32)
    for i in range(n - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    a = np.eye(n, dtype=np.float32) + np.float32(smooth_lambda) * (d2.T @ d2)
    return np.linalg.solve(a, y).astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float32) - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _count_boundary_edges(faces: np.ndarray) -> int:
    counts: Dict[Tuple[int, int], int] = {}
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for count in counts.values() if count == 1)


__all__ = [
    "PseudoLastConfig",
    "PseudoLastResult",
    "build_pseudo_last",
    "build_pseudo_last_from_paths",
    "save_pseudo_last_artifacts",
    "plot_section_overlays",
    "plot_pseudo_last_overlay",
]
