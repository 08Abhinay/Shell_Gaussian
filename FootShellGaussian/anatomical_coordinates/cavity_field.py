"""Bake the existing cavity clearance semantics into a grid_sample-able field.

``foot_prior.cavity.CavityEvaluator`` measures containment with two directional
ray casts per sample rather than a closed-volume SDF, because a shoe is open at
the collar and any parity-based inside/outside test would be wrong there. Read
literally, its per-sample loops are far too slow to rasterize a volume.

The decomposition that makes it cheap: both ray families have origins and
directions that depend on only two coordinates, so each clearance is an
algebraic expression in a *two-dimensional* height field.

  upper: origin is the footbed point below the sample, direction -Y (up).
         The hit therefore depends on (x, z) alone, giving a ceiling height
         ``ceiling_y(x, z)`` and ``clearance = sample_y - ceiling_y``.

  side:  origin is on the centerline at the sample's (x, y), direction +/-Z.
         The hit depends on (x, y) and the side, giving wall positions
         ``wall_z_neg(x, y)`` and ``wall_z_pos(x, y)`` and
         ``clearance = |wall_z - center_z| - |sample_z - center_z|``.

So three 2-D rasterizations fully determine the 3-D field, which is then
evaluated analytically on the lattice. Sign convention matches the source:
positive is inside the admissible cavity, negative is outside, and NaN marks
open space where no boundary exists (never invented as 0 or as a large value).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foot_prior.mesh import TriangleMesh


# Matches ``CavityEvaluator._upper_clearances`` / ``_side_clearances``, which
# reject hits closer than 8 * numerical_tolerance to avoid re-hitting the
# surface the ray starts on.
RAY_EPSILON_MULTIPLIER = 8.0


def coordinate_tolerance(*meshes: TriangleMesh) -> float:
    """Reproduce ``foot_prior.cavity._coordinate_tolerance``."""

    maximum = max(float(np.max(np.abs(mesh.vertices))) for mesh in meshes)
    return float(np.spacing(np.float32(max(1.0, maximum))))


def _raster_extreme(
    triangles: torch.Tensor,
    axis_u: int,
    axis_v: int,
    axis_w: int,
    grid_u: torch.Tensor,
    grid_v: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float,
    mode: str,
    chunk: int = 4096,
) -> torch.Tensor:
    """Rasterize triangles along one axis and reduce toward ``reference``.

    Projects each triangle onto the (u, v) plane, finds the grid columns it
    covers, interpolates its ``w`` coordinate there, and keeps the extreme value
    on one side of ``reference``:

      ``mode="below"``  -> largest w strictly less than reference - epsilon
      ``mode="above"``  -> smallest w strictly greater than reference + epsilon

    Returns a (U, V) field of NaN where the triangles cover nothing, which is
    exactly the "no boundary found" case the source code leaves unconstrained.
    """

    if mode not in {"below", "above"}:
        raise ValueError("mode must be 'below' or 'above'")
    device = triangles.device
    u_count = int(grid_u.numel())
    v_count = int(grid_v.numel())
    if reference.shape != (u_count, v_count):
        raise ValueError("reference must match the (U, V) lattice")
    sign = -1.0 if mode == "below" else 1.0
    result = torch.full(
        (u_count, v_count), float("inf"), dtype=torch.float64, device=device
    )

    step_u = float(grid_u[1] - grid_u[0]) if u_count > 1 else 1.0
    step_v = float(grid_v[1] - grid_v[0]) if v_count > 1 else 1.0
    origin_u = float(grid_u[0])
    origin_v = float(grid_v[0])

    planar = triangles[:, :, (axis_u, axis_v)]
    heights = triangles[:, :, axis_w]
    first = planar[:, 0]
    edge_a = planar[:, 1] - first
    edge_b = planar[:, 2] - first
    determinant = edge_a[:, 0] * edge_b[:, 1] - edge_b[:, 0] * edge_a[:, 1]
    usable = determinant.abs() > 1e-18

    lower_u = planar[:, :, 0].min(dim=1).values
    upper_u = planar[:, :, 0].max(dim=1).values
    lower_v = planar[:, :, 1].min(dim=1).values
    upper_v = planar[:, :, 1].max(dim=1).values
    first_u = torch.clamp(
        torch.ceil((lower_u - origin_u) / step_u).long(), 0, u_count - 1
    )
    last_u = torch.clamp(
        torch.floor((upper_u - origin_u) / step_u).long(), 0, u_count - 1
    )
    first_v = torch.clamp(
        torch.ceil((lower_v - origin_v) / step_v).long(), 0, v_count - 1
    )
    last_v = torch.clamp(
        torch.floor((upper_v - origin_v) / step_v).long(), 0, v_count - 1
    )
    spans_u = (last_u - first_u + 1).clamp(min=0)
    spans_v = (last_v - first_v + 1).clamp(min=0)
    active = torch.nonzero(usable & (spans_u > 0) & (spans_v > 0), as_tuple=False).squeeze(1)

    for start in range(0, int(active.numel()), chunk):
        block = active[start : start + chunk]
        if not block.numel():
            continue
        width = int(spans_u[block].max())
        height = int(spans_v[block].max())
        offset_u = torch.arange(width, device=device)
        offset_v = torch.arange(height, device=device)
        index_u = first_u[block][:, None] + offset_u[None, :]
        index_v = first_v[block][:, None] + offset_v[None, :]
        inside_u = offset_u[None, :] < spans_u[block][:, None]
        inside_v = offset_v[None, :] < spans_v[block][:, None]
        index_u = index_u.clamp(max=u_count - 1)
        index_v = index_v.clamp(max=v_count - 1)

        point_u = grid_u[index_u][:, :, None].expand(-1, -1, height)
        point_v = grid_v[index_v][:, None, :].expand(-1, width, -1)
        valid = inside_u[:, :, None] & inside_v[:, None, :]

        relative_u = point_u - first[block][:, 0][:, None, None]
        relative_v = point_v - first[block][:, 1][:, None, None]
        det = determinant[block][:, None, None]
        weight_a = (
            relative_u * edge_b[block][:, 1][:, None, None]
            - edge_b[block][:, 0][:, None, None] * relative_v
        ) / det
        weight_b = (
            edge_a[block][:, 0][:, None, None] * relative_v
            - relative_u * edge_a[block][:, 1][:, None, None]
        ) / det
        weight_first = 1.0 - weight_a - weight_b
        tolerance = 1e-10
        covered = (
            valid
            & (weight_first >= -tolerance)
            & (weight_a >= -tolerance)
            & (weight_b >= -tolerance)
        )
        if not bool(covered.any()):
            continue
        value = (
            weight_first * heights[block][:, 0][:, None, None]
            + weight_a * heights[block][:, 1][:, None, None]
            + weight_b * heights[block][:, 2][:, None, None]
        )
        reference_here = reference[index_u[:, :, None], index_v[:, None, :]]
        keep = covered & (sign * (value - reference_here) > epsilon)
        # Store the *unsigned* travel from the reference to the surface, so that
        # ``amin`` picks the first hit along the ray. ``sign * (value -
        # reference)`` is positive exactly for the kept entries in both modes:
        # a "below" search travels toward smaller w, an "above" search toward
        # larger w. Reducing the signed difference instead would select the
        # farthest surface, which is only indistinguishable from the nearest
        # when a single layer exists.
        distance = torch.where(
            keep, sign * (value - reference_here), torch.inf
        )
        flat_index = index_u[:, :, None] * v_count + index_v[:, None, :]
        result.view(-1).scatter_reduce_(
            0,
            flat_index.reshape(-1),
            distance.reshape(-1).double(),
            reduce="amin",
            include_self=True,
        )

    reference_flat = reference.reshape(-1)
    output = torch.where(
        torch.isfinite(result.view(-1)),
        reference_flat + sign * result.view(-1),
        torch.nan,
    )
    return output.view(u_count, v_count)


def _sample_height_field(
    triangles: torch.Tensor,
    grid_u: torch.Tensor,
    grid_v: torch.Tensor,
    axis_u: int = 0,
    axis_v: int = 2,
    axis_w: int = 1,
    reduce: str = "amin",
) -> torch.Tensor:
    """Rasterize triangle ``w`` onto a (u, v) lattice, NaN where uncovered.

    ``reduce="amin"`` mirrors ``mesh.sample_triangle_mesh_y``, which keeps the
    opening-nearest (smallest Y) surface when several layers overlap.
    """

    device = triangles.device
    u_count, v_count = int(grid_u.numel()), int(grid_v.numel())
    sign = 1.0 if reduce == "amin" else -1.0
    reference = torch.zeros((u_count, v_count), dtype=torch.float64, device=device)
    mode = "above" if reduce == "amin" else "below"
    return _raster_extreme(
        triangles,
        axis_u,
        axis_v,
        axis_w,
        grid_u,
        grid_v,
        reference - sign * 1e9,
        epsilon=-torch.inf,
        mode=mode,
    )


@dataclass(frozen=True)
class ShoeCavityField:
    """A baked signed-clearance volume plus the metadata to query it."""

    clearance: torch.Tensor  # (1, 1, D, H, W) float32, +inside, -outside
    valid: torch.Tensor  # (1, 1, D, H, W) float32 in {0, 1}
    distance: torch.Tensor  # (1, 1, D, H, W) signed distance to the obstacles
    lower: torch.Tensor  # (3,) world-space minimum corner (x, y, z)
    upper: torch.Tensor  # (3,) world-space maximum corner
    spacing: torch.Tensor  # (3,) cell size
    footbed_x: torch.Tensor  # (Nx,) lattice used by the footbed height field
    footbed_z: torch.Tensor  # (Nz,)
    footbed_y: torch.Tensor  # (Nx, Nz) NaN where the footbed does not reach
    open_above: torch.Tensor  # (Nx, Nz) 1 where the upper has a real opening
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.clearance.shape[2:])

    def to(self, device: torch.device | str) -> "ShoeCavityField":
        moved = {
            name: getattr(self, name).to(device)
            for name in (
                "clearance",
                "valid",
                "distance",
                "lower",
                "upper",
                "spacing",
                "footbed_x",
                "footbed_z",
                "footbed_y",
                "open_above",
            )
        }
        return ShoeCavityField(metadata=self.metadata, **moved)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clearance": self.clearance.cpu(),
                "valid": self.valid.cpu(),
                "distance": self.distance.cpu(),
                "lower": self.lower.cpu(),
                "upper": self.upper.cpu(),
                "spacing": self.spacing.cpu(),
                "footbed_x": self.footbed_x.cpu(),
                "footbed_z": self.footbed_z.cpu(),
                "footbed_y": self.footbed_y.cpu(),
                "open_above": self.open_above.cpu(),
                "metadata": self.metadata,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ShoeCavityField":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        return cls(**payload)


def sample_field(
    field: torch.Tensor,
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    padding_mode: str = "border",
    extend_outside: bool = False,
) -> torch.Tensor:
    """Trilinearly sample a (B, C, D, H, W) volume at (B, N, 3) world points.

    ``grid_sample`` reads its last axis as (x, y, z) indexing (W, H, D), so the
    world axes are reversed into (z, y, x) before normalization. The volume is
    stored with D over world X, H over world Y and W over world Z, which makes
    the reversal the identity mapping back onto (X, Y, Z).

    ``extend_outside`` continues a clearance field beyond the baked box. Border
    padding alone reports the boundary value with *zero* gradient, which is the
    worst possible behaviour for a vertex that has escaped: it is the moment a
    restoring force matters most. Subtracting the distance back to the box is a
    conservative linear extension - a point that far outside is at least that
    far past whatever bound the boundary reported - and it restores a gradient
    that points back inside. Only meaningful for signed-clearance channels, so
    it stays off for the validity mask.
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B, N, 3)")
    batch = points.shape[0]
    if field.shape[0] == 1 and batch > 1:
        field = field.expand(batch, -1, -1, -1, -1)
    # A single-shoe field carries one (3,) box; a batched one carries (B, 3).
    # Give both a broadcastable (.., 1, 3) shape against (B, N, 3) points.
    if lower.ndim == 2:
        lower = lower[:, None, :]
        upper = upper[:, None, :]
    span = (upper - lower).clamp(min=1e-12)
    unit = (points - lower) / span  # [0, 1] over world (x, y, z)
    normalized = 2.0 * unit - 1.0
    # (x, y, z) world -> grid_sample's (W, H, D) order, i.e. reverse.
    grid = normalized.flip(-1).view(batch, points.shape[1], 1, 1, 3)
    sampled = torch.nn.functional.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    values = sampled[:, :, :, 0, 0].permute(0, 2, 1)
    if extend_outside:
        clamped = torch.maximum(torch.minimum(points, upper), lower)
        outside = (points - clamped).norm(dim=-1)
        values = values - outside[:, :, None]
    return values


def inside_domain(
    points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Return (B, N) mask of points lying inside the baked domain."""

    return ((points >= lower) & (points <= upper)).all(dim=-1)


def _obstacle_triangles(
    shoe_mesh: TriangleMesh,
    footbed_source_face_indices: np.ndarray,
    tolerance: float,
    device: torch.device,
) -> torch.Tensor:
    """Split off the non-footbed faces exactly as ``CavityEvaluator`` does."""

    footbed_faces = np.unique(np.asarray(footbed_source_face_indices, dtype=np.int64))
    obstacle_faces = np.setdiff1d(
        np.arange(len(shoe_mesh.faces), dtype=np.int64),
        footbed_faces,
        assume_unique=True,
    )
    if len(obstacle_faces) == 0:
        raise ValueError("shoe contains no non-footbed obstacle faces")
    triangles = shoe_mesh.vertices[shoe_mesh.faces[obstacle_faces]]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    triangles = triangles[double_area > tolerance * tolerance]
    if len(triangles) == 0:
        raise ValueError("shoe contains no nondegenerate obstacle faces")
    return torch.as_tensor(triangles, dtype=torch.float64, device=device)


def build_cavity_field(
    shoe_mesh: TriangleMesh,
    footbed_mesh: TriangleMesh,
    footbed_source_face_indices: np.ndarray,
    normalized_centerline_xz: np.ndarray,
    foot_bounds: np.ndarray,
    target_spacing: float = 1.0 / 262.5,
    margin: float = 0.06,
    device: str = "cuda",
    longitudinal_sign: bool = True,
) -> ShoeCavityField:
    """Bake the signed cavity clearance onto a regular lattice.

    ``foot_bounds`` is a (2, 3) min/max box for the foot under its deterministic
    initialization; the domain is that box grown by ``margin`` so the optimizer
    has room to move without ever leaving the baked volume. ``target_spacing``
    defaults to one millimetre expressed in normalized shoe units.

    ``longitudinal_sign`` exists only so the +/-X ray family can be ablated.
    Turning it off restores the older, X-blind sign rule, which signs a sample
    behind the heel counter positive. It is never a tuning knob.
    """

    torch_device = torch.device(device)
    tolerance = coordinate_tolerance(shoe_mesh, footbed_mesh)
    epsilon = RAY_EPSILON_MULTIPLIER * tolerance
    triangles = _obstacle_triangles(
        shoe_mesh, footbed_source_face_indices, tolerance, torch_device
    )
    footbed_triangles = torch.as_tensor(
        footbed_mesh.vertices[footbed_mesh.faces],
        dtype=torch.float64,
        device=torch_device,
    )

    centerline = np.asarray(normalized_centerline_xz, dtype=np.float64)
    bounds = np.asarray(foot_bounds, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError("foot_bounds must have shape (2, 3)")
    lower = bounds[0] - margin
    upper = bounds[1] + margin
    # The footbed must stay inside the domain or support has no reference.
    lower[1] = min(lower[1], float(footbed_mesh.bounds[0, 1]) - 0.01)
    upper[1] = max(upper[1], float(footbed_mesh.bounds[1, 1]) + 0.01)

    counts = np.maximum(
        8, np.ceil((upper - lower) / float(target_spacing)).astype(np.int64) + 1
    )
    axes = [
        torch.linspace(
            float(lower[axis]),
            float(upper[axis]),
            int(counts[axis]),
            dtype=torch.float64,
            device=torch_device,
        )
        for axis in range(3)
    ]
    grid_x, grid_y, grid_z = axes

    # --- the three 2-D rasterizations -----------------------------------
    footbed_y = _sample_height_field(
        footbed_triangles, grid_x, grid_z, 0, 2, 1, reduce="amin"
    )
    # Upward ray from the footbed: shoe +Y points down, so the ceiling is the
    # largest Y strictly below the footbed height.
    ceiling_reference = torch.nan_to_num(footbed_y, nan=-1e9)
    ceiling_y = _raster_extreme(
        triangles, 0, 2, 1, grid_x, grid_z, ceiling_reference, epsilon, "below"
    )
    # A column that carries a footbed but meets no surface on the way up is a
    # genuine opening in the upper: the ankle collar, or the open top of a
    # sandal. That is the only place the foot may legitimately leave the
    # cavity, and it is read off the shoe itself rather than from an
    # anatomical bounding box. Columns with no footbed - behind the heel
    # counter, beyond the toe - are emphatically *not* openings, which is the
    # distinction a joint-space box cannot make.
    open_above = torch.isfinite(footbed_y) & torch.isnan(ceiling_y)
    ceiling_y = torch.where(torch.isnan(footbed_y), torch.nan, ceiling_y)

    center_z = torch.as_tensor(
        np.interp(
            np.clip(
                grid_x.cpu().numpy(), centerline[0, 0], centerline[-1, 0]
            ),
            centerline[:, 0],
            centerline[:, 1],
        ),
        dtype=torch.float64,
        device=torch_device,
    )
    covered = (grid_x >= centerline[0, 0]) & (grid_x <= centerline[-1, 0])
    center_reference = center_z[:, None].expand(-1, int(grid_y.numel()))
    wall_negative = _raster_extreme(
        triangles, 0, 1, 2, grid_x, grid_y, center_reference, epsilon, "below"
    )
    wall_positive = _raster_extreme(
        triangles, 0, 1, 2, grid_x, grid_y, center_reference, epsilon, "above"
    )

    # --- assemble the 3-D field analytically -----------------------------
    upper_clearance = grid_y[None, :, None] - ceiling_y[:, None, :]

    offset = grid_z[None, None, :] - center_z[:, None, None]
    distance_negative = (center_reference - wall_negative).abs()[:, :, None]
    distance_positive = (wall_positive - center_reference).abs()[:, :, None]
    near_center = offset.abs() <= tolerance
    chosen = torch.where(offset > 0.0, distance_positive, distance_negative)
    nearer = torch.minimum(
        torch.nan_to_num(distance_negative, nan=torch.inf),
        torch.nan_to_num(distance_positive, nan=torch.inf),
    )
    chosen = torch.where(near_center, nearer, chosen)
    chosen = torch.where(torch.isinf(chosen), torch.nan, chosen)
    side_clearance = chosen - offset.abs()
    side_clearance = torch.where(
        covered[:, None, None], side_clearance, torch.nan
    )

    stacked = torch.stack((upper_clearance, side_clearance), dim=0)
    finite = torch.isfinite(stacked)
    combined = torch.where(
        finite.any(dim=0),
        torch.where(finite, stacked, torch.inf).min(dim=0).values,
        torch.nan,
    )
    valid = torch.isfinite(combined)
    clearance = torch.where(valid, combined, torch.zeros_like(combined))

    # --- the longitudinal ray family, used for the sign only --------------
    # The source casts nothing along X, so the heel counter and the toe box are
    # invisible to the clearance above. That blind spot is not merely a missing
    # constraint: `outside` below is derived from `valid`, so a sample behind
    # the heel counter - where no ceiling and no side wall is found - was
    # signed *positive*, and the positive value grew with depth. Measured on
    # three shoes, a point 20 mm behind the counter reported +1.8 to +9.2 mm.
    # The optimizer was being told that leaving the shoe backwards took it
    # deeper inside, so the counter was only a thin barrier to tunnel through
    # with no restoring force on the far side.
    #
    # The same 2-D decomposition closes it. The ray origin is the footbed's
    # longitudinal midpoint, so the hit depends on (y, z) alone and gives
    # ``wall_x_neg(y, z)`` and ``wall_x_pos(y, z)``.
    #
    # This is used ONLY to sign the distance channel. ``clearance`` and
    # ``valid`` above are left exactly as the source defines them, because that
    # correspondence is what they are validated against.
    center_x = 0.5 * float(centerline[0, 0] + centerline[-1, 0])
    x_reference = torch.full(
        (int(grid_y.numel()), int(grid_z.numel())),
        center_x,
        dtype=torch.float64,
        device=torch_device,
    )
    wall_x_negative = _raster_extreme(
        triangles, 1, 2, 0, grid_y, grid_z, x_reference, epsilon, "below"
    )
    wall_x_positive = _raster_extreme(
        triangles, 1, 2, 0, grid_y, grid_z, x_reference, epsilon, "above"
    )
    offset_x = (grid_x - center_x)[:, None, None]
    span_negative = (x_reference - wall_x_negative).abs()[None]
    span_positive = (wall_x_positive - x_reference).abs()[None]
    near_center_x = offset_x.abs() <= tolerance
    chosen_x = torch.where(offset_x > 0.0, span_positive, span_negative)
    nearer_x = torch.minimum(
        torch.nan_to_num(span_negative, nan=torch.inf),
        torch.nan_to_num(span_positive, nan=torch.inf),
    )
    chosen_x = torch.where(near_center_x, nearer_x, chosen_x)
    chosen_x = torch.where(torch.isinf(chosen_x), torch.nan, chosen_x)
    longitudinal = chosen_x - offset_x.abs()
    longitudinal_outside = torch.isfinite(longitudinal) & (longitudinal < 0.0)
    del chosen_x, nearer_x, span_negative, span_positive, longitudinal

    # --- a true signed distance to the obstacle surface -------------------
    # The directional clearance above probes only +/-Y and +/-Z. Nothing casts
    # along X, so the toe box and the heel counter are invisible to it: a face
    # can sit comfortably "inside" the ceiling and both walls while physically
    # intersecting the front of the shoe. Measured on a real fit, 42% of the
    # exactly-colliding faces had *positive* directional clearance. The old
    # fitter tolerates that blind spot because it ranks candidates on exact SAT
    # collisions; a gradient-based fitter cannot, so the field carries an
    # honest unsigned distance too, signed by the directional inside/outside
    # test. Where no boundary exists (the open collar) the sign is taken as
    # positive, which reproduces the source contact policy exactly: the ankle
    # may pass through the opening but may not intersect the collar.
    spacing_array = np.asarray(
        [float((upper[axis] - lower[axis]) / (counts[axis] - 1)) for axis in range(3)]
    )
    occupancy = _voxelize_triangles(triangles, lower, spacing_array, counts)
    unsigned = build_obstacle_distance(occupancy, spacing_array)
    # A sample is outside if *any* ray family says so. Taking the union rather
    # than recomputing a combined minimum keeps the directional channel intact
    # while letting the longitudinal family contribute the sign it alone can
    # see.
    outside = valid & (combined < 0.0)
    if longitudinal_sign:
        outside = outside | longitudinal_outside
    distance = torch.where(outside, -unsigned.double(), unsigned.double())

    metadata = {
        "definition": {
            "positive": "sample lies inside its local cavity boundary",
            "negative": "sample lies beyond its local cavity boundary",
            "valid": "0 marks open space where no boundary exists",
            "open_above": "1 marks a footbed column with no surface above it",
        },
        "source": "foot_prior.cavity upper/side ray semantics, 2-D decomposition",
        "shape_dhw": [int(value) for value in combined.shape],
        "axis_order": "D over world X, H over world Y, W over world Z",
        "target_spacing_normalized": float(target_spacing),
        "target_spacing_mm": float(target_spacing) * 262.5,
        "actual_spacing_normalized": [
            float((upper[axis] - lower[axis]) / (counts[axis] - 1))
            for axis in range(3)
        ],
        "margin_normalized": float(margin),
        "numerical_tolerance": tolerance,
        "obstacle_triangle_count": int(triangles.shape[0]),
        "valid_fraction": float(valid.double().mean()),
        "open_above_fraction": float(open_above.double().mean()),
        "longitudinal_sign": bool(longitudinal_sign),
        "longitudinal_outside_fraction": float(
            longitudinal_outside.double().mean()
        ),
        "longitudinal_reference_x": center_x,
        "occupied_voxel_fraction": float(occupancy.double().mean()),
        "domain_lower": lower.tolist(),
        "domain_upper": upper.tolist(),
    }
    return ShoeCavityField(
        clearance=clearance.to(torch.float32)[None, None],
        valid=valid.to(torch.float32)[None, None],
        distance=distance.to(torch.float32)[None, None],
        lower=torch.as_tensor(lower, dtype=torch.float32, device=torch_device),
        upper=torch.as_tensor(upper, dtype=torch.float32, device=torch_device),
        spacing=torch.as_tensor(
            [
                float((upper[axis] - lower[axis]) / (counts[axis] - 1))
                for axis in range(3)
            ],
            dtype=torch.float32,
            device=torch_device,
        ),
        footbed_x=grid_x.to(torch.float32),
        footbed_z=grid_z.to(torch.float32),
        footbed_y=footbed_y.to(torch.float32),
        open_above=open_above.to(torch.float32),
        metadata=metadata,
    )


def sample_footbed_height(
    field: ShoeCavityField, points_xz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample the footbed height at (B, N, 2) world X/Z points.

    Returns the height and a coverage weight in [0, 1]; the weight falls to zero
    wherever the footbed does not reach, so support never invents a floor.
    """

    if points_xz.ndim != 3 or points_xz.shape[-1] != 2:
        raise ValueError("points_xz must have shape (B, N, 2)")
    batch = points_xz.shape[0]
    height = field.footbed_y
    axis_x, axis_z = field.footbed_x, field.footbed_z
    # A single-shoe field stores one lattice; a batched one stores a lattice per
    # shoe. Normalize both to the batched form.
    if height.ndim == 2:
        height = height[None].expand(batch, -1, -1)
        axis_x = axis_x[None].expand(batch, -1)
        axis_z = axis_z[None].expand(batch, -1)
    covered = torch.isfinite(height)
    filled = torch.where(covered, height, torch.zeros_like(height))
    volume = torch.stack((filled, covered.to(filled.dtype)), dim=1)

    lower = torch.stack((axis_x[:, 0], axis_z[:, 0]), dim=-1)[:, None, :]
    upper = torch.stack((axis_x[:, -1], axis_z[:, -1]), dim=-1)[:, None, :]
    unit = (points_xz - lower) / (upper - lower).clamp(min=1e-12)
    normalized = 2.0 * unit - 1.0
    # The stored field is (X, Z); grid_sample reads (x -> W, y -> H), so the
    # pair is reversed to (Z, X).
    grid = normalized.flip(-1).view(batch, points_xz.shape[1], 1, 2)
    sampled = torch.nn.functional.grid_sample(
        volume, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[:, :, :, 0]
    weight = sampled[:, 1].clamp(0.0, 1.0)
    safe = sampled[:, 0] / weight.clamp(min=1e-3)
    return safe, weight


def sample_open_above(
    field: ShoeCavityField, points_xz: torch.Tensor
) -> torch.Tensor:
    """Bilinearly sample the collar-opening mask at (B, N, 2) world X/Z points.

    Returns a weight in [0, 1]: 1 where the shoe has a footbed but nothing
    above it, so the foot leaves through a real opening in the upper. The value
    is deliberately soft at the rim of the opening rather than a hard in/out
    test, so a sample crossing the rim does not step-change the objective.
    """

    if points_xz.ndim != 3 or points_xz.shape[-1] != 2:
        raise ValueError("points_xz must have shape (B, N, 2)")
    batch = points_xz.shape[0]
    mask = field.open_above
    axis_x, axis_z = field.footbed_x, field.footbed_z
    if mask.ndim == 2:
        mask = mask[None].expand(batch, -1, -1)
        axis_x = axis_x[None].expand(batch, -1)
        axis_z = axis_z[None].expand(batch, -1)
    volume = mask[:, None].to(points_xz.dtype)

    lower = torch.stack((axis_x[:, 0], axis_z[:, 0]), dim=-1)[:, None, :]
    upper = torch.stack((axis_x[:, -1], axis_z[:, -1]), dim=-1)[:, None, :]
    unit = (points_xz - lower) / (upper - lower).clamp(min=1e-12)
    normalized = 2.0 * unit - 1.0
    # Stored as (X, Z); grid_sample reads (x -> W, y -> H), so reverse to (Z, X).
    grid = normalized.flip(-1).view(batch, points_xz.shape[1], 1, 2)
    sampled = torch.nn.functional.grid_sample(
        volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[:, 0, :, 0]
    return sampled.clamp(0.0, 1.0)


def _voxelize_triangles(
    triangles: torch.Tensor,
    lower: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
    oversample: float = 2.0,
) -> torch.Tensor:
    """Mark every voxel a triangle passes through.

    Each triangle is sampled on a barycentric lattice fine enough that
    consecutive samples are closer than half a voxel, so no wall is missed
    however it is oriented. Triangles are bucketed by the subdivision count they
    need, which keeps the whole thing vectorized.
    """

    device = triangles.device
    step = float(np.min(spacing)) / oversample
    edges = torch.stack(
        (
            (triangles[:, 1] - triangles[:, 0]).norm(dim=1),
            (triangles[:, 2] - triangles[:, 1]).norm(dim=1),
            (triangles[:, 0] - triangles[:, 2]).norm(dim=1),
        ),
        dim=1,
    ).max(dim=1).values
    divisions = torch.clamp(torch.ceil(edges / step).long(), 1, 96)
    occupancy = torch.zeros(
        int(np.prod(shape)), dtype=torch.bool, device=device
    )
    origin = torch.as_tensor(lower, dtype=torch.float64, device=device)
    cell = torch.as_tensor(spacing, dtype=torch.float64, device=device)
    limit = torch.as_tensor(shape - 1, dtype=torch.long, device=device)
    strides = torch.as_tensor(
        [int(shape[1] * shape[2]), int(shape[2]), 1], dtype=torch.long, device=device
    )

    for count in torch.unique(divisions).tolist():
        block = triangles[divisions == count]
        steps = torch.linspace(0.0, 1.0, count + 1, dtype=torch.float64, device=device)
        first, second = torch.meshgrid(steps, steps, indexing="ij")
        mask = (first + second) <= 1.0 + 1e-12
        weight_a = first[mask]
        weight_b = second[mask]
        weight_first = 1.0 - weight_a - weight_b
        for start in range(0, len(block), 2048):
            chunk = block[start : start + 2048]
            points = (
                weight_first[None, :, None] * chunk[:, 0][:, None, :]
                + weight_a[None, :, None] * chunk[:, 1][:, None, :]
                + weight_b[None, :, None] * chunk[:, 2][:, None, :]
            ).reshape(-1, 3)
            index = torch.round((points - origin) / cell).long()
            keep = ((index >= 0) & (index <= limit)).all(dim=1)
            if not bool(keep.any()):
                continue
            flat = (index[keep] * strides).sum(dim=1)
            occupancy[flat] = True
    return occupancy.view(*(int(value) for value in shape))


def build_obstacle_distance(
    occupancy: torch.Tensor, spacing: np.ndarray
) -> torch.Tensor:
    """Unsigned Euclidean distance to the nearest occupied voxel."""

    from scipy import ndimage

    empty = (~occupancy).cpu().numpy()
    distance = ndimage.distance_transform_edt(empty, sampling=tuple(spacing))
    return torch.as_tensor(
        np.asarray(distance, dtype=np.float32), device=occupancy.device
    )
