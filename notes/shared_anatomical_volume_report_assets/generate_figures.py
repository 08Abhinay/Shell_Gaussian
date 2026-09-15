"""Render deterministic report figures from the accepted geometry artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


OUTPUT_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation"
)
ASSET_ROOT = Path(__file__).resolve().parent
B3_ROOT = (
    OUTPUT_ROOT
    / "instance_anatomical_volume"
    / "batch_threads1_20260913_181358"
)
MAPPING_ROOT = (
    OUTPUT_ROOT
    / "instance_volume_mapping"
    / "batch_threads1_20260913_181358"
)
SHOE = "sandal_1"


def _load_mesh(relative: str) -> trimesh.Trimesh:
    mesh = trimesh.load(OUTPUT_ROOT / relative, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _display_points(points: np.ndarray) -> np.ndarray:
    """Put shoe length horizontally and anatomical height vertically."""

    points = np.asarray(points, dtype=np.float64)
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


def _face_colors(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    colors = np.asarray(mesh.visual.face_colors, dtype=np.float64)[face_indices] / 255.0
    colors[:, 3] = np.maximum(colors[:, 3], 0.32)
    return colors


def _mesh_panel(
    ax,
    mesh: trimesh.Trimesh,
    title: str,
    *,
    elevation: float = 18.0,
    azimuth: float = -64.0,
    maximum_faces: int = 120000,
    foot_crop: bool = False,
) -> None:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    indices = np.arange(len(faces), dtype=np.int64)
    if foot_crop:
        centroids = vertices[faces].mean(axis=1)
        indices = indices[centroids[:, 1] > -0.47]
    if len(indices) > maximum_faces:
        step = int(np.ceil(len(indices) / maximum_faces))
        indices = indices[::step]
    triangles = _display_points(vertices)[faces[indices]]
    collection = Poly3DCollection(
        triangles,
        facecolors=_face_colors(mesh, indices),
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection3d(collection)
    used = triangles.reshape(-1, 3)
    lower = used.min(axis=0)
    upper = used.max(axis=0)
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1.0e-6)
    radius = 0.54 * float(np.max(span))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold", pad=2)


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSET_ROOT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(15, 3.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3)
    ax.axis("off")
    labels = [
        "Support fit\n(previous update)",
        "Cavity\ndiagnosis",
        "Containment\nfit",
        "Corresponding\nanatomy",
        "Canonical\nvolume",
        "11-A\ntargets",
        "11-B\nvalid volumes",
        "11-C\nexact maps",
        "11-D\nsemantic fibers",
    ]
    states = ["done"] * 8 + ["next"]
    xs = np.linspace(0.9, 14.1, len(labels))
    for index, (x, label, state) in enumerate(zip(xs, labels, states)):
        color = "#2f855a" if state == "done" else "#dd6b20"
        box = FancyBboxPatch(
            (x - 0.68, 1.0),
            1.36,
            1.0,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, 1.5, label, ha="center", va="center", color="white", fontsize=9)
        if index + 1 < len(labels):
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.70, 1.5),
                    (xs[index + 1] - 0.70, 1.5),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="#4a5568",
                )
            )
    ax.text(
        7.5,
        2.55,
        "Progress from fitted support to a shared anatomical volume",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1a365d",
    )
    ax.text(
        7.5,
        0.38,
        "Section 1.1: exact geometry complete through 11-C  |  semantic (u,v,r) fibers remain",
        ha="center",
        fontsize=10.5,
        color="#4a5568",
    )
    _save(fig, "01_pipeline.png")


def _support_containment_figure() -> None:
    paths = [
        (f"support_fit/{SHOE}/support_fit_overlay.ply", "A. Accepted support fit"),
        (f"cavity_analysis/{SHOE}/cavity_overlay.ply", "B. Measured cavity conflicts"),
        (f"containment_fit/{SHOE}/containment_fit_overlay.ply", "C. Containment-fitted foot"),
        (f"containment_fit/{SHOE}/foot_clearance_colored.ply", "D. Final clearance classification"),
    ]
    fig = plt.figure(figsize=(15, 4.2))
    for index, (path, title) in enumerate(paths, 1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        _mesh_panel(ax, _load_mesh(path), title, elevation=20, azimuth=-68)
    fig.suptitle(
        "Support fit $\u2192$ collision diagnosis $\u2192$ containment fit: sandal_1",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Grey: shoe  |  blue: clear foot  |  yellow: near obstacle  |  magenta: beyond local boundary  |  red: exact collision",
        ha="center",
        fontsize=9,
        color="#4a5568",
    )
    _save(fig, "02_support_cavity_containment.png")


def _anatomy_figure() -> None:
    paths = [
        (f"anatomical_surface/{SHOE}/regions_surface.ply", "A. Dense foot surface regions"),
        (f"lower_leg_attachment/{SHOE}/lower_leg_collar_overlay.ply", "B. Lower-leg exit and collar"),
        (f"extended_anatomical_surface/{SHOE}/regions_components.ply", "C. Joined foot and lower leg"),
    ]
    fig = plt.figure(figsize=(13.2, 4.6))
    for index, (path, title) in enumerate(paths, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _mesh_panel(ax, _load_mesh(path), title, elevation=16, azimuth=-62)
    fig.suptitle(
        "Anatomical construction with stable correspondence",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _save(fig, "03_corresponding_anatomy.png")


_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _tetrahedral_plane_polygons(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    plane_z: float,
    point_values: np.ndarray | None = None,
    cell_values: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    points = vertices[tetrahedra]
    values_z = points[:, :, 2] - plane_z
    selected = np.flatnonzero(
        (np.min(values_z, axis=1) <= 0.0) & (np.max(values_z, axis=1) >= 0.0)
    )
    polygons: list[np.ndarray] = []
    colors: list[float] = []
    for cell_index in selected:
        cell = points[cell_index]
        z = cell[:, 2] - plane_z
        intersections: list[np.ndarray] = []
        interpolated: list[float] = []
        for first, second in _TET_EDGES:
            z0, z1 = z[first], z[second]
            if z0 == 0.0 and z1 == 0.0:
                candidates = ((first, 0.0), (second, 1.0))
            elif z0 == 0.0:
                candidates = ((first, 0.0),)
            elif z1 == 0.0:
                candidates = ((second, 1.0),)
            elif z0 * z1 < 0.0:
                fraction = -z0 / (z1 - z0)
                candidates = ((None, fraction),)
            else:
                candidates = ()
            for direct, fraction in candidates:
                if direct is None:
                    position = cell[first] + fraction * (cell[second] - cell[first])
                    value = None
                    if point_values is not None:
                        value = point_values[tetrahedra[cell_index, first]] + fraction * (
                            point_values[tetrahedra[cell_index, second]]
                            - point_values[tetrahedra[cell_index, first]]
                        )
                else:
                    position = cell[direct]
                    value = (
                        point_values[tetrahedra[cell_index, direct]]
                        if point_values is not None
                        else None
                    )
                if not any(np.linalg.norm(position - prior) <= 1.0e-12 for prior in intersections):
                    intersections.append(position)
                    if value is not None:
                        interpolated.append(float(value))
        if len(intersections) < 3:
            continue
        polygon = np.asarray([[point[0], -point[1]] for point in intersections])
        center = polygon.mean(axis=0)
        order = np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))
        polygons.append(polygon[order])
        if point_values is not None:
            colors.append(float(np.mean(interpolated)))
        elif cell_values is not None:
            colors.append(float(cell_values[cell_index]))
        else:
            colors.append(0.0)
    return polygons, np.asarray(colors, dtype=np.float64)


def _cross_section_panel(
    ax,
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    title: str,
    *,
    point_values: np.ndarray | None = None,
    cell_values: np.ndarray | None = None,
    cmap: str,
    vmin: float,
    vmax: float,
) -> matplotlib.cm.ScalarMappable:
    polygons, values = _tetrahedral_plane_polygons(
        vertices,
        tetrahedra,
        plane_z=0.0,
        point_values=point_values,
        cell_values=cell_values,
    )
    normalization = Normalize(vmin=vmin, vmax=vmax, clip=True)
    collection = PolyCollection(
        polygons,
        array=values,
        cmap=cmap,
        norm=normalization,
        edgecolors=(1, 1, 1, 0.10),
        linewidths=0.15,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return matplotlib.cm.ScalarMappable(norm=normalization, cmap=cmap)


def _canonical_volume_figure() -> None:
    reference = OUTPUT_ROOT / "anatomical_volume" / "reference"
    with np.load(reference / "canonical_volume.npz") as data:
        vertices = data["volume_vertices"]
        tetrahedra = data["tetrahedra"]
        harmonic_r = data["harmonic_r"]
    fig = plt.figure(figsize=(13.5, 5.1))
    ax_mesh = fig.add_subplot(1, 2, 1, projection="3d")
    _mesh_panel(
        ax_mesh,
        _load_mesh("anatomical_volume/reference/boundary_regions.ply"),
        "A. Closed inner boundary and fixed envelope",
        elevation=14,
        azimuth=-58,
    )
    ax_cut = fig.add_subplot(1, 2, 2)
    mappable = _cross_section_panel(
        ax_cut,
        vertices,
        tetrahedra,
        "B. Canonical sagittal section colored by harmonic $r$",
        point_values=harmonic_r,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(mappable, ax=ax_cut, fraction=0.035, pad=0.02)
    colorbar.set_label("harmonic $r$: anatomy 0 $\u2192$ envelope 1")
    fig.suptitle(
        "Canonical tetrahedral anatomical volume",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    _save(fig, "04_canonical_volume.png")


def _boundary_target_figure() -> None:
    paths = [
        (f"anatomical_volume/{SHOE}/boundary_target_overlay.ply", "A. Target over authoritative anatomy", False),
        (f"anatomical_volume/{SHOE}/boundary_target_overlay.ply", "B. Foot-region detail", True),
        (f"anatomical_volume/{SHOE}/computational_boundary_target.ply", "C. Computational target only", True),
    ]
    fig = plt.figure(figsize=(13.6, 4.7))
    for index, (path, title, crop) in enumerate(paths, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _mesh_panel(
            ax,
            _load_mesh(path),
            title,
            elevation=20,
            azimuth=-68,
            foot_crop=crop,
        )
    fig.suptitle(
        "Checkpoint 11-A: corresponding target, not yet a final valid volume",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Blue: computational target  |  grey: immutable fitted anatomy  |  purple: knee cap  |  red: target self-intersection",
        ha="center",
        fontsize=9,
        color="#4a5568",
    )
    _save(fig, "05_boundary_target.png")


def _surface_scalar_panel(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    foot_crop: bool = True,
) -> matplotlib.cm.ScalarMappable:
    local_faces = faces
    if foot_crop:
        keep = vertices[faces].mean(axis=1)[:, 1] > -0.47
        local_faces = faces[keep]
    display = _display_points(vertices)
    face_values = values[local_faces].max(axis=1)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = plt.get_cmap(cmap)(norm(face_values))
    collection = Poly3DCollection(
        display[local_faces], facecolors=colors, edgecolors="none", linewidths=0
    )
    ax.add_collection3d(collection)
    used = display[local_faces].reshape(-1, 3)
    lower, upper = used.min(axis=0), used.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.54 * float(np.max(upper - lower))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-68)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold")
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def _b3_figure() -> None:
    reference = OUTPUT_ROOT / "anatomical_volume" / "reference" / "canonical_volume.npz"
    continuation = B3_ROOT / SHOE / "continuation_state.npz"
    final_path = B3_ROOT / SHOE / "instance_volume.npz"
    with np.load(reference) as data:
        inner_indices = data["computational_inner_vertex_indices"]
        inner_faces = data["computational_inner_faces"]
        tetrahedra = data["tetrahedra"]
    with np.load(continuation) as data:
        b2_vertices = data["last_valid_volume_vertices"]
    with np.load(final_path) as data:
        final_vertices = data["volume_vertices"]
        corrections = np.linalg.norm(data["target_correction_vectors"], axis=1)
        determinants = data["jacobian_determinants"]
    report = json.loads((B3_ROOT / SHOE / "instance_volume.json").read_text())
    resolution = float(report["final"]["surface_resolution"])

    fig = plt.figure(figsize=(14.2, 4.8))
    ax_b2 = fig.add_subplot(1, 3, 1, projection="3d")
    _surface_scalar_panel(
        ax_b2,
        b2_vertices[inner_indices],
        inner_faces,
        np.zeros(len(inner_indices)),
        f"A. B2 safe warm start ($\\alpha={report['final']['source_b2_alpha']:.3f}$)",
        cmap="Blues",
        vmin=-1.0,
        vmax=1.0,
    )
    ax_final = fig.add_subplot(1, 3, 2, projection="3d")
    correction_map = _surface_scalar_panel(
        ax_final,
        final_vertices[inner_indices],
        inner_faces,
        corrections / resolution,
        "B. B3 boundary correction",
        cmap="magma",
        vmin=0.0,
        vmax=0.5,
    )
    cb1 = fig.colorbar(correction_map, ax=ax_final, fraction=0.035, pad=0.0)
    cb1.set_label("correction / surface resolution")
    ax_cut = fig.add_subplot(1, 3, 3)
    determinant_map = _cross_section_panel(
        ax_cut,
        final_vertices,
        tetrahedra,
        "C. Final tetrahedra: $\det(F_k)$",
        cell_values=determinants,
        cmap="turbo",
        vmin=0.02,
        vmax=2.0,
    )
    cb2 = fig.colorbar(determinant_map, ax=ax_cut, fraction=0.035, pad=0.02)
    cb2.set_label("Jacobian determinant")
    fig.suptitle(
        "Checkpoint 11-B on the difficult sandal_1 case",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    _save(fig, "06_b3_deformation.png")


def _mapping_figure() -> None:
    reference = OUTPUT_ROOT / "anatomical_volume" / "reference" / "canonical_volume.npz"
    final_path = B3_ROOT / SHOE / "instance_volume.npz"
    with np.load(reference) as data:
        canonical = data["volume_vertices"]
        tetrahedra = data["tetrahedra"]
        harmonic_r = data["harmonic_r"]
    with np.load(final_path) as data:
        instance = data["volume_vertices"]

    centroids = canonical[tetrahedra].mean(axis=1)
    candidates = np.flatnonzero(np.abs(centroids[:, 2]) < 0.018)
    x_bins = np.linspace(0.05, 0.90, 5)
    y_bins = np.linspace(0.08, 1.35, 4)
    chosen: list[int] = []
    for y in y_bins:
        for x in x_bins:
            score = (centroids[candidates, 0] - x) ** 2 + (
                -centroids[candidates, 1] - y
            ) ** 2
            for index in candidates[np.argsort(score)]:
                if int(index) not in chosen:
                    chosen.append(int(index))
                    break
    chosen = chosen[:12]
    weights = np.full(4, 0.25)
    canonical_samples = np.einsum("j,njk->nk", weights, canonical[tetrahedra[chosen]])
    instance_samples = np.einsum("j,njk->nk", weights, instance[tetrahedra[chosen]])
    sample_colors = plt.get_cmap("tab20")(np.linspace(0.0, 0.92, len(chosen)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, vertices, title, samples in (
        (axes[0], canonical, "A. Canonical volume $A$", canonical_samples),
        (axes[1], instance, "B. sandal_1 volume $A_i$", instance_samples),
    ):
        _cross_section_panel(
            ax,
            vertices,
            tetrahedra,
            title,
            point_values=harmonic_r,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        for label, (point, color) in enumerate(zip(samples, sample_colors), 1):
            ax.scatter(point[0], -point[1], s=42, color=color, edgecolor="black", linewidth=0.45, zorder=5)
            ax.text(point[0] + 0.012, -point[1] + 0.012, str(label), fontsize=7, weight="bold")
    fig.suptitle(
        "Checkpoint 11-C: identical tetrahedron IDs and barycentric weights identify corresponding points",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Matching number and color = one exact canonical coordinate transported through $\\chi_i$",
        ha="center",
        fontsize=9.5,
        color="#4a5568",
    )
    _save(fig, "07_exact_mapping.png")


def _batch_figure() -> None:
    summary = json.loads((B3_ROOT / "batch_summary.json").read_text())
    names = list(summary["results"])
    display_names = [
        name.replace("_", " ").replace("birkenstock arizona sandal", "birkenstock")
        .replace("crocs by speedyart studio", "crocs speedyart")
        .replace("duinn shoes womens hiking sandal sport", "duinn hiking sandal")
        .replace("priest karol wojtyas sports shoes", "priest sports shoe")
        .replace("ww ii german jack boots", "WWII jack boots")
        for name in names
    ]
    minimum_determinants = []
    maximum_conditions = []
    maximum_corrections = []
    statuses = []
    for name in names:
        report = json.loads((B3_ROOT / name / "instance_volume.json").read_text())
        final = report["final"]
        minimum_determinants.append(final["jacobian_determinant"]["minimum"])
        maximum_conditions.append(final["condition_number"]["maximum"])
        maximum_corrections.append(
            final["target_correction_magnitude"]["maximum"] / final["surface_resolution"]
        )
        statuses.append(report["status"])

    positions = np.arange(len(names))
    colors = ["#2f855a" if status == "final_exact_target" else "#3182ce" for status in statuses]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), sharey=True)
    axes[0].barh(positions, minimum_determinants, color=colors)
    axes[0].axvline(0.02, color="#c53030", linestyle="--", linewidth=1.3, label="limit 0.02")
    axes[0].set_xlabel("minimum $\\det(F_k)$")
    axes[0].set_title("No inverted/near-flat cells")
    axes[0].legend(fontsize=8)
    axes[1].barh(positions, maximum_conditions, color=colors)
    axes[1].axvline(20.0, color="#c53030", linestyle="--", linewidth=1.3, label="limit 20")
    axes[1].set_xlabel("maximum condition number")
    axes[1].set_title("Bounded distortion")
    axes[1].legend(fontsize=8)
    axes[2].barh(positions, maximum_corrections, color=colors)
    axes[2].axvline(0.5, color="#c53030", linestyle="--", linewidth=1.3, label="limit 0.5h")
    axes[2].set_xlabel("maximum correction / $h$")
    axes[2].set_title("Computational-boundary correction")
    axes[2].legend(fontsize=8)
    axes[0].set_yticks(positions, display_names, fontsize=8.3)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Checkpoint 11-B3 final validation for all 15 accepted shoes",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Green: exact target (4)  |  blue: safely corrected target (11)  |  all 15 passed independent post-validation",
        ha="center",
        fontsize=9.5,
        color="#4a5568",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, "08_batch_validation.png")


def _document_progress_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    blocks = [
        (0.25, 2.30, 2.25, 1.35, "Section 1.1\nCanonical domain $A$", "#2f855a"),
        (2.80, 2.30, 2.25, 1.35, "$\\chi_i$: canonical\n$\\to$ instance", "#2f855a"),
        (5.35, 2.30, 2.25, 1.35, "$\\Phi_i$: instance\n$\\to$ canonical", "#2f855a"),
        (7.90, 2.30, 2.25, 1.35, "Semantic\n$(u,v,r)$ fibers", "#dd6b20"),
        (10.75, 2.30, 2.50, 1.35, "Section 1.2\nmaterial field", "#718096"),
    ]
    for x, y, w, h, label, color in blocks:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="white"
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=11, weight="bold")
    for x0, x1 in ((2.50, 2.80), (5.05, 5.35), (7.60, 7.90), (10.15, 10.75)):
        ax.add_patch(FancyArrowPatch((x0, 2.98), (x1, 2.98), arrowstyle="-|>", mutation_scale=14, color="#4a5568"))
    ax.text(3.9, 1.35, "IMPLEMENTED AND VALIDATED FOR 15 SHOES", ha="center", fontsize=12, weight="bold", color="#276749")
    ax.plot([0.45, 7.4], [1.12, 1.12], color="#2f855a", linewidth=6, solid_capstyle="round")
    ax.text(9.02, 1.35, "NEXT: 11-D", ha="center", fontsize=12, weight="bold", color="#c05621")
    ax.plot([8.1, 9.95], [1.12, 1.12], color="#dd6b20", linewidth=6, solid_capstyle="round")
    ax.text(12.0, 1.35, "NOT STARTED", ha="center", fontsize=12, weight="bold", color="#4a5568")
    ax.plot([10.95, 13.05], [1.12, 1.12], color="#718096", linewidth=6, solid_capstyle="round")
    ax.text(6.75, 4.18, "Position relative to representation.pdf", ha="center", fontsize=15, weight="bold", color="#1a365d")
    _save(fig, "09_document_progress.png")


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    _pipeline_figure()
    _support_containment_figure()
    _anatomy_figure()
    _canonical_volume_figure()
    _boundary_target_figure()
    _b3_figure()
    _mapping_figure()
    _batch_figure()
    _document_progress_figure()


if __name__ == "__main__":
    main()
