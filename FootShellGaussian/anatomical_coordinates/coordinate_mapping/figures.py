"""Diagnostic figures: the anatomy in the shoe, the map, and what got addressed.

Three figures per shoe, each answering a different question:

    alignment.png  does the anatomy sit inside the shoe, and where do the
                   fibers run? Shoe and anatomy are drawn together, because
                   either one alone says nothing about the fit.
    flow.png       what does the map actually do - how far does each part of
                   the canonical anatomy travel, and does anything compress?
    coverage.png   which parts of the shoe surface got an anatomical address.

Projections use the shoe frame: X runs heel to toe, Y is positive downward
toward the sole, Z is width. Y is negated for display so the side views read
upright.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

INK, GRID = "#22252c", "#c9ced8"
SHOE, ANATOMY = "#7fa8c4", "#d8a06a"
FIBER, ORIGIN, ENDPOINT = "#7a5ba6", "#3f8f5f", "#c2603f"
ADDRESSED, INSIDE, BEYOND = "#4c8ca8", "#c2603f", "#b9a24a"


def _style(axis, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_facecolor("white")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=8, length=3)
    if xlabel:
        axis.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK, fontsize=9)


def _silhouette(axis, vertices, faces, a, b, colour, alpha=0.06, limit=14000, seed=0):
    """Project a mesh as overlapping translucent triangles.

    Cheaper and more honest than a hull: concavities stay visible, and a shoe
    is mostly concavity.
    """

    if len(faces) > limit:
        faces = faces[np.random.default_rng(seed).choice(len(faces), limit, replace=False)]
    corners = vertices[faces][:, :, [a, b]].copy()
    if b == 1:
        corners[:, :, 1] *= -1.0
    axis.add_collection(
        PolyCollection(corners, facecolors=colour, edgecolors="none", alpha=alpha)
    )


def _fibers(axis, curves, a, b, every: int = 28, width: float = 0.9):
    """Every fiber's endpoints, but only some of the curves.

    Drawing all the curves turns the panel into a thicket. Drawing only a few
    of the endpoints hides where the anatomy actually is. So: every origin and
    every outer end as a dot, a readable subset as lines.
    """

    segments, origins, endpoints = [], [], []
    for index in range(curves.shape[0]):
        curve = curves[index]
        good = np.isfinite(curve).all(axis=1)
        if good.sum() < 2:
            continue
        path = curve[good][:, [a, b]].copy()
        if b == 1:
            path[:, 1] *= -1.0
        origins.append(path[0])
        endpoints.append(path[-1])
        if index % every == 0:
            segments.append(path)
    if not origins:
        return
    if segments:
        axis.add_collection(
            LineCollection(segments, colors=FIBER, linewidths=width, alpha=0.8, zorder=2)
        )
    origins = np.asarray(origins)
    endpoints = np.asarray(endpoints)
    # Endpoints are numerous and spread wide; kept small and soft so they
    # mark the outer reach without burying the shoe behind them.
    axis.scatter(endpoints[:, 0], endpoints[:, 1], s=4.5, c=ENDPOINT,
                 linewidths=0, zorder=3, alpha=0.45)
    axis.scatter(origins[:, 0], origins[:, 1], s=6, c=ORIGIN,
                 linewidths=0, zorder=4, alpha=0.8)


def alignment_figure(
    path: Path, name: str, shoe_vertices, shoe_faces, anatomy_vertices, anatomy_faces,
    fiber_curves, canonical_vertices, landed, fitted,
) -> None:
    """The anatomy inside the shoe, with fibers running outward through it."""

    error = np.linalg.norm(landed - fitted, axis=1) * 262.5
    figure = plt.figure(figsize=(15, 5.2), dpi=130)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.24)

    for column, (a, b, xl, yl, title) in enumerate((
        (0, 1, "heel to toe", "height", "side"),
        (0, 2, "heel to toe", "width", "from above"),
    )):
        axis = figure.add_subplot(grid[0, column])
        _silhouette(axis, shoe_vertices, shoe_faces, a, b, SHOE, alpha=0.30)
        _silhouette(axis, anatomy_vertices, anatomy_faces, a, b, ANATOMY, alpha=0.34)
        if fiber_curves is not None:
            _fibers(axis, fiber_curves, a, b)
        axis.set_title(title, color=INK, fontsize=10)
        axis.autoscale_view()
        axis.set_aspect("equal")
        _style(axis, xl, yl)
        if column == 0:
            handles = [
                plt.Line2D([], [], marker="s", ls="", color=SHOE, label="shoe", alpha=0.5),
                plt.Line2D([], [], marker="s", ls="", color=ANATOMY, label="anatomy", alpha=0.7),
                plt.Line2D([], [], color=FIBER, label="fibers"),
                plt.Line2D([], [], marker="o", ls="", color=ORIGIN, label="on the foot"),
                plt.Line2D([], [], marker="o", ls="", color=ENDPOINT, label="outer end"),
            ]
            axis.legend(handles=handles, loc="lower left", fontsize=7.5,
                        frameon=False, labelcolor=INK)

    axis = figure.add_subplot(grid[0, 2])
    axis.hist(error, bins=40, color=ADDRESSED, alpha=0.85)
    axis.set_title("map landing error", color=INK, fontsize=10)
    _style(axis, "millimetres", "vertices")

    figure.suptitle(
        f"{name.replace('_', ' ')}    anatomy in the shoe, with fibers    "
        f"landing rms {np.sqrt((error ** 2).mean()):.3f} mm  ·  "
        f"p99 {np.percentile(error, 99):.3f} mm",
        color=INK, fontsize=11,
    )
    figure.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def flow_figure(path: Path, name: str, canonical, landed, faces, determinant=None) -> None:
    """What the map does: where the canonical anatomy goes, and by how much."""

    travel = np.linalg.norm(landed - canonical, axis=1) * 262.5
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=130)

    axis = axes[0]
    _silhouette(axis, canonical, faces, 0, 1, GRID, alpha=0.10)
    _silhouette(axis, landed, faces, 0, 1, ANATOMY, alpha=0.10)
    step = max(1, len(canonical) // 900)
    start, finish = canonical[::step], landed[::step]
    axis.quiver(
        start[:, 0], -start[:, 1],
        (finish - start)[:, 0], -(finish - start)[:, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.0022,
        color=FIBER, alpha=0.7,
    )
    axis.set_title("canonical → fitted", color=INK, fontsize=10)
    axis.set_aspect("equal"); axis.autoscale_view(); _style(axis, "heel to toe", "height")

    axis = axes[1]
    spread = axis.scatter(landed[:, 0], -landed[:, 1], s=2.0, c=travel,
                          cmap="viridis", linewidths=0)
    axis.set_title("how far each part travelled", color=INK, fontsize=10)
    axis.set_aspect("equal"); _style(axis, "heel to toe", "height")
    bar = figure.colorbar(spread, ax=axis, fraction=0.045)
    bar.set_label("millimetres", color=INK, fontsize=8)
    bar.ax.tick_params(colors=INK, labelsize=7)

    axis = axes[2]
    if determinant is not None and np.isfinite(determinant).any():
        axis.hist(determinant, bins=40, color=ADDRESSED, alpha=0.85)
        axis.axvline(1.0, color=INK, lw=0.9, ls="--")
        axis.set_title("local volume change", color=INK, fontsize=10)
        _style(axis, "determinant (1 = preserved)", "samples")
    else:
        axis.hist(travel, bins=40, color=ADDRESSED, alpha=0.85)
        axis.set_title("travel distance", color=INK, fontsize=10)
        _style(axis, "millimetres", "vertices")

    figure.suptitle(
        f"{name.replace('_', ' ')}    the coordinate map    "
        f"median travel {np.median(travel):.1f} mm  ·  max {travel.max():.1f} mm",
        color=INK, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def coverage_figure(
    path: Path, name: str, points, addressed, inside, depth,
    shoe_vertices=None, shoe_faces=None,
) -> None:
    """Which parts of the shoe surface received an anatomical address."""

    beyond = ~addressed & ~inside
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=130)
    for axis, (a, b, xl, yl) in zip(axes[:2], (
        (0, 1, "heel to toe", "height"), (0, 2, "heel to toe", "width")
    )):
        if shoe_vertices is not None:
            _silhouette(axis, shoe_vertices, shoe_faces, a, b, SHOE, alpha=0.09)
        for mask, colour, label in ((addressed, ADDRESSED, "addressed"),
                                    (inside, INSIDE, "inside the foot"),
                                    (beyond, BEYOND, "beyond the domain")):
            if mask.any():
                y = -points[mask, b] if b == 1 else points[mask, b]
                axis.scatter(points[mask, a], y, s=1.1, c=colour, label=label,
                             linewidths=0, alpha=0.7)
        axis.set_aspect("equal"); axis.autoscale_view(); _style(axis, xl, yl)
    axes[0].legend(loc="upper right", fontsize=7.5, frameon=False,
                   markerscale=6, labelcolor=INK)

    valid = depth[addressed]
    valid = valid[np.isfinite(valid)]
    if valid.size:
        axes[2].hist(valid, bins=45, color=ADDRESSED, alpha=0.85)
    axes[2].set_title("distance out from the foot", color=INK, fontsize=10)
    _style(axes[2], "outward distance", "points")

    figure.suptitle(
        f"{name.replace('_', ' ')}    {addressed.mean() * 100:.2f}% addressed  ·  "
        f"{inside.mean() * 100:.2f}% inside the foot  ·  {beyond.mean() * 100:.2f}% beyond",
        color=INK, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def write_coverage_overlay(path: Path, points, addressed, inside) -> None:
    """The sampled shoe points, coloured by classification, as a PLY cloud."""

    colours = np.tile(np.array([[185, 162, 74]], dtype=np.uint8), (len(points), 1))
    colours[addressed] = (76, 140, 168)
    colours[inside] = (194, 96, 63)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    for point, colour in zip(points, colours):
        lines.append(
            f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
            f"{colour[0]} {colour[1]} {colour[2]}"
        )
    Path(path).write_text("\n".join(lines) + "\n")
