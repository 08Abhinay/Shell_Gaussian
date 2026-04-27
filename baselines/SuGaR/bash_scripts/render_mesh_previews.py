#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def normalize_points(points):
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def sample_mesh(mesh_path, n_points):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError(f"Open3D could not read a mesh from {mesh_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        colors = np.full((len(points), 3), 0.72, dtype=np.float32)
    return normalize_points(points), np.clip(colors, 0.0, 1.0)


def render_view(points, colors, output_path, elev, azim, title):
    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.15,
        linewidths=0,
        marker=".",
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render quick headless mesh previews.")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--points", default=120_000, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points, colors = sample_mesh(args.mesh, args.points)

    views = [
        ("front", 10, -90),
        ("side", 10, 0),
        ("back", 10, 90),
        ("top", 75, -90),
        ("iso", 25, -45),
    ]

    frames = []
    for name, elev, azim in views:
        out = args.out_dir / f"{args.mesh.stem}_{name}.png"
        render_view(points, colors, out, elev=elev, azim=azim, title=name)
        frames.append(imageio.imread(out))

    gif_path = args.out_dir / f"{args.mesh.stem}_preview.gif"
    imageio.mimsave(gif_path, frames, duration=0.75)
    print(f"Preview images saved in {args.out_dir}")
    print(f"Preview GIF saved at {gif_path}")


if __name__ == "__main__":
    main()
