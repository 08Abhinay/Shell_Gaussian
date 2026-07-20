#!/usr/bin/env python3
"""Blender-side worker for the direct evaluation dataset pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


RESOLUTION_X = 1536
RESOLUTION_Y = 1024
FOV_X_DEG = 21.0
CAMERA_RADIUS = 1.0
ELEVATIONS_DEG = (0.0, -25.0, 20.0, 45.0, 65.0)
VIEWS_PER_RING = 36
TEST_STRIDE = 6
REFERENCE_HORIZONTAL_OCCUPANCY = 0.84
BORDER_MARGIN_FRACTION = 0.03
SAMPLES = 64
MIN_INVDEPTH_MASK_IOU = 0.98
PROJECTION_BBOX_TOLERANCE_PX = 8.0


def rotation_x(angle_rad: float) -> np.ndarray:
    sine, cosine = math.sin(angle_rad), math.cos(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, sine, 0.0],
            [0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


BLENDER_TO_EFFECTIVE_GSHELL = rotation_x(-math.pi / 2.0)
BLENDER_TO_SAVED_GSHELL = rotation_x(-math.pi)


def argv_after_separator() -> list[str]:
    argv = list(os.sys.argv)
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "build"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shoe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv_after_separator())


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def manifest_entry(path: Path, shoe: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [record for record in payload["shoes"] if record["name"] == shoe]
    if len(matches) != 1:
        raise ValueError(f"Manifest does not contain exactly one entry for {shoe!r}")
    return matches[0]


def reset_scene(resolution_x: int = RESOLUTION_X, resolution_y: int = RESOLUTION_Y) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.taa_render_samples = SAMPLES
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs[1].default_value = 0.25
    add_area_light("Key", (1.8, -2.2, 2.8), 100.0, 4.0)
    add_area_light("Fill", (-2.5, -0.8, 1.5), 50.0, 5.0)
    add_area_light("Rim", (0.0, 2.8, 2.2), 75.0, 4.0)


def add_area_light(name: str, location: tuple[float, float, float], energy: float, size: float) -> None:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    direction = -light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def import_glb(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
    return meshes


def deselect_all() -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def bake_hierarchy(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    for obj in objects:
        world = obj.matrix_world.copy()
        mesh = obj.data.copy()
        mesh.transform(world)
        obj.data = mesh
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
    bpy.context.view_layer.update()
    return objects


def axis_vector(token: str) -> np.ndarray:
    sign = -1.0 if token.startswith("-") else 1.0
    vectors = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    return sign * vectors[token[-1]]


def source_axis_matrix(entry: dict[str, Any]) -> np.ndarray:
    axes = entry["source_axes"]
    matrix = np.stack(
        [axis_vector(axes["length"]), axis_vector(axes["width"]), axis_vector(axes["up"])],
        axis=0,
    )
    if entry.get("mirror_width", False):
        matrix[1] *= -1.0
    return matrix


def flip_normals(obj: bpy.types.Object) -> None:
    if hasattr(obj.data, "flip_normals"):
        obj.data.flip_normals()
        obj.data.update()


def apply_axis_transform(objects: list[bpy.types.Object], transform: np.ndarray) -> None:
    matrix = Matrix(
        (
            (*transform[0].tolist(), 0.0),
            (*transform[1].tolist(), 0.0),
            (*transform[2].tolist(), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    for obj in objects:
        obj.data.transform(matrix)
        if np.linalg.det(transform) < 0.0:
            flip_normals(obj)
        obj.data.update()


def object_bbox(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([obj.matrix_world @ vertex.co for vertex in obj.data.vertices], dtype=np.float64)
    if vertices.size == 0:
        raise RuntimeError(f"Mesh object is empty: {obj.name}")
    return vertices.min(axis=0), vertices.max(axis=0)


def component_records(objects: list[bpy.types.Object]) -> list[dict[str, Any]]:
    records = []
    for index, obj in enumerate(objects):
        bbox_min, bbox_max = object_bbox(obj)
        records.append(
            {
                "index": index,
                "name": obj.name,
                "center": ((bbox_min + bbox_max) * 0.5).tolist(),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "vertices": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
            }
        )
    return records


def join_and_separate_loose_parts(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    deselect_all()
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def delete_objects(objects: list[bpy.types.Object]) -> None:
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    deselect_all()


def select_components(
    objects: list[bpy.types.Object], selection: dict[str, Any]
) -> tuple[list[bpy.types.Object], dict[str, Any]]:
    if selection.get("mode") == "all":
        records = component_records(objects)
        return objects, {
            "mode": "all",
            "components_before": records,
            "kept_components": [record["name"] for record in records],
        }
    if selection.get("separate_loose_parts", False):
        objects = join_and_separate_loose_parts(objects)
    records = component_records(objects)
    axis = str(selection["axis"])
    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
    centers = np.asarray([record["center"][axis_index] for record in records])
    mins = np.asarray([record["bbox_min"][axis_index] for record in records])
    maxs = np.asarray([record["bbox_max"][axis_index] for record in records])
    pivot = float(0.5 * (mins.min() + maxs.max()))
    if selection["side"] == "min":
        keep = centers <= pivot
    else:
        keep = centers >= pivot
    kept = [obj for index, obj in enumerate(objects) if keep[index]]
    removed = [obj for index, obj in enumerate(objects) if not keep[index]]
    if not kept:
        raise RuntimeError("Component selection removed every mesh component")
    delete_objects(removed)
    return kept, {
        "mode": "axis-side",
        "axis": axis,
        "side": selection["side"],
        "pivot": pivot,
        "separate_loose_parts": bool(selection.get("separate_loose_parts", False)),
        "components_before": records,
        "kept_components": [obj.name for obj in kept],
    }


def all_vertices(objects: list[bpy.types.Object]) -> np.ndarray:
    chunks = []
    for obj in objects:
        chunks.append(
            np.asarray([obj.matrix_world @ vertex.co for vertex in obj.data.vertices], dtype=np.float64)
        )
    return np.concatenate(chunks, axis=0)


def translate_objects(objects: list[bpy.types.Object], offset: np.ndarray) -> None:
    vector = Vector(offset.tolist())
    for obj in objects:
        obj.location += vector
    bpy.context.view_layer.update()


def scale_objects(objects: list[bpy.types.Object], scale: float) -> None:
    matrix = Matrix.Scale(float(scale), 4)
    for obj in objects:
        obj.data.transform(matrix)
        obj.location *= float(scale)
        obj.data.update()
    bpy.context.view_layer.update()


def orbit_eye(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return np.array(
        [
            radius * math.cos(azimuth) * cos_elevation,
            radius * math.sin(azimuth) * cos_elevation,
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def c2w_from_eye(eye: np.ndarray) -> np.ndarray:
    forward = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix


def camera_schedule() -> list[dict[str, Any]]:
    frames = []
    for ring_index, elevation_deg in enumerate(ELEVATIONS_DEG):
        for azimuth_index in range(VIEWS_PER_RING):
            azimuth_deg = -90.0 + 10.0 * azimuth_index
            blender_c2w = c2w_from_eye(orbit_eye(CAMERA_RADIUS, azimuth_deg, elevation_deg))
            frames.append(
                {
                    "ring_index": ring_index,
                    "azimuth_index": azimuth_index,
                    "elevation_deg": elevation_deg,
                    "azimuth_deg": azimuth_deg,
                    "blender_c2w": blender_c2w,
                    "saved_c2w": BLENDER_TO_SAVED_GSHELL @ blender_c2w,
                    "effective_c2w": BLENDER_TO_EFFECTIVE_GSHELL @ blender_c2w,
                }
            )
    return frames


def projected_bounds(points: np.ndarray, c2w: np.ndarray) -> tuple[float, float, float, float] | None:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]
    depth = -camera[:, 2]
    if np.any(depth <= 1e-6):
        return None
    tan_x = math.tan(math.radians(FOV_X_DEG) * 0.5)
    aspect = RESOLUTION_X / RESOLUTION_Y
    tan_y = tan_x / aspect
    x = camera[:, 0] / depth / tan_x
    y = camera[:, 1] / depth / tan_y
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def scale_for_camera_contract(points: np.ndarray, schedule: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    max_radius = float(np.linalg.norm(points, axis=1).max())
    upper = 0.95 * CAMERA_RADIUS / max(max_radius, 1e-9)

    def reference_width(scale: float) -> float:
        bounds = projected_bounds(points * scale, schedule[0]["blender_c2w"])
        return float("inf") if bounds is None else (bounds[1] - bounds[0]) * 0.5

    def every_view_fits(scale: float) -> bool:
        limit = 1.0 - 2.0 * BORDER_MARGIN_FRACTION
        for frame in schedule:
            bounds = projected_bounds(points * scale, frame["blender_c2w"])
            if bounds is None or max(abs(value) for value in bounds) > limit:
                return False
        return True

    low, high = 0.0, upper
    for _ in range(50):
        middle = 0.5 * (low + high)
        if reference_width(middle) <= REFERENCE_HORIZONTAL_OCCUPANCY:
            low = middle
        else:
            high = middle
    occupancy_scale = low

    low, high = 0.0, upper
    for _ in range(50):
        middle = 0.5 * (low + high)
        if every_view_fits(middle):
            low = middle
        else:
            high = middle
    fit_scale = low
    scale = min(occupancy_scale, fit_scale)
    return scale, {
        "reference_target": REFERENCE_HORIZONTAL_OCCUPANCY,
        "reference_occupancy": reference_width(scale),
        "occupancy_limited_scale": occupancy_scale,
        "all_view_fit_limited_scale": fit_scale,
        "limiting_rule": "reference_occupancy" if occupancy_scale <= fit_scale else "all_view_border",
        "border_margin_fraction": BORDER_MARGIN_FRACTION,
    }


def canonicalize(entry: dict[str, Any], model_path: Path) -> tuple[list[bpy.types.Object], dict[str, Any]]:
    objects = bake_hierarchy(import_glb(model_path))
    axis_matrix = source_axis_matrix(entry)
    apply_axis_transform(objects, axis_matrix)
    objects, selection_summary = select_components(
        objects, entry.get("selection", {"mode": "all"})
    )
    points = all_vertices(objects)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    translate_objects(objects, -center)
    centered_points = all_vertices(objects)
    schedule = camera_schedule()
    scale, framing = scale_for_camera_contract(centered_points, schedule)
    scale_objects(objects, scale)
    canonical_points = all_vertices(objects)
    summary = {
        "source_axes": entry["source_axes"],
        "mirror_width": bool(entry.get("mirror_width", False)),
        "axis_matrix": axis_matrix.tolist(),
        "selection": selection_summary,
        "source_bbox_min": bbox_min.tolist(),
        "source_bbox_max": bbox_max.tolist(),
        "center_offset": (-center).tolist(),
        "uniform_scale": scale,
        "canonical_bbox_min": canonical_points.min(axis=0).tolist(),
        "canonical_bbox_max": canonical_points.max(axis=0).tolist(),
        "framing": framing,
    }
    print(
        f"[setup] {entry['name']}: objects={len(objects)}, scale={scale:.8g}, "
        f"occupancy={framing['reference_occupancy']:.4f}, "
        f"limiter={framing['limiting_rule']}"
    )
    return objects, summary


def ensure_camera() -> bpy.types.Object:
    data = bpy.data.cameras.new("Camera")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.angle = math.radians(FOV_X_DEG)
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def configure_depth_output(temp_dir: Path) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    file_output = nodes.new("CompositorNodeOutputFile")
    file_output.base_path = str(temp_dir)
    file_output.format.file_format = "OPEN_EXR"
    file_output.format.color_mode = "RGB"
    file_output.format.color_depth = "32"
    file_output.file_slots[0].path = "depth_"
    scene.node_tree.links.new(render_layers.outputs["Depth"], file_output.inputs[0])


def save_float_image(
    rgb: np.ndarray, path: Path, file_format: str, color_mode: str = "RGB"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = rgb.shape
    rgba = np.concatenate(
        [rgb.astype(np.float32), np.ones((height, width, 1), dtype=np.float32)], axis=2
    )
    image = bpy.data.images.new(
        f"save_{path.stem}", width=width, height=height, alpha=True, float_buffer=False
    )
    image.pixels.foreach_set(rgba.reshape(-1))
    scene = bpy.context.scene
    old_format = scene.render.image_settings.file_format
    old_color_mode = scene.render.image_settings.color_mode
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = color_mode
    image.save_render(str(path), scene=scene)
    scene.render.image_settings.file_format = old_format
    scene.render.image_settings.color_mode = old_color_mode
    bpy.data.images.remove(image)


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 1.0


def save_sparse_npy(path: Path, values: np.ndarray) -> None:
    """Write a normal float32 NPY while leaving all-background disk pages sparse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=values.shape)
    for row in np.flatnonzero(np.any(values != 0.0, axis=1)):
        columns = np.flatnonzero(values[row] != 0.0)
        start, stop = int(columns[0]), int(columns[-1]) + 1
        mapped[row, start:stop] = values[row, start:stop]
    mapped.flush()
    del mapped


def render_frame(
    camera: bpy.types.Object,
    c2w: np.ndarray,
    temp_dir: Path,
    image_path: Path,
    mask_path: Path,
    invdepth_path: Path | None,
) -> tuple[np.ndarray, float]:
    camera.matrix_world = Matrix(c2w.tolist())
    rgba_path = temp_dir / "rgba.png"
    depth_path = temp_dir / "depth_0001.exr"
    rgba_path.unlink(missing_ok=True)
    depth_path.unlink(missing_ok=True)
    bpy.context.scene.render.filepath = str(rgba_path)
    bpy.ops.render.render(write_still=True)
    rgba_image = bpy.data.images.load(str(rgba_path), check_existing=False)
    width, height = rgba_image.size[:]
    rgba = np.asarray(rgba_image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    alpha = rgba[..., 3]
    mask_bottom_left = alpha > 1e-4
    if not mask_bottom_left.any():
        raise RuntimeError(f"Empty render mask: {image_path.name}")
    if (
        mask_bottom_left[0].any()
        or mask_bottom_left[-1].any()
        or mask_bottom_left[:, 0].any()
        or mask_bottom_left[:, -1].any()
    ):
        raise RuntimeError(f"Shoe touches image border: {image_path.name}")
    rgb = rgba[..., :3] * alpha[..., None] + (1.0 - alpha[..., None])
    save_float_image(rgb, image_path, "JPEG")
    save_float_image(
        np.repeat(mask_bottom_left[..., None].astype(np.float32), 3, axis=2),
        mask_path,
        "PNG",
        "BW",
    )

    minimum_iou = 1.0
    if invdepth_path is not None:
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth pass not produced: {depth_path}")
        depth_image = bpy.data.images.load(str(depth_path), check_existing=False)
        depth = np.asarray(depth_image.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., 0]
        valid = np.isfinite(depth) & (depth > 0.0) & mask_bottom_left
        inverse_depth = np.zeros_like(depth, dtype=np.float32)
        inverse_depth[valid] = 1.0 / np.maximum(depth[valid], 1e-8)
        inverse_depth = np.flipud(inverse_depth).copy()
        mask_top_left = np.flipud(mask_bottom_left)
        minimum_iou = binary_iou(inverse_depth > 0.0, mask_top_left)
        if minimum_iou < MIN_INVDEPTH_MASK_IOU:
            raise RuntimeError(
                f"Depth/mask IoU {minimum_iou:.6f} is below {MIN_INVDEPTH_MASK_IOU:.2f}"
            )
        save_sparse_npy(invdepth_path, inverse_depth)
        bpy.data.images.remove(depth_image)
    bpy.data.images.remove(rgba_image)
    return np.flipud(mask_bottom_left), minimum_iou


def mesh_triangles(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        object_vertices = np.asarray(
            [obj.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float64
        )
        vertices.append(object_vertices)
        triangles.extend(
            tuple(offset + int(index) for index in triangle.vertices)
            for triangle in mesh.loop_triangles
        )
        offset += len(object_vertices)
    return np.concatenate(vertices, axis=0), np.asarray(triangles, dtype=np.int64)


def write_reference_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    faces = np.empty(
        len(triangles),
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))], align=False),
    )
    faces["count"] = 3
    faces["indices"] = triangles.astype(np.int32, copy=False)
    with path.open("wb") as handle:
        handle.write(header)
        np.ascontiguousarray(vertices, dtype="<f4").tofile(handle)
        faces.tofile(handle)


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.where(mask)
    return np.array([columns.min(), columns.max(), rows.min(), rows.max()], dtype=np.float64)


def projected_pixel_bbox(vertices: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    camera = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]
    depth = -camera[:, 2]
    valid = depth > 1e-6
    camera = camera[valid]
    depth = depth[valid]
    tan_x = math.tan(math.radians(FOV_X_DEG) * 0.5)
    tan_y = tan_x / (RESOLUTION_X / RESOLUTION_Y)
    x_ndc = camera[:, 0] / depth / tan_x
    y_ndc = camera[:, 1] / depth / tan_y
    x = (x_ndc + 1.0) * 0.5 * (RESOLUTION_X - 1)
    y = (1.0 - y_ndc) * 0.5 * (RESOLUTION_Y - 1)
    return np.array([x.min(), x.max(), y.min(), y.max()], dtype=np.float64)


def frame_payload(frame: dict[str, Any], index: int) -> dict[str, Any]:
    basename = f"img{index + 1:03d}"
    return {
        "file_path": f"image/{basename}.jpg",
        "invdepth_path": f"invdepth/{basename}.npy",
        "camera_angle_x": math.radians(FOV_X_DEG),
        "transform_matrix": frame["saved_c2w"].tolist(),
        "ring_index": frame["ring_index"],
        "azimuth_index": frame["azimuth_index"],
        "elevation_deg": frame["elevation_deg"],
        "azimuth_deg": frame["azimuth_deg"],
    }


def write_transforms(output: Path, schedule: list[dict[str, Any]]) -> None:
    all_payload = [frame_payload(frame, index) for index, frame in enumerate(schedule)]
    test_indices = set(range(0, len(schedule), TEST_STRIDE))
    train_payload = [frame for index, frame in enumerate(all_payload) if index not in test_indices]
    test_payload = [frame for index, frame in enumerate(all_payload) if index in test_indices]
    common = {
        "render_pose_convention": "blender_c2w_opengl_camera_z_up",
        "pose_convention": "legacy_gshell_saved_c2w_for_fixed_loader",
        "effective_pose_convention": "gshell_trainer_c2w_x_length_y_down_z_width",
    }
    json_dump(output / "transforms.json", {**common, "frames": all_payload})
    json_dump(output / "transforms_train.json", {**common, "frames": train_payload})
    json_dump(output / "transforms_test.json", {**common, "frames": test_payload})


def build(entry: dict[str, Any], source_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reset_scene()
    model_path = source_root / entry["model"]
    objects, canonicalization = canonicalize(entry, model_path)
    schedule = camera_schedule()
    camera = ensure_camera()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{entry['name']}_render_"))
    configure_depth_output(temp_dir)
    masks: dict[int, np.ndarray] = {}
    minimum_iou = 1.0
    projection_indices = {ring * VIEWS_PER_RING for ring in range(len(ELEVATIONS_DEG))}
    try:
        for index, frame in enumerate(schedule):
            basename = f"img{index + 1:03d}"
            mask, iou = render_frame(
                camera,
                frame["blender_c2w"],
                temp_dir,
                output / "image" / f"{basename}.jpg",
                output / "mask" / f"{basename}.png",
                output / "invdepth" / f"{basename}.npy",
            )
            minimum_iou = min(minimum_iou, iou)
            if index in projection_indices:
                masks[index] = mask
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    blender_vertices, triangles = mesh_triangles(objects)
    homogeneous = np.concatenate([blender_vertices, np.ones((len(blender_vertices), 1))], axis=1)
    effective_vertices = (BLENDER_TO_EFFECTIVE_GSHELL @ homogeneous.T).T[:, :3]
    write_reference_ply(output / "reference_mesh.ply", effective_vertices, triangles)

    projection_rows = []
    max_error = 0.0
    for index in sorted(projection_indices):
        projected = projected_pixel_bbox(effective_vertices, schedule[index]["effective_c2w"])
        rendered = mask_bbox(masks[index])
        error = float(np.abs(projected - rendered).max())
        max_error = max(max_error, error)
        projection_rows.append(
            {
                "frame": f"img{index + 1:03d}",
                "projected_bbox": projected.tolist(),
                "rendered_mask_bbox": rendered.tolist(),
                "maximum_error_px": error,
            }
        )
    if max_error > PROJECTION_BBOX_TOLERANCE_PX:
        raise RuntimeError(
            f"Reference mesh projection error {max_error:.3f}px exceeds "
            f"{PROJECTION_BBOX_TOLERANCE_PX:.1f}px"
        )

    write_transforms(output, schedule)
    metadata = {
        "version": 1,
        "shoe": entry["name"],
        "source_model": str(model_path),
        "source_sha256": entry["sha256"],
        "reviewed_manifest_entry": True,
        "blender_version": bpy.app.version_string,
        "canonical_geometry": canonicalization,
        "camera_contract": {
            "resolution": [RESOLUTION_X, RESOLUTION_Y],
            "fov_x_deg": FOV_X_DEG,
            "radius": CAMERA_RADIUS,
            "elevations_deg": list(ELEVATIONS_DEG),
            "views_per_ring": VIEWS_PER_RING,
            "view_count": len(schedule),
            "first_azimuth_deg": -90.0,
            "azimuth_step_deg": 10.0,
            "saved_pose_formula": "Rx(-180deg) @ blender_c2w",
            "effective_loader_pose_formula": "Rx(+90deg) @ saved_c2w",
        },
        "split": {
            "train_count": 150,
            "test_count": 30,
            "test_stride": TEST_STRIDE,
            "assets_are_shared": True,
        },
        "inverse_depth": {
            "format": "float32_numpy_inverse_camera_z_depth",
            "storage": "standard_npy_with_sparse_background_allocation",
            "origin": "top_left",
            "minimum_mask_iou": minimum_iou,
            "second_layer": False,
        },
        "reference_mesh": {
            "path": "reference_mesh.ply",
            "coordinate_system": "effective_gshell_x_length_y_down_z_width",
            "vertices": len(effective_vertices),
            "faces": len(triangles),
        },
        "reference_mesh_projection": {
            "sampled_frames": projection_rows,
            "maximum_bbox_error_px": max_error,
            "tolerance_px": PROJECTION_BBOX_TOLERANCE_PX,
            "passed": max_error <= PROJECTION_BBOX_TOLERANCE_PX,
        },
    }
    json_dump(output / "blender_canonicalization.json", metadata)


def audit(entry: dict[str, Any], source_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reset_scene(768, 512)
    objects, canonicalization = canonicalize(entry, source_root / entry["model"])
    camera = ensure_camera()
    bpy.context.scene.render.resolution_x = 768
    bpy.context.scene.render.resolution_y = 512
    views = (
        ("side", -90.0, 0.0),
        ("toe", 0.0, 0.0),
        ("opposite", 90.0, 0.0),
        ("heel", 180.0, 0.0),
        ("top", -90.0, 65.0),
    )
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{entry['name']}_audit_"))
    try:
        for label, azimuth, elevation in views:
            render_frame(
                camera,
                c2w_from_eye(orbit_eye(CAMERA_RADIUS, azimuth, elevation)),
                temp_dir,
                output / f"{label}.jpg",
                output / f"{label}_mask.png",
                None,
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    json_dump(
        output / "audit.json",
        {
            "shoe": entry["name"],
            "source_model": entry["model"],
            "canonical_geometry": canonicalization,
            "expected_semantics": {
                "+X": "heel_to_toe",
                "+Y": "width",
                "+Z": "physical_up",
                "side_view": "camera_at_-Y_toe_points_right",
            },
            "mesh_objects": len(objects),
        },
    )


def main() -> None:
    args = parse_args()
    entry = manifest_entry(args.manifest.resolve(), args.shoe)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Worker output is not empty: {args.output}")
    if args.action == "build":
        build(entry, args.source_root.resolve(), args.output.resolve())
    else:
        audit(entry, args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
