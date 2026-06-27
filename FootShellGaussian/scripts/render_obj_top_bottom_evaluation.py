#!/usr/bin/env python3
"""Render external shoe assets into the legacy evaluation dataset layout.

This script is intended to be executed through Blender:

    blender --background --python render_obj_top_bottom_evaluation.py -- ...

It reconstructs the behavior of the previously deleted helper used for
generating the synthetic external evaluation shoes. The recovered
implementation focuses on the mode that the pipeline actually used in
practice: ``multi_elevation_360``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


RESOLUTION_X = 1536
RESOLUTION_Y = 1024
FOV_X_DEG = 22.5
SAMPLES = 64
CAMERA_RADIUS = 1.0
FIT_MARGIN = 1.1
BACKGROUND_GRAY = 92.0 / 255.0
VAL_STRIDE = 6
MULTI_ELEVATIONS_DEG = (-25.0, -5.0, 20.0, 45.0, 65.0)
VIEWS_PER_RING = 36


@dataclass
class RenderedFrame:
    file_path: str
    camera_angle_x: float
    transform_matrix: list[list[float]]
    ring_index: int
    azimuth_index: int
    elevation_deg: float
    azimuth_deg: float
    invdepth_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        default="multi_elevation_360",
        choices=["multi_elevation_360"],
        help="Recovered renderer currently supports the historical mode used by the pipeline.",
    )
    parser.add_argument("--shoe", action="append", default=None, help="Render only this shoe name.")
    parser.add_argument("--render-invdepth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(_argv_after_double_dash())


def _argv_after_double_dash() -> list[str]:
    argv = list(os.sys.argv)
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if "shoes" not in payload or not isinstance(payload["shoes"], list):
        raise ValueError(f"Manifest must contain a shoes list: {path}")
    return payload


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "PNG"
    scene.eevee.taa_render_samples = SAMPLES
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (BACKGROUND_GRAY, BACKGROUND_GRAY, BACKGROUND_GRAY, 1.0)
        bg.inputs[1].default_value = 1.0


def import_model(model_path: Path) -> list[bpy.types.Object]:
    suffix = model_path.suffix.lower()
    before = {obj.name for obj in bpy.data.objects}
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(model_path), automatic_bone_orientation=True)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(model_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(model_path))
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {model_path}")
    return mesh_objects


def deselect_all() -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def apply_object_transforms(objects: list[bpy.types.Object]) -> None:
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    deselect_all()


def axis_vector(spec: str) -> np.ndarray:
    sign = -1.0 if spec.startswith("-") else 1.0
    axis = spec[-1].upper()
    mapping = {
        "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if axis not in mapping:
        raise ValueError(f"Unsupported axis spec: {spec}")
    return sign * mapping[axis]


def source_axes_matrix(source_axes: dict[str, str]) -> np.ndarray:
    return np.stack(
        [
            axis_vector(source_axes["length"]),
            axis_vector(source_axes["width"]),
            axis_vector(source_axes["up"]),
        ],
        axis=0,
    )


def flip_mesh_normals(obj: bpy.types.Object) -> None:
    mesh = obj.data
    if hasattr(mesh, "flip_normals"):
        mesh.flip_normals()
        mesh.update()


def apply_axis_transform(objects: list[bpy.types.Object], transform_3x3: np.ndarray) -> None:
    matrix = Matrix(
        (
            (float(transform_3x3[0, 0]), float(transform_3x3[0, 1]), float(transform_3x3[0, 2]), 0.0),
            (float(transform_3x3[1, 0]), float(transform_3x3[1, 1]), float(transform_3x3[1, 2]), 0.0),
            (float(transform_3x3[2, 0]), float(transform_3x3[2, 1]), float(transform_3x3[2, 2]), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    det = float(np.linalg.det(transform_3x3))
    for obj in objects:
        obj.data.transform(matrix)
        if det < 0.0:
            flip_mesh_normals(obj)
        obj.data.update()


def join_objects(objects: list[bpy.types.Object], joined_name: str) -> bpy.types.Object:
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = joined_name
    deselect_all()
    return joined


def separate_loose_parts(joined: bpy.types.Object) -> list[bpy.types.Object]:
    deselect_all()
    joined.select_set(True)
    bpy.context.view_layer.objects.active = joined
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    deselect_all()
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def object_bbox_world(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    if not obj.data.vertices:
        origin = np.array(list(obj.matrix_world.translation), dtype=np.float64)
        return origin.copy(), origin.copy()
    verts_world = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices], dtype=np.float64)
    return verts_world.min(axis=0), verts_world.max(axis=0)


def component_records(objects: list[bpy.types.Object]) -> list[dict[str, Any]]:
    rows = []
    for idx, obj in enumerate(objects):
        bbox_min, bbox_max = object_bbox_world(obj)
        extent = bbox_max - bbox_min
        rows.append(
            {
                "index": idx,
                "name": obj.name,
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "center": ((bbox_min + bbox_max) * 0.5).tolist(),
                "extent": extent.tolist(),
                "bbox_volume": float(np.prod(extent)),
                "vertices": int(len(obj.data.vertices)),
                "faces": int(len(obj.data.polygons)),
            }
        )
    return rows


def delete_objects(objects: list[bpy.types.Object]) -> None:
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    deselect_all()


def apply_component_selection(
    objects: list[bpy.types.Object],
    selection_cfg: dict[str, Any] | None,
) -> tuple[list[bpy.types.Object], dict[str, Any]]:
    if not selection_cfg:
        records = component_records(objects)
        return objects, {
            "mode": None,
            "separate_loose_parts": False,
            "selection_axis": None,
            "selection_threshold": 0.0,
            "component_index": None,
            "component_count_before_selection": len(records),
            "components_before_selection": records,
            "kept_component_names": [row["name"] for row in records],
            "kept_component_count": len(records),
        }

    work_objects = list(objects)
    separate = bool(selection_cfg.get("separate_loose_parts", False))
    if separate:
        joined = join_objects(work_objects, "selected_mesh")
        work_objects = separate_loose_parts(joined)

    records = component_records(work_objects)
    mode = selection_cfg.get("mode")
    if mode != "axis-min":
        raise ValueError(f"Unsupported selection mode: {mode}")

    axis_name = str(selection_cfg.get("axis", "Y")).upper()
    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis_name]
    mins = np.array([row["bbox_min"][axis_index] for row in records], dtype=np.float64)
    global_min = float(mins.min())
    maxs = np.array([row["bbox_max"][axis_index] for row in records], dtype=np.float64)
    extent = float(maxs.max() - mins.min())
    threshold = max(extent * 1e-7, 1e-7)
    keep_mask = np.abs(mins - global_min) <= threshold
    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        raise RuntimeError("Component selection removed every component")

    kept = [obj for idx, obj in enumerate(work_objects) if keep_mask[idx]]
    removed = [obj for idx, obj in enumerate(work_objects) if not keep_mask[idx]]
    if removed:
        delete_objects(removed)

    kept_records = [records[idx] for idx in keep_indices.tolist()]
    return kept, {
        "mode": mode,
        "separate_loose_parts": separate,
        "selection_axis": axis_name,
        "selection_threshold": threshold,
        "component_index": int(keep_indices[0]),
        "component_count_before_selection": len(records),
        "components_before_selection": records,
        "kept_component_names": [row["name"] for row in kept_records],
        "kept_component_count": len(kept_records),
    }


def scene_bbox(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for obj in objects:
        bbox_min, bbox_max = object_bbox_world(obj)
        mins.append(bbox_min)
        maxs.append(bbox_max)
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def translate_objects(objects: list[bpy.types.Object], offset: np.ndarray) -> None:
    offset_vec = Vector((float(offset[0]), float(offset[1]), float(offset[2])))
    for obj in objects:
        obj.location += offset_vec
    bpy.context.view_layer.update()


def scale_objects(objects: list[bpy.types.Object], scale: float) -> None:
    scale_matrix = Matrix.Scale(float(scale), 4)
    for obj in objects:
        obj.data.transform(scale_matrix)
        obj.location *= float(scale)
        obj.data.update()
    bpy.context.view_layer.update()


def normalization_scale(bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    centered_min = bbox_min.copy()
    centered_max = bbox_max.copy()
    extent = centered_max - centered_min
    half_extent = 0.5 * extent
    sphere_radius = float(np.linalg.norm(half_extent))
    half_fov_x = math.radians(FOV_X_DEG) * 0.5
    tan_x = math.tan(half_fov_x)
    tan_y = tan_x * (RESOLUTION_Y / RESOLUTION_X)
    required_distance = FIT_MARGIN * sphere_radius / min(tan_x, tan_y)
    scale = CAMERA_RADIUS / required_distance if required_distance > 0.0 else 1.0
    return centered_min, centered_max, required_distance, scale


def canonicalize_geometry(
    mesh_objects: list[bpy.types.Object],
    shoe_cfg: dict[str, Any],
) -> dict[str, Any]:
    apply_object_transforms(mesh_objects)
    axis_mat = source_axes_matrix(shoe_cfg["source_axes"])
    apply_axis_transform(mesh_objects, axis_mat)

    selected_objects, selection_summary = apply_component_selection(
        [obj for obj in bpy.context.scene.objects if obj.type == "MESH"],
        shoe_cfg.get("selection"),
    )

    bbox_min_raw, bbox_max_raw = scene_bbox(selected_objects)
    center = 0.5 * (bbox_min_raw + bbox_max_raw)
    translate_objects(selected_objects, -center)
    bbox_min_centered, bbox_max_centered = scene_bbox(selected_objects)
    centered_min, centered_max, required_distance, scale = normalization_scale(
        bbox_min_centered, bbox_max_centered
    )
    scale_objects(selected_objects, scale)
    bbox_min_scaled, bbox_max_scaled = scene_bbox(selected_objects)

    mesh_objects_after = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    material_names = set()
    vertex_count = 0
    face_count = 0
    for obj in mesh_objects_after:
        vertex_count += len(obj.data.vertices)
        face_count += len(obj.data.polygons)
        for slot in obj.material_slots:
            if slot.material is not None:
                material_names.add(slot.material.name)

    return {
        "objects": mesh_objects_after,
        "source_axes_matrix": axis_mat,
        "raw_bbox_min": bbox_min_raw,
        "raw_bbox_max": bbox_max_raw,
        "canonical_bbox_min": bbox_min_scaled,
        "canonical_bbox_max": bbox_max_scaled,
        "normalization": {
            "center_offset": (-center).tolist(),
            "bbox_min_before_scale": centered_min.tolist(),
            "bbox_max_before_scale": centered_max.tolist(),
            "bbox_extent_before_scale": (centered_max - centered_min).tolist(),
            "required_distance_before_scale": required_distance,
            "target_camera_radius": CAMERA_RADIUS,
            "scale": scale,
        },
        "component_selection": selection_summary,
        "mesh_summary": {
            "objects": len(mesh_objects_after),
            "vertices": int(vertex_count),
            "faces": int(face_count),
            "materials": len(material_names),
        },
    }


def ensure_camera() -> bpy.types.Object:
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.angle = math.radians(FOV_X_DEG)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return cam


def configure_depth_output(temp_dir: Path) -> bpy.types.Node:
    scene = bpy.context.scene
    nt = scene.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    render_layers = nt.nodes.new("CompositorNodeRLayers")
    file_output = nt.nodes.new("CompositorNodeOutputFile")
    file_output.base_path = str(temp_dir)
    file_output.format.file_format = "OPEN_EXR"
    file_output.format.color_mode = "RGB"
    file_output.format.color_depth = "32"
    file_output.file_slots[0].path = "depth_"
    nt.links.new(render_layers.outputs["Depth"], file_output.inputs[0])
    return file_output


def c2w_from_eye(eye: np.ndarray, target: np.ndarray = np.zeros(3, dtype=np.float64)) -> np.ndarray:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def orbit_eye(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    return np.array(
        [
            radius * math.cos(az) * cos_el,
            radius * math.sin(az) * cos_el,
            radius * math.sin(el),
        ],
        dtype=np.float64,
    )


def render_image_and_depth(
    camera: bpy.types.Object,
    c2w: np.ndarray,
    temp_dir: Path,
    image_path: Path,
    mask_path: Path,
    invdepth_path: Path | None,
) -> None:
    scene = bpy.context.scene
    camera.matrix_world = Matrix(c2w.tolist())

    rgba_path = temp_dir / "rgba.png"
    depth_path = temp_dir / "depth_0001.exr"
    if rgba_path.exists():
        rgba_path.unlink()
    if depth_path.exists():
        depth_path.unlink()

    scene.render.filepath = str(rgba_path)
    bpy.ops.render.render(write_still=True)

    rgba_img = bpy.data.images.load(str(rgba_path), check_existing=False)
    width, height = rgba_img.size[:]
    rgba = np.array(rgba_img.pixels[:], dtype=np.float32).reshape(height, width, 4)
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + BACKGROUND_GRAY * (1.0 - alpha)
    mask = (alpha[..., 0] > 1e-4).astype(np.float32)

    save_float_image(
        rgb,
        image_path,
        file_format="JPEG",
        alpha_value=1.0,
    )
    save_float_image(
        np.repeat(mask[..., None], 3, axis=2),
        mask_path,
        file_format="PNG",
        alpha_value=1.0,
    )

    if invdepth_path is not None:
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth render did not produce {depth_path}")
        depth_img = bpy.data.images.load(str(depth_path), check_existing=False)
        depth = np.array(depth_img.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., 0]
        valid = np.isfinite(depth) & (depth > 0.0) & (mask > 0.5)
        invdepth = np.zeros_like(depth, dtype=np.float32)
        invdepth[valid] = 1.0 / np.maximum(depth[valid], 1e-8)
        invdepth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(invdepth_path, invdepth.astype(np.float32))
        bpy.data.images.remove(depth_img)

    bpy.data.images.remove(rgba_img)


def save_float_image(rgb: np.ndarray, path: Path, file_format: str, alpha_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = rgb.shape
    rgba = np.concatenate(
        [rgb.astype(np.float32), np.full((height, width, 1), float(alpha_value), dtype=np.float32)],
        axis=2,
    )
    image = bpy.data.images.new(
        name=f"save_{path.stem}",
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    image.pixels.foreach_set(rgba.reshape(-1))
    scene = bpy.context.scene
    old_format = scene.render.image_settings.file_format
    old_color_mode = scene.render.image_settings.color_mode
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = "RGB"
    image.save_render(str(path), scene=scene)
    scene.render.image_settings.file_format = old_format
    scene.render.image_settings.color_mode = old_color_mode
    bpy.data.images.remove(image)


def render_multi_elevation_dataset(
    shoe_name: str,
    shoe_root: Path,
    render_invdepth: bool,
) -> tuple[list[RenderedFrame], dict[str, Any]]:
    output_root = shoe_root / "multi_elevation_360"
    all_dir = output_root / "all"
    image_dir = all_dir / "image"
    mask_dir = all_dir / "mask"
    invdepth_dir = all_dir / "invdepth"
    if output_root.exists():
        shutil.rmtree(output_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if render_invdepth:
        invdepth_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{shoe_name}_render_"))
    try:
        ensure_camera()
        configure_depth_output(temp_dir)
        camera = bpy.context.scene.camera
        frames: list[RenderedFrame] = []
        frame_idx = 1
        for ring_index, elevation_deg in enumerate(MULTI_ELEVATIONS_DEG):
            for azimuth_index in range(VIEWS_PER_RING):
                azimuth_deg = 90.0 - 10.0 * azimuth_index
                eye = orbit_eye(CAMERA_RADIUS, azimuth_deg, elevation_deg)
                c2w = c2w_from_eye(eye)

                basename = f"img{frame_idx:03d}"
                image_path = image_dir / f"{basename}.jpg"
                mask_path = mask_dir / f"{basename}.png"
                invdepth_path = invdepth_dir / f"{basename}.npy" if render_invdepth else None
                render_image_and_depth(camera, c2w, temp_dir, image_path, mask_path, invdepth_path)

                frames.append(
                    RenderedFrame(
                        file_path=f"image/{basename}.jpg",
                        camera_angle_x=math.radians(FOV_X_DEG),
                        transform_matrix=c2w.tolist(),
                        ring_index=ring_index,
                        azimuth_index=azimuth_index,
                        elevation_deg=float(elevation_deg),
                        azimuth_deg=float(azimuth_deg),
                        invdepth_path=f"invdepth/{basename}.npy" if render_invdepth else None,
                    )
                )
                frame_idx += 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    split_summary = write_splits(output_root, frames, render_invdepth)
    summary = {
        "view_count": len(frames),
        "camera_angle_x": math.radians(FOV_X_DEG),
        "camera_radius": CAMERA_RADIUS,
        "elevations_deg": list(MULTI_ELEVATIONS_DEG),
        "views_per_ring": VIEWS_PER_RING,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "invdepth_dir": str(invdepth_dir) if render_invdepth else None,
        "split": split_summary,
    }
    json_dump(output_root / "multi_elevation_summary.json", summary)
    return frames, summary


def write_frame_json(path: Path, frames: list[dict[str, Any]]) -> None:
    json_dump(path, {"frames": frames})


def frame_to_payload(frame: RenderedFrame) -> dict[str, Any]:
    payload = {
        "file_path": frame.file_path,
        "camera_angle_x": frame.camera_angle_x,
        "transform_matrix": frame.transform_matrix,
        "ring_index": frame.ring_index,
        "azimuth_index": frame.azimuth_index,
        "elevation_deg": frame.elevation_deg,
        "azimuth_deg": frame.azimuth_deg,
    }
    if frame.invdepth_path is not None:
        payload["invdepth_path"] = frame.invdepth_path
    return payload


def copy_frame_assets(src_root: Path, dst_root: Path, src_frame: RenderedFrame, dst_index: int, render_invdepth: bool) -> dict[str, Any]:
    basename = f"img{dst_index:03d}"
    src_image = src_root / src_frame.file_path
    src_mask = src_root / src_frame.file_path.replace("image/", "mask/").replace(".jpg", ".png")
    dst_image_rel = f"image/{basename}.jpg"
    dst_mask_rel = f"mask/{basename}.png"
    dst_image = dst_root / dst_image_rel
    dst_mask = dst_root / dst_mask_rel
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_mask, dst_mask)

    payload = {
        "file_path": dst_image_rel,
        "camera_angle_x": src_frame.camera_angle_x,
        "transform_matrix": src_frame.transform_matrix,
        "ring_index": src_frame.ring_index,
        "azimuth_index": src_frame.azimuth_index,
        "elevation_deg": src_frame.elevation_deg,
        "azimuth_deg": src_frame.azimuth_deg,
        "all_file_path": src_frame.file_path,
    }

    if render_invdepth and src_frame.invdepth_path is not None:
        src_inv = src_root / src_frame.invdepth_path
        dst_inv_rel = f"invdepth/{basename}.npy"
        dst_inv = dst_root / dst_inv_rel
        dst_inv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_inv, dst_inv)
        payload["invdepth_path"] = dst_inv_rel
        payload["all_invdepth_path"] = src_frame.invdepth_path
    return payload


def write_splits(output_root: Path, frames: list[RenderedFrame], render_invdepth: bool) -> dict[str, Any]:
    all_dir = output_root / "all"
    write_frame_json(all_dir / "transforms.json", [frame_to_payload(frame) for frame in frames])

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    train_frames: list[dict[str, Any]] = []
    val_frames: list[dict[str, Any]] = []
    train_idx = 1
    val_idx = 1

    for all_index, frame in enumerate(frames):
        if all_index % VAL_STRIDE == 0:
            val_frames.append(copy_frame_assets(all_dir, val_dir, frame, val_idx, render_invdepth))
            val_idx += 1
        else:
            train_frames.append(copy_frame_assets(all_dir, train_dir, frame, train_idx, render_invdepth))
            train_idx += 1

    write_frame_json(train_dir / "transforms.json", train_frames)
    write_frame_json(val_dir / "transforms.json", val_frames)
    return {
        "train_count": len(train_frames),
        "val_count": len(val_frames),
        "val_stride": VAL_STRIDE,
    }


def root_summary_row(
    shoe_cfg: dict[str, Any],
    shoe_root: Path,
    canonicalization: dict[str, Any],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    bbox_min = np.array(canonicalization["canonical_bbox"]["min"], dtype=np.float64)
    bbox_max = np.array(canonicalization["canonical_bbox"]["max"], dtype=np.float64)
    extent = bbox_max - bbox_min
    row = {
        "shoe": shoe_cfg["name"],
        "status": status,
        "model": str(canonicalization["source_model"]),
        "turntable_dir": str(shoe_root / "turntable"),
        "top_to_bottom_dir": str(shoe_root / "top_to_bottom"),
        "multi_elevation_360_dir": str(shoe_root / "multi_elevation_360"),
        "canonical_extent_x": float(extent[0]),
        "canonical_extent_y": float(extent[1]),
        "canonical_extent_z": float(extent[2]),
        "scale": float(canonicalization["normalization"]["scale"]),
    }
    if error is not None:
        row["error"] = error
    return row


def render_one_shoe(shoe_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    shoe_name = str(shoe_cfg["name"])
    source_model = args.source_root / shoe_cfg["model"]
    shoe_root = args.output_root / shoe_name
    if shoe_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output shoe dir exists; pass --overwrite: {shoe_root}")
        shutil.rmtree(shoe_root)
    shoe_root.mkdir(parents=True, exist_ok=True)

    reset_scene()
    mesh_objects = import_model(source_model)
    canonical = canonicalize_geometry(mesh_objects, shoe_cfg)
    frames, summary = render_multi_elevation_dataset(shoe_name, shoe_root, args.render_invdepth)

    canonicalization_payload = {
        "shoe": shoe_name,
        "source_model": str(source_model),
        "renderer": {
            "blender_version": bpy.app.version_string,
            "engine": bpy.context.scene.render.engine,
            "resolution": [RESOLUTION_X, RESOLUTION_Y],
            "fov_x_deg": FOV_X_DEG,
            "samples": SAMPLES,
            "camera_radius": CAMERA_RADIUS,
            "margin": FIT_MARGIN,
        },
        "source_axes": {
            "length_to_canonical_x": shoe_cfg["source_axes"]["length"],
            "width_to_canonical_y": shoe_cfg["source_axes"]["width"],
            "up_to_canonical_z": shoe_cfg["source_axes"]["up"],
        },
        "mesh": canonical["mesh_summary"],
        "component_selection": canonical["component_selection"],
        "raw_bbox": {
            "min": canonical["raw_bbox_min"].tolist(),
            "max": canonical["raw_bbox_max"].tolist(),
            "extent": (canonical["raw_bbox_max"] - canonical["raw_bbox_min"]).tolist(),
        },
        "canonical_bbox": {
            "min": canonical["canonical_bbox_min"].tolist(),
            "max": canonical["canonical_bbox_max"].tolist(),
            "extent": (canonical["canonical_bbox_max"] - canonical["canonical_bbox_min"]).tolist(),
        },
        "normalization": canonical["normalization"],
        "turntable": None,
        "top_to_bottom": None,
        "multi_elevation_360": summary,
    }
    json_dump(shoe_root / "synthetic_canonicalization.json", canonicalization_payload)
    return {
        "row": root_summary_row(shoe_cfg, shoe_root, canonicalization_payload, "ok", None),
        "payload": canonicalization_payload,
        "frames": frames,
    }


def write_root_summaries(output_root: Path, rows: list[dict[str, Any]]) -> None:
    json_dump(output_root / "summary.json", {"rows": rows})
    if not rows:
        return
    csv_path = output_root / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    selected = set(args.shoe or [])

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    args.output_root.mkdir(parents=True, exist_ok=True)

    for shoe_cfg in manifest["shoes"]:
        shoe_name = str(shoe_cfg["name"])
        if selected and shoe_name not in selected:
            continue

        try:
            result = render_one_shoe(shoe_cfg, args)
            rows.append(result["row"])
            print(f"[ok] {shoe_name}")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "shoe": shoe_name,
                    "status": "failed",
                    "model": str(args.source_root / shoe_cfg["model"]),
                    "error": error,
                }
            )
            errors.append(f"{shoe_name}: {error}")
            print(f"[failed] {shoe_name}: {error}")

    write_root_summaries(args.output_root, rows)
    if errors:
        raise SystemExit("Rendering failed for: " + "; ".join(errors))


if __name__ == "__main__":
    main()
