#!/usr/bin/env python3
"""Render canonical RGB images and masks from external shoe assets.

This script is intended to be executed through Blender:

    blender --background --python blender_renderer.py -- ...

Use ``pipeline.py render`` as the public entry point. This worker deliberately
writes no camera transforms, depth maps, train/validation splits, or summaries.
"""

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


SUPPORTED_MODEL_SUFFIXES = frozenset({".dae", ".fbx", ".glb", ".gltf", ".obj"})
SUPPORTED_AXIS_TOKENS = frozenset({"X", "Y", "Z", "-X", "-Y", "-Z"})
AXIS_INDEX_TO_NAME = {0: "X", 1: "Y", 2: "Z"}
AXIS_NAME_TO_INDEX = {name: idx for idx, name in AXIS_INDEX_TO_NAME.items()}
RESOLUTION_X = 1536
RESOLUTION_Y = 1024
FOV_X_DEG = 22.5
SAMPLES = 64
CAMERA_RADIUS = 1.0
RING_RADII = (1.0, 1.0, 1.0, 1.25, 1.5)
REFERENCE_HORIZONTAL_OCCUPANCY = 0.84
BORDER_MARGIN_FRACTION = 0.02
CONSERVATIVE_FIT_RELAXATION = 1.15
BACKGROUND_GRAY = 1.0
COMPOSITE_BACKGROUND_VALUE = 4.0
WORLD_BACKGROUND_STRENGTH = 1.0
MULTI_ELEVATIONS_DEG = (-25.0, -5.0, 20.0, 45.0, 65.0)
VIEWS_PER_RING = 36
AUTO_SELECTION_PROBE_VIEWS = (
    (20.0, 0.0),
    (20.0, 60.0),
    (20.0, 120.0),
    (20.0, 180.0),
    (20.0, 240.0),
    (20.0, 300.0),
)
PROBE_MASK_DOWNSAMPLE = 4
PROBE_MIN_COMPONENT_PIXELS = 64
PROBE_LARGE_COMPONENT_RATIO = 0.08
PROBE_MIN_VERTEX_RATIO = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shoe", action="append", default=None, help="Render only this shoe name.")
    parser.add_argument(
        "--selection-debug-dir",
        type=Path,
        default=None,
        help="Optional directory for keeping auto-selection probe renders and reports.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(_argv_after_double_dash())


def _argv_after_double_dash() -> list[str]:
    argv = list(os.sys.argv)
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def validate_selection_cfg(selection_cfg: dict[str, Any], shoe_name: str) -> None:
    mode = selection_cfg.get("mode")
    if mode not in {"axis-min", "axis-side"}:
        raise ValueError(f"{shoe_name}: unsupported selection mode {mode!r}")

    axis = str(selection_cfg.get("axis", "Y")).upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"{shoe_name}: unsupported selection axis {axis!r}")

    if mode == "axis-side":
        side = str(selection_cfg.get("side", "min")).lower()
        if side not in {"min", "max"}:
            raise ValueError(f"{shoe_name}: unsupported selection side {side!r}")

    separate = selection_cfg.get("separate_loose_parts", False)
    if not isinstance(separate, bool):
        raise ValueError(f"{shoe_name}: separate_loose_parts must be a boolean")


def validate_manifest(payload: dict[str, Any], source_root: Path) -> None:
    seen_names: set[str] = set()
    for idx, shoe_cfg in enumerate(payload["shoes"]):
        if not isinstance(shoe_cfg, dict):
            raise ValueError(f"Manifest shoe entry {idx} must be an object")

        shoe_name = str(shoe_cfg.get("name", "")).strip()
        if not shoe_name:
            raise ValueError(f"Manifest shoe entry {idx} is missing a non-empty name")
        if shoe_name in seen_names:
            raise ValueError(f"Manifest contains duplicate shoe name: {shoe_name}")
        seen_names.add(shoe_name)

        model_rel = shoe_cfg.get("model")
        if not isinstance(model_rel, str) or not model_rel.strip():
            raise ValueError(f"{shoe_name}: model must be a non-empty string")
        model_path = source_root / model_rel
        if model_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            raise ValueError(
                f"{shoe_name}: unsupported model suffix {model_path.suffix!r}; "
                f"supported: {sorted(SUPPORTED_MODEL_SUFFIXES)}"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"{shoe_name}: model file not found: {model_path}")

        source_axes = shoe_cfg.get("source_axes", "auto")
        if isinstance(source_axes, str):
            if source_axes != "auto":
                raise ValueError(f"{shoe_name}: source_axes string must be 'auto' when provided as text")
        elif isinstance(source_axes, dict):
            for axis_name in ("length", "width", "up"):
                axis_value = source_axes.get(axis_name)
                if not isinstance(axis_value, str) or axis_value not in SUPPORTED_AXIS_TOKENS:
                    raise ValueError(
                        f"{shoe_name}: source_axes.{axis_name} must be one of "
                        f"{sorted(SUPPORTED_AXIS_TOKENS)}"
                    )
        else:
            raise ValueError(f"{shoe_name}: source_axes must be either an object or 'auto'")

        selection_cfg = shoe_cfg.get("selection")
        if selection_cfg is not None:
            if not isinstance(selection_cfg, dict):
                raise ValueError(f"{shoe_name}: selection must be an object when provided")
            validate_selection_cfg(selection_cfg, shoe_name)


def load_manifest(path: Path, source_root: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if "shoes" not in payload or not isinstance(payload["shoes"], list):
        raise ValueError(f"Manifest must contain a shoes list: {path}")
    validate_manifest(payload, source_root)
    return payload


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def clone_shoe_cfg_with_selection(
    shoe_cfg: dict[str, Any],
    selection_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cloned = dict(shoe_cfg)
    if selection_cfg is None:
        cloned.pop("selection", None)
    else:
        cloned["selection"] = dict(selection_cfg)
    return cloned


def selection_cfg_label(selection_cfg: dict[str, Any] | None) -> str:
    if selection_cfg is None:
        return "no_split"
    mode = str(selection_cfg["mode"])
    axis = str(selection_cfg.get("axis", "")).upper()
    side = str(selection_cfg.get("side", "")).lower()
    pieces = [mode]
    if axis:
        pieces.append(axis)
    if side:
        pieces.append(side)
    if selection_cfg.get("separate_loose_parts", False):
        pieces.append("loose")
    return "_".join(pieces)


def auto_selection_candidates() -> list[dict[str, Any] | None]:
    candidates: list[dict[str, Any] | None] = [None]
    for axis_name in ("X", "Y", "Z"):
        for side in ("min", "max"):
            candidates.append(
                {
                    "mode": "axis-side",
                    "axis": axis_name,
                    "side": side,
                    "separate_loose_parts": True,
                }
            )
    return candidates


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
    # The raw dataset needs only Blender's combined RGBA render. Keeping the
    # compositor disabled avoids carrying over depth/compositor state.
    scene.use_nodes = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (BACKGROUND_GRAY, BACKGROUND_GRAY, BACKGROUND_GRAY, 1.0)
        bg.inputs[1].default_value = WORLD_BACKGROUND_STRENGTH


def import_model(model_path: Path) -> list[bpy.types.Object]:
    suffix = model_path.suffix.lower()
    before = {obj.name for obj in bpy.data.objects}
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(model_path), automatic_bone_orientation=True)
    elif suffix == ".dae":
        if not hasattr(bpy.ops.wm, "collada_import"):
            raise RuntimeError("This Blender build does not expose wm.collada_import for .dae assets")
        bpy.ops.wm.collada_import(filepath=str(model_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(model_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(model_path))
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    relink_missing_images(model_path)

    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {model_path}")
    return mesh_objects


def relink_missing_images(model_path: Path) -> None:
    search_roots: list[Path] = []
    for root in (model_path.parent, model_path.parent.parent):
        if root.is_dir() and root not in search_roots:
            search_roots.append(root)

    if not search_roots:
        return

    indexed: dict[str, Path] = {}
    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".webp",
                ".exr",
            }:
                indexed.setdefault(path.name.lower(), path)

    for image in bpy.data.images:
        resolved = Path(bpy.path.abspath(image.filepath))
        if resolved.is_file():
            continue
        replacement = indexed.get(Path(image.filepath).name.lower())
        if replacement is None:
            continue
        image.filepath = str(replacement)
        try:
            image.reload()
        except RuntimeError:
            continue


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


def axis_token(axis_name: str, sign: float) -> str:
    return axis_name if sign >= 0.0 else f"-{axis_name}"


def all_vertex_coords_world(objects: list[bpy.types.Object]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for obj in objects:
        if not obj.data.vertices:
            continue
        verts_world = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices], dtype=np.float64)
        if verts_world.size > 0:
            chunks.append(verts_world)
    if not chunks:
        raise RuntimeError("No mesh vertices available for source axis inference")
    return np.concatenate(chunks, axis=0)


def slice_vertices(coords: np.ndarray, axis_index: int, side: str, ratio: float = 0.08) -> np.ndarray:
    axis_values = coords[:, axis_index]
    axis_min = float(axis_values.min())
    axis_max = float(axis_values.max())
    extent = max(axis_max - axis_min, 1e-8)
    margin = max(extent * ratio, 1e-5)
    if side == "min":
        mask = axis_values <= axis_min + margin
        if not np.any(mask):
            order = np.argsort(axis_values)
            return coords[order[: min(len(order), 32)]]
    else:
        mask = axis_values >= axis_max - margin
        if not np.any(mask):
            order = np.argsort(axis_values)
            return coords[order[-min(len(order), 32) :]]
    return coords[mask]


def projected_slice_area(points: np.ndarray, axis_indices: tuple[int, int]) -> float:
    if points.size == 0:
        return 0.0
    projected = points[:, axis_indices]
    if len(projected) >= 10:
        lo = np.quantile(projected, 0.05, axis=0)
        hi = np.quantile(projected, 0.95, axis=0)
    else:
        lo = projected.min(axis=0)
        hi = projected.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return float(span[0] * span[1])


def infer_auto_source_axes(mesh_objects: list[bpy.types.Object]) -> tuple[dict[str, str], dict[str, Any]]:
    coords = all_vertex_coords_world(mesh_objects)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extents = bbox_max - bbox_min

    axis_order = np.argsort(extents)
    up_index = int(axis_order[0])
    width_index = int(axis_order[1])
    length_index = int(axis_order[2])

    up_name = AXIS_INDEX_TO_NAME[up_index]
    width_name = AXIS_INDEX_TO_NAME[width_index]
    length_name = AXIS_INDEX_TO_NAME[length_index]

    up_projection_axes = tuple(idx for idx in range(3) if idx != up_index)
    up_min_slice = slice_vertices(coords, up_index, "min")
    up_max_slice = slice_vertices(coords, up_index, "max")
    up_min_area = projected_slice_area(up_min_slice, up_projection_axes)
    up_max_area = projected_slice_area(up_max_slice, up_projection_axes)
    up_sign = 1.0 if up_min_area >= up_max_area else -1.0

    length_projection_axes = (width_index, up_index)
    length_min_slice = slice_vertices(coords, length_index, "min")
    length_max_slice = slice_vertices(coords, length_index, "max")
    length_min_area = projected_slice_area(length_min_slice, length_projection_axes)
    length_max_area = projected_slice_area(length_max_slice, length_projection_axes)
    length_sign = 1.0 if length_max_area <= length_min_area else -1.0

    length_token = axis_token(length_name, length_sign)
    up_token = axis_token(up_name, up_sign)
    width_positive = axis_token(width_name, 1.0)
    width_negative = axis_token(width_name, -1.0)
    if np.linalg.det(source_axes_matrix({"length": length_token, "width": width_positive, "up": up_token})) > 0.0:
        width_token = width_positive
    else:
        width_token = width_negative

    resolved = {
        "length": length_token,
        "width": width_token,
        "up": up_token,
    }
    summary = {
        "mode": "auto",
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": extents.tolist(),
        "extent_rank": {
            "length_axis": length_name,
            "width_axis": width_name,
            "up_axis": up_name,
        },
        "slice_area_heuristics": {
            "up_min_area": up_min_area,
            "up_max_area": up_max_area,
            "length_min_area": length_min_area,
            "length_max_area": length_max_area,
        },
        "resolved_source_axes": resolved,
    }
    return resolved, summary


def resolve_source_axes(
    mesh_objects: list[bpy.types.Object],
    shoe_cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    source_axes_cfg = shoe_cfg.get("source_axes", "auto")
    if source_axes_cfg == "auto":
        return infer_auto_source_axes(mesh_objects)
    if isinstance(source_axes_cfg, dict):
        resolved = dict(source_axes_cfg)
        return resolved, {
            "mode": "provided",
            "resolved_source_axes": resolved,
        }
    raise ValueError(f"{shoe_cfg['name']}: unsupported source_axes config {source_axes_cfg!r}")


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
    axis_name = str(selection_cfg.get("axis", "Y")).upper()
    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis_name]
    mins = np.array([row["bbox_min"][axis_index] for row in records], dtype=np.float64)
    maxs = np.array([row["bbox_max"][axis_index] for row in records], dtype=np.float64)
    extent = float(maxs.max() - mins.min())
    threshold = max(extent * 1e-7, 1e-7)
    side = None
    pivot = None

    if mode == "axis-min":
        global_min = float(mins.min())
        keep_mask = np.abs(mins - global_min) <= threshold
    elif mode == "axis-side":
        centers = np.array([row["center"][axis_index] for row in records], dtype=np.float64)
        side = str(selection_cfg.get("side", "min")).lower()
        pivot = float(0.5 * (mins.min() + maxs.max()))
        if side == "min":
            keep_mask = centers <= pivot + threshold
        else:
            keep_mask = centers >= pivot - threshold
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

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
        "selection_side": side,
        "selection_pivot": pivot,
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


def bake_objects_to_world(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """Convert object-local mesh coordinates into world coordinates.

    GLB imports often preserve a node hierarchy. After loose-part separation, moving
    ``obj.location`` can be parent-local instead of world-local, which breaks the
    centering assumption used by the camera-fit math.
    """
    baked: list[bpy.types.Object] = []
    for obj in objects:
        world_matrix = obj.matrix_world.copy()
        det = float(world_matrix.to_3x3().determinant())
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
        obj.data.transform(world_matrix)
        if det < 0.0:
            flip_mesh_normals(obj)
        obj.data.update()
        baked.append(obj)
    bpy.context.view_layer.update()
    return baked


def bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )


def projected_bounds(
    points: np.ndarray,
    c2w: np.ndarray,
    scale: float,
) -> tuple[float, float, float, float]:
    points_h = np.concatenate(
        [points * float(scale), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_points = points_h @ np.linalg.inv(c2w).T
    depth = -camera_points[:, 2]
    if np.any(depth <= 1e-8):
        return (-math.inf, math.inf, -math.inf, math.inf)
    tan_x = math.tan(math.radians(FOV_X_DEG) * 0.5)
    tan_y = tan_x * (RESOLUTION_Y / RESOLUTION_X)
    u = 0.5 + camera_points[:, 0] / depth / (2.0 * tan_x)
    v = 0.5 + camera_points[:, 1] / depth / (2.0 * tan_y)
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def binary_search_scale(predicate: Any, upper: float) -> float:
    low = 0.0
    high = float(upper)
    for _ in range(48):
        middle = 0.5 * (low + high)
        if predicate(middle):
            low = middle
        else:
            high = middle
    return low


def production_camera_poses() -> list[np.ndarray]:
    poses = []
    for elevation_deg, radius in zip(MULTI_ELEVATIONS_DEG, RING_RADII):
        for azimuth_index in range(VIEWS_PER_RING):
            azimuth_deg = 90.0 - 10.0 * azimuth_index
            poses.append(c2w_from_eye(orbit_eye(radius, azimuth_deg, elevation_deg)))
    return poses


def normalization_scale(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mesh_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    centered_min = bbox_min.copy()
    centered_max = bbox_max.copy()
    corners = bbox_corners(centered_min, centered_max)
    half_diagonal = float(np.linalg.norm(0.5 * (centered_max - centered_min)))
    if half_diagonal <= 0.0:
        return centered_min, centered_max, CAMERA_RADIUS, 1.0

    upper = 0.98 * min(RING_RADII) / half_diagonal
    reference_pose = production_camera_poses()[0]

    def reference_width_fits(candidate_scale: float) -> bool:
        u_min, u_max, _, _ = projected_bounds(mesh_points, reference_pose, candidate_scale)
        return u_max - u_min <= REFERENCE_HORIZONTAL_OCCUPANCY

    target_scale = binary_search_scale(reference_width_fits, upper)
    margin = BORDER_MARGIN_FRACTION

    def every_view_fits(candidate_scale: float) -> bool:
        for pose in production_camera_poses():
            u_min, u_max, v_min, v_max = projected_bounds(corners, pose, candidate_scale)
            if min(u_min, v_min) < margin or max(u_max, v_max) > 1.0 - margin:
                return False
        return True

    safe_scale = binary_search_scale(every_view_fits, upper)
    scale = min(target_scale, safe_scale * CONSERVATIVE_FIT_RELAXATION)
    required_distance = CAMERA_RADIUS / scale
    return centered_min, centered_max, required_distance, scale


def canonicalize_geometry(
    mesh_objects: list[bpy.types.Object],
    shoe_cfg: dict[str, Any],
) -> dict[str, Any]:
    mesh_objects = bake_objects_to_world(mesh_objects)
    resolved_source_axes, source_axes_summary = resolve_source_axes(mesh_objects, shoe_cfg)
    axis_mat = source_axes_matrix(resolved_source_axes)
    apply_axis_transform(mesh_objects, axis_mat)

    selected_objects, selection_summary = apply_component_selection(
        [obj for obj in bpy.context.scene.objects if obj.type == "MESH"],
        shoe_cfg.get("selection"),
    )
    selected_objects = bake_objects_to_world(selected_objects)

    bbox_min_raw, bbox_max_raw = scene_bbox(selected_objects)
    center = 0.5 * (bbox_min_raw + bbox_max_raw)
    translate_objects(selected_objects, -center)
    bbox_min_centered, bbox_max_centered = scene_bbox(selected_objects)
    mesh_points_centered = all_vertex_coords_world(selected_objects)
    centered_min, centered_max, required_distance, scale = normalization_scale(
        bbox_min_centered,
        bbox_max_centered,
        mesh_points_centered,
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
        "resolved_source_axes": resolved_source_axes,
        "source_axes_summary": source_axes_summary,
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
            "camera_radii": list(RING_RADII),
            "reference_horizontal_occupancy": REFERENCE_HORIZONTAL_OCCUPANCY,
            "border_margin_fraction": BORDER_MARGIN_FRACTION,
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


def render_image_and_mask(
    camera: bpy.types.Object,
    c2w: np.ndarray,
    temp_dir: Path,
    image_path: Path,
    mask_path: Path,
) -> None:
    scene = bpy.context.scene
    camera.matrix_world = Matrix(c2w.tolist())

    rgba_path = temp_dir / "rgba.png"
    if rgba_path.exists():
        rgba_path.unlink()

    scene.render.filepath = str(rgba_path)
    bpy.ops.render.render(write_still=True)

    rgba_img = bpy.data.images.load(str(rgba_path), check_existing=False)
    width, height = rgba_img.size[:]
    rgba = np.array(rgba_img.pixels[:], dtype=np.float32).reshape(height, width, 4)
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + COMPOSITE_BACKGROUND_VALUE * (1.0 - alpha)
    mask = (alpha[..., 0] > 1e-4).astype(np.float32)
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise RuntimeError(f"Rendered shoe touches the image border: {image_path.name}")

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
        color_mode="BW",
    )

    bpy.data.images.remove(rgba_img)


def load_saved_mask(mask_path: Path) -> np.ndarray:
    mask_img = bpy.data.images.load(str(mask_path), check_existing=False)
    width, height = mask_img.size[:]
    rgba = np.array(mask_img.pixels[:], dtype=np.float32).reshape(height, width, 4)
    bpy.data.images.remove(mask_img)
    return rgba[..., 0] > 0.5


def count_large_mask_components(mask: np.ndarray) -> int:
    sampled = np.ascontiguousarray(mask[::PROBE_MASK_DOWNSAMPLE, ::PROBE_MASK_DOWNSAMPLE] > 0)
    total_foreground = int(sampled.sum())
    if total_foreground == 0:
        return 0

    min_area = max(PROBE_MIN_COMPONENT_PIXELS, int(total_foreground * PROBE_LARGE_COMPONENT_RATIO))
    height, width = sampled.shape
    seen = np.zeros_like(sampled, dtype=bool)
    large_components = 0

    ys, xs = np.nonzero(sampled)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = cy + dy
                nx = cx + dx
                if 0 <= ny < height and 0 <= nx < width and sampled[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if area >= min_area:
            large_components += 1
    return large_components


def summarize_probe_mask(mask: np.ndarray) -> dict[str, Any]:
    return {
        "foreground_ratio": float(mask.mean()),
        "large_component_count": int(count_large_mask_components(mask)),
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    }


def kept_component_records(selection_summary: dict[str, Any]) -> list[dict[str, Any]]:
    name_to_record = {
        str(row["name"]): row
        for row in selection_summary.get("components_before_selection", [])
    }
    records = []
    for name in selection_summary.get("kept_component_names", []):
        record = name_to_record.get(str(name))
        if record is not None:
            records.append(record)
    return records


def probe_candidate_report(
    shoe_name: str,
    source_model: Path,
    shoe_cfg: dict[str, Any],
    selection_cfg: dict[str, Any] | None,
    selection_debug_dir: Path | None,
) -> dict[str, Any]:
    candidate_label = selection_cfg_label(selection_cfg)
    probe_debug_dir = None
    if selection_debug_dir is not None:
        probe_debug_dir = selection_debug_dir / shoe_name / candidate_label
        if probe_debug_dir.exists():
            shutil.rmtree(probe_debug_dir)
        probe_debug_dir.mkdir(parents=True, exist_ok=True)

    probe_root = probe_debug_dir or Path(tempfile.mkdtemp(prefix=f"{shoe_name}_{candidate_label}_probe_"))
    try:
        reset_scene()
        mesh_objects = import_model(source_model)
        probe_cfg = clone_shoe_cfg_with_selection(shoe_cfg, selection_cfg)
        canonical = canonicalize_geometry(mesh_objects, probe_cfg)
        selection_summary = canonical["component_selection"]
        kept_records = kept_component_records(selection_summary)
        total_vertices = sum(int(row.get("vertices", 0)) for row in selection_summary.get("components_before_selection", []))
        kept_vertices = sum(int(row.get("vertices", 0)) for row in kept_records)
        vertex_ratio = float(kept_vertices / total_vertices) if total_vertices > 0 else 1.0

        ensure_camera()
        camera = bpy.context.scene.camera

        view_summaries = []
        for probe_index, (elevation_deg, azimuth_deg) in enumerate(AUTO_SELECTION_PROBE_VIEWS):
            probe_name = f"probe_{probe_index:02d}"
            image_path = probe_root / f"{probe_name}.jpg"
            mask_path = probe_root / f"{probe_name}.png"
            eye = orbit_eye(CAMERA_RADIUS, azimuth_deg, elevation_deg)
            c2w = c2w_from_eye(eye)
            render_image_and_mask(camera, c2w, probe_root, image_path, mask_path)
            mask = load_saved_mask(mask_path)
            view_summary = summarize_probe_mask(mask)
            view_summary["elevation_deg"] = float(elevation_deg)
            view_summary["azimuth_deg"] = float(azimuth_deg)
            view_summary["image_path"] = str(image_path) if probe_debug_dir is not None else None
            view_summary["mask_path"] = str(mask_path) if probe_debug_dir is not None else None
            view_summaries.append(view_summary)

        multi_component_views = sum(int(view["large_component_count"] > 1) for view in view_summaries)
        border_touch_views = sum(int(view["touches_border"]) for view in view_summaries)
        max_large_components = max((int(view["large_component_count"]) for view in view_summaries), default=0)
        min_foreground_ratio = min((float(view["foreground_ratio"]) for view in view_summaries), default=0.0)
        avg_foreground_ratio = float(np.mean([float(view["foreground_ratio"]) for view in view_summaries])) if view_summaries else 0.0

        passed = (
            max_large_components == 1
            and multi_component_views == 0
            and border_touch_views == 0
            and min_foreground_ratio > 0.0
        )
        if selection_cfg is not None:
            passed = passed and vertex_ratio >= PROBE_MIN_VERTEX_RATIO

        report = {
            "candidate": candidate_label,
            "selection": selection_cfg,
            "passed": bool(passed),
            "component_count_before_selection": int(selection_summary.get("component_count_before_selection", 0)),
            "kept_component_count": int(selection_summary.get("kept_component_count", 0)),
            "kept_vertex_ratio": vertex_ratio,
            "probe": {
                "view_count": len(view_summaries),
                "multi_component_views": int(multi_component_views),
                "border_touch_views": int(border_touch_views),
                "max_large_component_count": int(max_large_components),
                "min_foreground_ratio": min_foreground_ratio,
                "avg_foreground_ratio": avg_foreground_ratio,
                "views": view_summaries,
            },
        }
        if probe_debug_dir is not None:
            report["debug_dir"] = str(probe_debug_dir)
            json_dump(probe_debug_dir / "probe_summary.json", report)
        return report
    finally:
        if probe_debug_dir is None:
            shutil.rmtree(probe_root, ignore_errors=True)


def resolve_selection_for_render(
    shoe_cfg: dict[str, Any],
    source_model: Path,
    selection_debug_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_selection = shoe_cfg.get("selection")
    if requested_selection is not None:
        return dict(shoe_cfg), {
            "mode": "explicit",
            "requested_selection": requested_selection,
            "resolved_selection": requested_selection,
            "chosen_candidate": selection_cfg_label(requested_selection),
            "auto_probe": None,
        }

    candidate_reports: list[dict[str, Any]] = []
    no_split_report = probe_candidate_report(
        str(shoe_cfg["name"]),
        source_model,
        shoe_cfg,
        None,
        selection_debug_dir,
    )
    candidate_reports.append(no_split_report)
    if no_split_report["passed"]:
        return clone_shoe_cfg_with_selection(shoe_cfg, None), {
            "mode": "auto_keep_all",
            "requested_selection": None,
            "resolved_selection": None,
            "chosen_candidate": no_split_report["candidate"],
            "auto_probe": {
                "candidates": candidate_reports,
            },
        }

    chosen_report = None
    for selection_cfg in auto_selection_candidates()[1:]:
        report = probe_candidate_report(
            str(shoe_cfg["name"]),
            source_model,
            shoe_cfg,
            selection_cfg,
            selection_debug_dir,
        )
        candidate_reports.append(report)
        if report["passed"]:
            chosen_report = report
            break

    if chosen_report is None:
        return clone_shoe_cfg_with_selection(shoe_cfg, None), {
            "mode": "auto_fallback_no_split",
            "requested_selection": None,
            "resolved_selection": None,
            "chosen_candidate": no_split_report["candidate"],
            "auto_probe": {
                "candidates": candidate_reports,
            },
        }

    return clone_shoe_cfg_with_selection(shoe_cfg, chosen_report["selection"]), {
        "mode": "auto_split",
        "requested_selection": None,
        "resolved_selection": chosen_report["selection"],
        "chosen_candidate": chosen_report["candidate"],
        "auto_probe": {
            "candidates": candidate_reports,
        },
    }


def save_float_image(
    rgb: np.ndarray,
    path: Path,
    file_format: str,
    alpha_value: float,
    color_mode: str = "RGB",
) -> None:
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
    scene.render.image_settings.color_mode = color_mode
    image.save_render(str(path), scene=scene)
    scene.render.image_settings.file_format = old_format
    scene.render.image_settings.color_mode = old_color_mode
    bpy.data.images.remove(image)


def render_views(shoe_name: str, shoe_root: Path) -> int:
    image_dir = shoe_root / "images"
    mask_dir = shoe_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{shoe_name}_render_"))
    try:
        ensure_camera()
        camera = bpy.context.scene.camera
        frame_index = 1
        for elevation_deg, radius in zip(MULTI_ELEVATIONS_DEG, RING_RADII):
            for azimuth_index in range(VIEWS_PER_RING):
                azimuth_deg = 90.0 - 10.0 * azimuth_index
                c2w = c2w_from_eye(orbit_eye(radius, azimuth_deg, elevation_deg))
                basename = f"img{frame_index:03d}"
                render_image_and_mask(
                    camera,
                    c2w,
                    temp_dir,
                    image_dir / f"{basename}.jpg",
                    mask_dir / f"{basename}.png",
                )
                frame_index += 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return frame_index - 1


def render_one_shoe(shoe_cfg: dict[str, Any], args: argparse.Namespace) -> int:
    shoe_name = str(shoe_cfg["name"])
    source_model = args.source_root / shoe_cfg["model"]
    shoe_root = args.output_root / shoe_name
    if shoe_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output shoe dir exists; pass --overwrite: {shoe_root}")
        shutil.rmtree(shoe_root)
    shoe_root.mkdir(parents=True, exist_ok=True)

    resolved_shoe_cfg, selection = resolve_selection_for_render(
        shoe_cfg,
        source_model,
        args.selection_debug_dir,
    )

    reset_scene()
    mesh_objects = import_model(source_model)
    canonical = canonicalize_geometry(mesh_objects, resolved_shoe_cfg)
    print(
        f"[setup] {shoe_name}: selection={selection['chosen_candidate']}, "
        f"axes={canonical['resolved_source_axes']}"
    )
    return render_views(shoe_name, shoe_root)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest, args.source_root)
    selected = set(args.shoe or [])
    unknown_selected = sorted(selected - {str(shoe_cfg["name"]) for shoe_cfg in manifest["shoes"]})
    if unknown_selected:
        raise SystemExit(f"Selected shoe(s) not found in manifest: {', '.join(unknown_selected)}")

    errors: list[str] = []
    args.output_root.mkdir(parents=True, exist_ok=True)

    for shoe_cfg in manifest["shoes"]:
        shoe_name = str(shoe_cfg["name"])
        if selected and shoe_name not in selected:
            continue

        try:
            view_count = render_one_shoe(shoe_cfg, args)
            print(f"[ok] {shoe_name}: {view_count} images and masks")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            errors.append(f"{shoe_name}: {error}")
            print(f"[failed] {shoe_name}: {error}")

    if errors:
        raise SystemExit("Rendering failed for: " + "; ".join(errors))


if __name__ == "__main__":
    main()
