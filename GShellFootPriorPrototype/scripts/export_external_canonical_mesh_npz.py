#!/usr/bin/env python3
"""Export an evaluation asset in the exact canonical Blender space used for rendering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_tools.golden_set_evaluation import blender_renderer as evaluation_renderer  # noqa: E402


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--canonicalization-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv_after_double_dash())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def shoe_config(metadata: dict[str, Any]) -> dict[str, Any]:
    axes = metadata["source_axes"]
    config: dict[str, Any] = {
        "name": metadata["shoe"],
        "source_axes": {
            "length": axes["length_to_canonical_x"],
            "width": axes["width_to_canonical_y"],
            "up": axes["up_to_canonical_z"],
        },
    }
    selection = metadata.get("component_selection", {})
    if selection.get("mode") is not None:
        config["selection"] = {
            "mode": selection["mode"],
            "axis": selection["selection_axis"],
            "separate_loose_parts": bool(selection.get("separate_loose_parts", False)),
        }
        if selection.get("selection_side") is not None:
            config["selection"]["side"] = selection["selection_side"]
    return config


def evaluated_triangle_mesh(objects: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = evaluation_renderer.bpy.context.evaluated_depsgraph_get()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0

    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            world = evaluated.matrix_world
            object_vertices = np.asarray(
                [tuple(world @ vertex.co) for vertex in mesh.vertices],
                dtype=np.float32,
            )
            mesh.calc_loop_triangles()
            object_faces = np.asarray(
                [triangle.vertices[:] for triangle in mesh.loop_triangles],
                dtype=np.int32,
            )
            if object_vertices.size and object_faces.size:
                vertices.append(object_vertices)
                faces.append(object_faces + vertex_offset)
                vertex_offset += object_vertices.shape[0]
        finally:
            evaluated.to_mesh_clear()

    if not vertices or not faces:
        raise RuntimeError("Canonicalization produced no triangle mesh")
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def validate_bbox(vertices: np.ndarray, metadata: dict[str, Any]) -> None:
    expected_min = np.asarray(metadata["canonical_bbox"]["min"], dtype=np.float32)
    expected_max = np.asarray(metadata["canonical_bbox"]["max"], dtype=np.float32)
    actual_min = vertices.min(axis=0)
    actual_max = vertices.max(axis=0)
    if not (
        np.allclose(actual_min, expected_min, atol=1e-5, rtol=1e-5)
        and np.allclose(actual_max, expected_max, atol=1e-5, rtol=1e-5)
    ):
        raise RuntimeError(
            "Exported canonical bbox does not match the rendered dataset metadata: "
            f"actual=({actual_min.tolist()}, {actual_max.tolist()}), "
            f"expected=({expected_min.tolist()}, {expected_max.tolist()})"
        )


def main() -> None:
    args = parse_args()
    metadata = load_json(args.canonicalization_json.resolve())
    model = args.model.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)

    evaluation_renderer.reset_scene()
    imported = evaluation_renderer.import_model(model)
    canonical = evaluation_renderer.canonicalize_geometry(imported, shoe_config(metadata))
    vertices, faces = evaluated_triangle_mesh(canonical["objects"])
    validate_bbox(vertices, metadata)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices, faces=faces)
    print(
        f"Exported {vertices.shape[0]} vertices and {faces.shape[0]} triangles "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
