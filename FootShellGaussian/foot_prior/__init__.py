"""Foot-shape prior utilities for FootShellGaussian.

These helpers are intentionally independent from the GShell training loop for
now. They give us a small place to grow SUPR-Foot loading and SDF constraints
without disturbing the copied baseline code.
"""

from .foot_sdf import (
    FootMeshForSDF,
    FootSDFBuildConfig,
    FootSDFConfig,
    FootSDFGrid,
    build_and_save_signed_sdf_from_obj,
    build_signed_sdf_grid_from_mesh,
    cap_single_boundary_loop,
    find_boundary_loops,
    load_obj_mesh,
    save_signed_sdf_npz,
)
from .foot_alignment import (
    FootAlignment,
    FootAlignmentConfig,
    MeshData,
    build_alignment_from_meshes,
    classify_shoe_points,
    compact_mesh,
    colors_from_regions,
    get_single_boundary_loop,
    load_triangle_mesh,
    make_hybrid_mesh,
    mesh_bounds,
    point_to_polyline_distance,
    query_foot_sdf_in_shoe_space,
    region_summary,
    remap_supr_to_shoe_axes,
    select_faces_by_centroid_z,
    transform_points,
    write_colored_ply,
    write_obj_mesh,
)
from .supr_foot import FootMesh, load_supr_foot_posed, load_supr_foot_template

__all__ = [
    "FootAlignment",
    "FootAlignmentConfig",
    "FootMesh",
    "FootMeshForSDF",
    "FootSDFBuildConfig",
    "FootSDFConfig",
    "FootSDFGrid",
    "MeshData",
    "build_alignment_from_meshes",
    "build_and_save_signed_sdf_from_obj",
    "build_signed_sdf_grid_from_mesh",
    "cap_single_boundary_loop",
    "classify_shoe_points",
    "compact_mesh",
    "colors_from_regions",
    "find_boundary_loops",
    "get_single_boundary_loop",
    "load_supr_foot_posed",
    "load_supr_foot_template",
    "load_obj_mesh",
    "load_triangle_mesh",
    "make_hybrid_mesh",
    "mesh_bounds",
    "point_to_polyline_distance",
    "query_foot_sdf_in_shoe_space",
    "region_summary",
    "remap_supr_to_shoe_axes",
    "save_signed_sdf_npz",
    "select_faces_by_centroid_z",
    "transform_points",
    "write_colored_ply",
    "write_obj_mesh",
]
