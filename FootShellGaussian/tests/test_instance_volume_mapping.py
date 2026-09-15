"""Focused tests for exact Checkpoint 11-C instance-volume mapping."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from foot_prior.anatomical_volume import (
    BOUNDARY_FOOT_SKIN,
    BOUNDARY_KNEE_TRUNCATION,
    BOUNDARY_OUTER_ENVELOPE,
    CanonicalAnatomicalVolume,
)
from foot_prior.anatomy import array_digest
from foot_prior.instance_volume_mapping import (
    InstanceVolumeMap,
    VolumePointStatus,
    _build_tetrahedron_spatial_index,
    _generalized_winding_numbers,
    _tetrahedron_matrices,
    load_instance_volume_map,
)
from foot_prior.instance_volume_optimization import optimization_configuration


def _canonical_volume(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    boundary_faces: np.ndarray | None = None,
    boundary_labels: np.ndarray | None = None,
) -> CanonicalAnatomicalVolume:
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    faces = (
        np.asarray(boundary_faces, dtype=np.int64)
        if boundary_faces is not None
        else np.empty((0, 3), dtype=np.int64)
    )
    labels = (
        np.asarray(boundary_labels, dtype=np.int16)
        if boundary_labels is not None
        else np.empty(0, dtype=np.int16)
    )
    matrices = _tetrahedron_matrices(vertices, tetrahedra)
    signed_volumes = np.linalg.det(matrices) / 6.0
    return CanonicalAnatomicalVolume(
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        boundary_faces=faces,
        boundary_labels=labels,
        harmonic_r=np.linspace(0.0, 1.0, len(vertices)),
        harmonic_r_gradient=np.zeros((len(tetrahedra), 3)),
        computational_inner_vertex_indices=np.arange(
            min(3, len(vertices)), dtype=np.int64
        ),
        computational_inner_faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        computational_inner_face_labels=np.asarray(
            (BOUNDARY_FOOT_SKIN,), dtype=np.int16
        ),
        zero_boundary_vertex_indices=np.empty(0, dtype=np.int64),
        knee_cap_face_indices=np.empty(0, dtype=np.int64),
        knee_cap_natural_vertex_indices=np.empty(0, dtype=np.int64),
        computational_to_canonical_face_indices=np.empty(0, dtype=np.int64),
        computational_to_canonical_barycentric=np.empty((0, 3)),
        computational_to_canonical_distances=np.empty(0),
        canonical_to_computational_face_indices=np.empty(0, dtype=np.int64),
        canonical_to_computational_barycentric=np.empty((0, 3)),
        canonical_to_computational_distances=np.empty(0),
        initial_self_intersection_pairs=np.empty((0, 2), dtype=np.int64),
        repair_zone_canonical_vertex_indices=np.empty(0, dtype=np.int64),
        outer_vertex_indices=np.asarray((0,), dtype=np.int64),
        tetrahedron_signed_volumes=signed_volumes,
        tetrahedron_mean_ratio_quality=np.ones(len(tetrahedra)),
        topology_digest="synthetic-topology",
        envelope_topology_digest="synthetic-envelope",
        extended_surface_digest="synthetic-surface",
        diagnostics={},
    )


def _map(
    canonical: CanonicalAnatomicalVolume,
    instance_vertices: np.ndarray,
    *,
    boundary_labels: np.ndarray | None = None,
    transforms: dict[str, np.ndarray] | None = None,
    inner_triangles: np.ndarray | None = None,
    outer_triangles: np.ndarray | None = None,
) -> InstanceVolumeMap:
    vertices = np.asarray(instance_vertices, dtype=np.float64)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    identity = np.eye(4, dtype=np.float64)
    matrices = {
        "normalized_shoe_to_posed_supr": identity,
        "posed_supr_to_normalized_shoe": identity,
        "original_shoe_to_posed_supr": identity,
        "posed_supr_to_original_shoe": identity,
    }
    if transforms is not None:
        matrices.update(transforms)
    return InstanceVolumeMap(
        shoe_name="synthetic",
        canonical_volume=canonical,
        instance_vertices=vertices,
        instance_inverse_matrices=np.linalg.inv(
            _tetrahedron_matrices(vertices, canonical.tetrahedra)
        ),
        spatial_index=_build_tetrahedron_spatial_index(
            vertices, canonical.tetrahedra, 1.0e-10 * diagonal
        ),
        boundary_face_labels=(
            np.asarray(boundary_labels, dtype=np.int16)
            if boundary_labels is not None
            else np.zeros((len(canonical.tetrahedra), 4), dtype=np.int16)
        ),
        inner_triangles=(
            np.asarray(inner_triangles, dtype=np.float64)
            if inner_triangles is not None
            else vertices[np.asarray(((0, 1, 2),))]
        ),
        outer_triangles=(
            np.asarray(outer_triangles, dtype=np.float64)
            if outer_triangles is not None
            else vertices[np.asarray(((0, 1, 2),))]
        ),
        transforms=matrices,
        spatial_tolerance=1.0e-10 * diagonal,
        round_trip_tolerance=1.0e-9 * diagonal,
        instance_status="final_exact_target",
        instance_volume_json_digest="instance-json",
        instance_volume_vertices_digest="instance-vertices",
        containment_fit_json_digest="containment-json",
    )


def test_affine_forward_inverse_and_harmonic_r() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    canonical = _canonical_volume(vertices, np.asarray(((0, 1, 2, 3),)))
    linear = np.asarray(((1.2, 0.1, 0.0), (0.0, 0.9, 0.1), (0.1, 0.0, 1.1)))
    translation = np.asarray((0.3, -0.2, 0.4))
    instance = vertices @ linear.T + translation
    mapping = _map(canonical, instance)
    weights = np.asarray(((0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)))
    ids = np.zeros(len(weights), dtype=np.int64)
    points = mapping.canonical_to_instance(ids, weights)
    result = mapping.instance_to_canonical(points)
    np.testing.assert_array_equal(result.coordinates.tetrahedron_indices, ids)
    np.testing.assert_allclose(result.coordinates.barycentric_weights, weights, atol=1e-14)
    np.testing.assert_allclose(
        result.canonical_points, weights @ canonical.volume_vertices, atol=1e-14
    )
    np.testing.assert_allclose(
        result.harmonic_r, weights @ canonical.harmonic_r, atol=1e-14
    )
    assert np.all(result.status_codes == VolumePointStatus.VALID_INTERIOR)


def test_shared_face_selects_smallest_tetrahedron_deterministically() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    tetrahedra = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64)
    canonical = _canonical_volume(vertices, tetrahedra)
    mapping = _map(canonical, vertices)
    points = np.asarray(((1.0 / 3.0, 1.0 / 3.0, 0.0), (0.0, 0.0, 0.0)))
    result = mapping.instance_to_canonical(points)
    np.testing.assert_array_equal(result.coordinates.tetrahedron_indices, (0, 0))
    assert np.all(
        result.status_codes
        == VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY
    )


def test_nonadjacent_overlap_is_rejected() -> None:
    first = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    canonical_vertices = np.vstack((first, first + np.asarray((2.0, 0.0, 0.0))))
    tetrahedra = np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int64)
    canonical = _canonical_volume(canonical_vertices, tetrahedra)
    instance = canonical_vertices.copy()
    instance[4:] = first
    mapping = _map(canonical, instance)
    result = mapping.instance_to_canonical(np.asarray(((0.25, 0.25, 0.25),)))
    assert result.status_codes[0] == VolumePointStatus.NONINJECTIVE_OVERLAP
    assert not result.mappable_mask[0]
    assert result.coordinates.tetrahedron_indices[0] == -1


def test_boundary_labels_and_nonfinite_rows() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    canonical = _canonical_volume(vertices, np.asarray(((0, 1, 2, 3),)))
    labels = np.asarray(
        ((BOUNDARY_KNEE_TRUNCATION, BOUNDARY_OUTER_ENVELOPE, BOUNDARY_FOOT_SKIN, 0),),
        dtype=np.int16,
    )
    mapping = _map(canonical, vertices, boundary_labels=labels)
    weights = np.asarray(
        (
            (0.0, 1 / 3, 1 / 3, 1 / 3),
            (1 / 3, 0.0, 1 / 3, 1 / 3),
            (1 / 3, 1 / 3, 0.0, 1 / 3),
            (1 / 3, 1 / 3, 1 / 3, 0.0),
        )
    )
    points = weights @ vertices
    points = np.vstack((points, (np.nan, 0.0, 0.0)))
    result = mapping.instance_to_canonical(points)
    np.testing.assert_array_equal(
        result.status_codes,
        (
            VolumePointStatus.VALID_KNEE_TRUNCATION,
            VolumePointStatus.VALID_OUTER_BOUNDARY,
            VolumePointStatus.VALID_ANATOMICAL_BOUNDARY,
            VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY,
            VolumePointStatus.NONFINITE_INPUT,
        ),
    )
    assert np.all(result.mappable_mask[:4]) and not result.mappable_mask[4]
    assert not result.footwear_support_mask[0]


def test_coordinate_frames_round_trip() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    canonical = _canonical_volume(vertices, np.asarray(((0, 1, 2, 3),)))
    forward = np.eye(4)
    forward[:3, :3] *= 2.0
    forward[:3, 3] = (0.2, -0.3, 0.4)
    inverse = np.linalg.inv(forward)
    transforms = {
        "posed_supr_to_normalized_shoe": forward,
        "normalized_shoe_to_posed_supr": inverse,
        "posed_supr_to_original_shoe": forward,
        "original_shoe_to_posed_supr": inverse,
    }
    mapping = _map(canonical, vertices, transforms=transforms)
    weights = np.asarray(((0.1, 0.2, 0.3, 0.4),))
    for frame in ("normalized_shoe", "original_shoe"):
        point = mapping.canonical_to_instance(
            np.asarray((0,)), weights, output_frame=frame
        )
        result = mapping.instance_to_canonical(point, input_frame=frame)
        np.testing.assert_allclose(result.coordinates.barycentric_weights, weights)
    with pytest.raises(ValueError, match="unsupported coordinate frame"):
        mapping.instance_to_canonical(vertices[:1], input_frame="wrong")  # type: ignore[arg-type]


def test_winding_classifies_closed_tetrahedron() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    winding = np.abs(
        _generalized_winding_numbers(
            np.asarray(((0.1, 0.1, 0.1), (2.0, 2.0, 2.0))), vertices[faces]
        )
    )
    assert winding[0] > 0.5
    assert winding[1] < 0.5


def test_unmapped_points_are_classified_without_snapping() -> None:
    outer_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    inner_vertices = np.asarray(
        ((0.2, 0.2, 0.2), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6))
    )
    canonical = _canonical_volume(
        outer_vertices, np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    )
    mapping = _map(
        canonical,
        outer_vertices,
        inner_triangles=inner_vertices[faces],
        outer_triangles=outer_vertices[faces],
    )
    status = mapping._classify_unmapped(
        np.asarray(((0.25, 0.25, 0.25), (3.0, 3.0, 3.0)))
    )
    np.testing.assert_array_equal(
        status,
        (
            VolumePointStatus.INSIDE_COMPUTATIONAL_ANATOMY,
            VolumePointStatus.OUTSIDE_OUTER_ENVELOPE,
        ),
    )


@pytest.fixture()
def synthetic_artifacts(tmp_path: Path) -> tuple[CanonicalAnatomicalVolume, Path, Path]:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    canonical = _canonical_volume(
        vertices,
        tetrahedra,
        boundary_faces=np.asarray(((1, 2, 3),)),
        boundary_labels=np.asarray((BOUNDARY_OUTER_ENVELOPE,)),
    )
    instance_directory = tmp_path / "instances" / "synthetic"
    containment_directory = tmp_path / "containment" / "synthetic"
    instance_directory.mkdir(parents=True)
    containment_directory.mkdir(parents=True)
    corrections = np.zeros((len(canonical.computational_inner_vertex_indices), 3))
    np.savez(
        instance_directory / "instance_volume.npz",
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        harmonic_r=canonical.harmonic_r,
        jacobian_determinants=np.ones(1),
        jacobian_singular_values=np.ones((1, 3)),
        condition_numbers=np.ones(1),
        target_correction_vectors=corrections,
    )
    metadata = {
        "schema_version": 2,
        "stage": "instance_volume_optimization",
        "status": "final_exact_target",
        "shoe_name": "synthetic",
        "reached_beta": 1.0,
        "configuration": optimization_configuration(),
        "digests": {
            "canonical_volume_topology_sha256": canonical.topology_digest,
            "instance_volume_vertices_sha256": array_digest(vertices),
        },
        "final": {
            "inner_self_intersection_count": 0,
            "inner_outer_intersection_count": 0,
            "outer_boundary_exact": True,
            "authoritative_anatomy_fidelity_passed": True,
            "surface_resolution": 1.0,
            "target_correction_magnitude": {"maximum": 0.0},
            "corrected_triangle_area_ratio": {"minimum": 1.0, "maximum": 1.0},
        },
    }
    (instance_directory / "instance_volume.json").write_text(json.dumps(metadata))
    (instance_directory / "instance_volume.vtk").write_text("synthetic")
    identity = np.eye(4).tolist()
    containment = {
        "schema_version": 5,
        "shoe_profile": "normal",
        "status": "residual_target_fit",
        "transforms": {
            "posed_supr_to_normalized_shoe": identity,
            "normalized_shoe_to_posed_supr": identity,
            "posed_supr_to_original_shoe": identity,
            "original_shoe_to_posed_supr": identity,
        },
    }
    (containment_directory / "containment_fit.json").write_text(json.dumps(containment))
    return canonical, instance_directory, containment_directory


def test_loader_validates_synthetic_artifacts(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    mapping = load_instance_volume_map(canonical, instance, containment)
    assert mapping.shoe_name == "synthetic"
    assert len(mapping.spatial_index.broad_tetrahedron_indices) == 1


def test_loader_rejects_nonfinal_metadata(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    path = instance / "instance_volume.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = "failed_11_b3"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not a compatible final B3"):
        load_instance_volume_map(canonical, instance, containment)


def test_loader_rejects_changed_outer_vertex(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    canonical = replace(canonical, volume_vertices=canonical.volume_vertices.copy())
    canonical.volume_vertices[0, 0] = 0.1
    with pytest.raises(ValueError, match="arrays are invalid"):
        load_instance_volume_map(canonical, instance, containment)


def test_loader_rejects_changed_harmonic_field(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    npz_path = instance / "instance_volume.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["harmonic_r"] = arrays["harmonic_r"].copy()
    arrays["harmonic_r"][1] += 0.1
    np.savez(npz_path, **arrays)
    with pytest.raises(ValueError, match="arrays are invalid"):
        load_instance_volume_map(canonical, instance, containment)


def test_loader_rejects_changed_vertex_digest(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    path = instance / "instance_volume.json"
    metadata = json.loads(path.read_text())
    metadata["digests"]["instance_volume_vertices_sha256"] = "changed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="vertex digest mismatch"):
        load_instance_volume_map(canonical, instance, containment)


def test_loader_rejects_invalid_transform_pair(
    synthetic_artifacts: tuple[CanonicalAnatomicalVolume, Path, Path]
) -> None:
    canonical, instance, containment = synthetic_artifacts
    path = containment / "containment_fit.json"
    metadata = json.loads(path.read_text())
    metadata["transforms"]["original_shoe_to_posed_supr"][0][0] = 2.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="transform pair is not invertible"):
        load_instance_volume_map(canonical, instance, containment)
