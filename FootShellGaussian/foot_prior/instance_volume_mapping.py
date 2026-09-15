"""Exact Checkpoint 11-C maps between canonical and instance volumes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from enum import IntEnum
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .alignment import transform_points
from .anatomical_volume import (
    BOUNDARY_ANKLE_TRANSITION,
    BOUNDARY_FOOT_SKIN,
    BOUNDARY_KNEE_TRUNCATION,
    BOUNDARY_LOWER_LEG_SKIN,
    BOUNDARY_OUTER_ENVELOPE,
    CanonicalAnatomicalVolume,
    map_volume_coordinates,
)
from .anatomy import array_digest
from .instance_volume_optimization import (
    B3_FINAL_MAXIMUM_CONDITION_NUMBER,
    B3_FINAL_MAXIMUM_SINGULAR_VALUE,
    B3_FINAL_MINIMUM_DETERMINANT,
    B3_FINAL_MINIMUM_SINGULAR_VALUE,
    B3_MAXIMUM_CORRECTION_RESOLUTIONS,
    B3_MAXIMUM_TRIANGLE_AREA_RATIO,
    B3_MINIMUM_TRIANGLE_AREA_RATIO,
    optimization_configuration,
)


CoordinateFrame = Literal[
    "posed_supr",
    "normalized_shoe",
    "original_shoe",
]

MAPPING_SPATIAL_AXIS_CELLS = 32
MAPPING_MAXIMUM_CELLS_PER_TETRAHEDRON = 512
MAPPING_MAXIMUM_PAIR_EVALUATIONS = 262_144
MAPPING_BARYCENTRIC_TOLERANCE = 1.0e-10
MAPPING_SPATIAL_TOLERANCE_RELATIVE = 1.0e-10
MAPPING_ROUND_TRIP_TOLERANCE_RELATIVE = 1.0e-9
MAPPING_WINDING_AMBIGUITY_TOLERANCE = 1.0e-6

_VALID_STATUSES = (
    "final_exact_target",
    "final_corrected_target",
)
_VALID_CONTAINMENT_STATUSES = (
    "contained_target_fit",
    "residual_target_fit",
)
_INSTANCE_ARRAY_NAMES = {
    "volume_vertices",
    "tetrahedra",
    "harmonic_r",
    "jacobian_determinants",
    "jacobian_singular_values",
    "condition_numbers",
    "target_correction_vectors",
}
_ANATOMICAL_LABELS = {
    BOUNDARY_FOOT_SKIN,
    BOUNDARY_ANKLE_TRANSITION,
    BOUNDARY_LOWER_LEG_SKIN,
}
_FRAME_TRANSFORMS = {
    "normalized_shoe": (
        "normalized_shoe_to_posed_supr",
        "posed_supr_to_normalized_shoe",
    ),
    "original_shoe": (
        "original_shoe_to_posed_supr",
        "posed_supr_to_original_shoe",
    ),
}


class VolumePointStatus(IntEnum):
    """Classification of one instance-volume point query."""

    VALID_INTERIOR = 0
    VALID_INTERNAL_TETRAHEDRON_BOUNDARY = 1
    VALID_ANATOMICAL_BOUNDARY = 2
    VALID_OUTER_BOUNDARY = 3
    VALID_KNEE_TRUNCATION = 4
    INSIDE_COMPUTATIONAL_ANATOMY = 10
    OUTSIDE_OUTER_ENVELOPE = 11
    NUMERICAL_AMBIGUITY = 12
    NONINJECTIVE_OVERLAP = 13
    NONFINITE_INPUT = 14


_MAPPABLE_STATUS_CODES = np.asarray(
    (
        VolumePointStatus.VALID_INTERIOR,
        VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY,
        VolumePointStatus.VALID_ANATOMICAL_BOUNDARY,
        VolumePointStatus.VALID_OUTER_BOUNDARY,
        VolumePointStatus.VALID_KNEE_TRUNCATION,
    ),
    dtype=np.int16,
)
_FOOTWEAR_SUPPORT_STATUS_CODES = np.asarray(
    (
        VolumePointStatus.VALID_INTERIOR,
        VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY,
        VolumePointStatus.VALID_ANATOMICAL_BOUNDARY,
        VolumePointStatus.VALID_OUTER_BOUNDARY,
    ),
    dtype=np.int16,
)


@dataclass(frozen=True)
class TetrahedralCoordinates:
    """Exact volume coordinates: one cell ID and four barycentric weights."""

    tetrahedron_indices: np.ndarray
    barycentric_weights: np.ndarray


@dataclass(frozen=True)
class VolumeMappingResult:
    """Batched inverse-map result with explicit per-point validity."""

    coordinates: TetrahedralCoordinates
    canonical_points: np.ndarray
    harmonic_r: np.ndarray
    status_codes: np.ndarray

    @property
    def mappable_mask(self) -> np.ndarray:
        return np.isin(self.status_codes, _MAPPABLE_STATUS_CODES)

    @property
    def footwear_support_mask(self) -> np.ndarray:
        return np.isin(self.status_codes, _FOOTWEAR_SUPPORT_STATUS_CODES)


@dataclass(frozen=True)
class _TetrahedronSpatialIndex:
    """Conservative uniform-grid index over tetrahedron AABBs."""

    minimum: np.ndarray
    maximum: np.ndarray
    origin: np.ndarray
    cell_size: float
    shape: np.ndarray
    bins: dict[tuple[int, int, int], np.ndarray]
    broad_tetrahedron_indices: np.ndarray

    def point_cells(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.floor((points - self.origin) / self.cell_size).astype(
            np.int64
        )
        in_bounds = np.all(coordinates >= 0, axis=1) & np.all(
            coordinates < self.shape, axis=1
        )
        return coordinates, in_bounds

    def candidates(self, cell: tuple[int, int, int]) -> np.ndarray:
        local = self.bins.get(cell)
        if local is None:
            return self.broad_tetrahedron_indices
        if not len(self.broad_tetrahedron_indices):
            return local
        return np.union1d(local, self.broad_tetrahedron_indices)


@dataclass(frozen=True)
class InstanceVolumeMap:
    """Validated exact map for one final Checkpoint 11-B3 volume."""

    shoe_name: str
    canonical_volume: CanonicalAnatomicalVolume
    instance_vertices: np.ndarray
    instance_inverse_matrices: np.ndarray
    spatial_index: _TetrahedronSpatialIndex
    boundary_face_labels: np.ndarray
    inner_triangles: np.ndarray
    outer_triangles: np.ndarray
    transforms: dict[str, np.ndarray]
    spatial_tolerance: float
    round_trip_tolerance: float
    instance_status: str
    instance_volume_json_digest: str
    instance_volume_vertices_digest: str
    containment_fit_json_digest: str

    def canonical_to_instance(
        self,
        tetrahedron_indices: np.ndarray,
        barycentric_weights: np.ndarray,
        *,
        output_frame: CoordinateFrame = "posed_supr",
    ) -> np.ndarray:
        """Evaluate chi_i for exact canonical tetrahedral coordinates."""

        _validate_coordinate_frame(output_frame)
        points = map_volume_coordinates(
            tetrahedron_indices,
            barycentric_weights,
            self.instance_vertices,
            self.canonical_volume.tetrahedra,
        )
        return self._from_posed_supr(points, output_frame)

    def instance_to_canonical(
        self,
        points: np.ndarray,
        *,
        input_frame: CoordinateFrame = "posed_supr",
    ) -> VolumeMappingResult:
        """Evaluate Phi_i and classify every supplied physical point."""

        _validate_coordinate_frame(input_frame)
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != (3,):
            raise ValueError("points must have shape (N, 3)")

        posed = np.full(values.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        if np.any(finite):
            posed[finite] = self._to_posed_supr(values[finite], input_frame)
        return self._locate_posed_points(posed, finite)

    def _to_posed_supr(
        self, points: np.ndarray, frame: CoordinateFrame
    ) -> np.ndarray:
        if frame == "posed_supr":
            return np.asarray(points, dtype=np.float64).copy()
        return transform_points(points, self.transforms[_FRAME_TRANSFORMS[frame][0]])

    def _from_posed_supr(
        self, points: np.ndarray, frame: CoordinateFrame
    ) -> np.ndarray:
        if frame == "posed_supr":
            return np.asarray(points, dtype=np.float64).copy()
        return transform_points(points, self.transforms[_FRAME_TRANSFORMS[frame][1]])

    def _locate_posed_points(
        self, points: np.ndarray, finite: np.ndarray
    ) -> VolumeMappingResult:
        count = len(points)
        cell_ids = np.full(count, -1, dtype=np.int64)
        weights = np.full((count, 4), np.nan, dtype=np.float64)
        status = np.full(
            count, int(VolumePointStatus.NONFINITE_INPUT), dtype=np.int16
        )
        found = np.zeros(count, dtype=bool)

        finite_indices = np.flatnonzero(finite)
        if len(finite_indices):
            coordinates, in_bounds = self.spatial_index.point_cells(
                points[finite_indices]
            )
            grouped: dict[tuple[int, int, int], list[int]] = {}
            for local_index in np.flatnonzero(in_bounds):
                cell = tuple(int(value) for value in coordinates[local_index])
                grouped.setdefault(cell, []).append(int(finite_indices[local_index]))
            for cell in sorted(grouped):
                query_indices = np.asarray(grouped[cell], dtype=np.int64)
                candidates = self.spatial_index.candidates(cell)
                if len(candidates):
                    self._locate_group(
                        points,
                        query_indices,
                        candidates,
                        cell_ids,
                        weights,
                        status,
                        found,
                    )

            unresolved = (
                finite
                & ~found
                & (status != int(VolumePointStatus.NONINJECTIVE_OVERLAP))
            )
            if np.any(unresolved):
                status[unresolved] = self._classify_unmapped(points[unresolved])

        canonical_points = np.full((count, 3), np.nan, dtype=np.float64)
        harmonic_r = np.full(count, np.nan, dtype=np.float64)
        if np.any(found):
            valid_indices = np.flatnonzero(found)
            valid_cells = cell_ids[valid_indices]
            valid_weights = weights[valid_indices]
            canonical_points[valid_indices] = map_volume_coordinates(
                valid_cells,
                valid_weights,
                self.canonical_volume.volume_vertices,
                self.canonical_volume.tetrahedra,
            )
            harmonic_r[valid_indices] = np.einsum(
                "ni,ni->n",
                valid_weights,
                self.canonical_volume.harmonic_r[
                    self.canonical_volume.tetrahedra[valid_cells]
                ],
            )
        return VolumeMappingResult(
            coordinates=TetrahedralCoordinates(cell_ids, weights),
            canonical_points=canonical_points,
            harmonic_r=harmonic_r,
            status_codes=status,
        )

    def _locate_group(
        self,
        points: np.ndarray,
        query_indices: np.ndarray,
        candidates: np.ndarray,
        cell_ids: np.ndarray,
        weights: np.ndarray,
        status: np.ndarray,
        found: np.ndarray,
    ) -> None:
        maximum_points = max(
            1, MAPPING_MAXIMUM_PAIR_EVALUATIONS // max(1, len(candidates))
        )
        for offset in range(0, len(query_indices), maximum_points):
            selected = query_indices[offset : offset + maximum_points]
            query = points[selected]
            in_aabb = np.all(
                query[:, None, :] >= self.spatial_index.minimum[candidates][None],
                axis=2,
            ) & np.all(
                query[:, None, :] <= self.spatial_index.maximum[candidates][None],
                axis=2,
            )
            query_local, candidate_local = np.nonzero(in_aabb)
            if not len(query_local):
                continue
            tetrahedron_ids = candidates[candidate_local]
            tetrahedra = self.canonical_volume.tetrahedra[tetrahedron_ids]
            differences = (
                query[query_local] - self.instance_vertices[tetrahedra[:, 0]]
            )
            last_three = np.einsum(
                "nij,nj->ni",
                self.instance_inverse_matrices[tetrahedron_ids],
                differences,
            )
            candidate_weights = np.column_stack(
                (1.0 - np.sum(last_three, axis=1), last_three)
            )
            tolerance = MAPPING_BARYCENTRIC_TOLERANCE
            accepted = np.all(candidate_weights >= -tolerance, axis=1) & np.all(
                candidate_weights <= 1.0 + tolerance, axis=1
            )
            if not np.any(accepted):
                continue
            query_local = query_local[accepted]
            tetrahedron_ids = tetrahedron_ids[accepted]
            candidate_weights = candidate_weights[accepted]
            candidate_weights = np.clip(candidate_weights, 0.0, 1.0)
            candidate_weights /= np.sum(candidate_weights, axis=1)[:, None]
            self._store_candidates(
                selected,
                query_local,
                tetrahedron_ids,
                candidate_weights,
                cell_ids,
                weights,
                status,
                found,
            )

    def _store_candidates(
        self,
        selected_queries: np.ndarray,
        query_local: np.ndarray,
        tetrahedron_ids: np.ndarray,
        candidate_weights: np.ndarray,
        cell_ids: np.ndarray,
        weights: np.ndarray,
        status: np.ndarray,
        found: np.ndarray,
    ) -> None:
        counts = np.bincount(query_local, minlength=len(selected_queries))
        starts = np.cumsum(counts) - counts
        single_local = np.flatnonzero(counts == 1)
        if len(single_local):
            pair_indices = starts[single_local]
            global_indices = selected_queries[single_local]
            chosen_cells = tetrahedron_ids[pair_indices]
            chosen_weights = candidate_weights[pair_indices]
            cell_ids[global_indices] = chosen_cells
            weights[global_indices] = chosen_weights
            status[global_indices] = np.asarray(
                [
                    int(self._boundary_status(cell, value))
                    for cell, value in zip(chosen_cells, chosen_weights)
                ],
                dtype=np.int16,
            )
            found[global_indices] = True

        for local_index in np.flatnonzero(counts > 1):
            start = int(starts[local_index])
            stop = start + int(counts[local_index])
            ids = tetrahedron_ids[start:stop]
            values = candidate_weights[start:stop]
            global_index = int(selected_queries[local_index])
            canonical = np.einsum(
                "ni,nij->nj",
                values,
                self.canonical_volume.volume_vertices[
                    self.canonical_volume.tetrahedra[ids]
                ],
            )
            disagreement = np.linalg.norm(canonical - canonical[0], axis=1)
            strictly_inside = np.any(
                np.min(values, axis=1) > MAPPING_BARYCENTRIC_TOLERANCE
            )
            if strictly_inside or float(np.max(disagreement)) > self.round_trip_tolerance:
                status[global_index] = int(
                    VolumePointStatus.NONINJECTIVE_OVERLAP
                )
                continue
            chosen_offset = int(np.argmin(ids))
            cell_ids[global_index] = int(ids[chosen_offset])
            weights[global_index] = values[chosen_offset]
            status[global_index] = int(self._multiple_boundary_status(ids, values))
            found[global_index] = True

    def _boundary_status(
        self, tetrahedron_index: int, weights: np.ndarray
    ) -> VolumePointStatus:
        zero = np.flatnonzero(weights <= MAPPING_BARYCENTRIC_TOLERANCE)
        if not len(zero):
            return VolumePointStatus.VALID_INTERIOR
        labels = self.boundary_face_labels[int(tetrahedron_index), zero]
        return _status_from_boundary_labels(labels)

    def _multiple_boundary_status(
        self, tetrahedron_indices: np.ndarray, weights: np.ndarray
    ) -> VolumePointStatus:
        labels: list[int] = []
        for tetrahedron_index, value in zip(tetrahedron_indices, weights):
            zero = np.flatnonzero(value <= MAPPING_BARYCENTRIC_TOLERANCE)
            labels.extend(
                int(label)
                for label in self.boundary_face_labels[
                    int(tetrahedron_index), zero
                ]
            )
        return _status_from_boundary_labels(np.asarray(labels, dtype=np.int16))

    def _classify_unmapped(self, points: np.ndarray) -> np.ndarray:
        outer = np.abs(_generalized_winding_numbers(points, self.outer_triangles))
        inner = np.abs(_generalized_winding_numbers(points, self.inner_triangles))
        result = np.full(
            len(points), int(VolumePointStatus.NUMERICAL_AMBIGUITY), dtype=np.int16
        )
        tolerance = MAPPING_WINDING_AMBIGUITY_TOLERANCE
        stable = (
            np.isfinite(outer)
            & np.isfinite(inner)
            & (np.abs(outer - 0.5) > tolerance)
            & (np.abs(inner - 0.5) > tolerance)
        )
        outside = stable & (outer < 0.5)
        inside_anatomy = stable & (outer > 0.5) & (inner > 0.5)
        result[outside] = int(VolumePointStatus.OUTSIDE_OUTER_ENVELOPE)
        result[inside_anatomy] = int(
            VolumePointStatus.INSIDE_COMPUTATIONAL_ANATOMY
        )
        return result


def mapping_configuration() -> dict[str, Any]:
    """Return the fixed, shoe-independent Checkpoint 11-C policy."""

    return {
        "spatial_index": "conservative_uniform_tetrahedron_aabb_grid",
        "spatial_axis_cells": MAPPING_SPATIAL_AXIS_CELLS,
        "maximum_cells_per_tetrahedron": (
            MAPPING_MAXIMUM_CELLS_PER_TETRAHEDRON
        ),
        "maximum_pair_evaluations_per_chunk": (
            MAPPING_MAXIMUM_PAIR_EVALUATIONS
        ),
        "barycentric_tolerance": MAPPING_BARYCENTRIC_TOLERANCE,
        "spatial_tolerance_relative_to_instance_diagonal": (
            MAPPING_SPATIAL_TOLERANCE_RELATIVE
        ),
        "round_trip_tolerance_relative_to_canonical_diagonal": (
            MAPPING_ROUND_TRIP_TOLERANCE_RELATIVE
        ),
        "winding_ambiguity_tolerance": MAPPING_WINDING_AMBIGUITY_TOLERANCE,
        "overlap_policy": "reject_disagreeing_canonical_reconstructions",
        "shared_boundary_policy": "smallest_tetrahedron_id",
        "invalid_point_policy": "classify_without_nearest_tetrahedron_snapping",
        "coordinate_frames": [
            "posed_supr",
            "normalized_shoe",
            "original_shoe",
        ],
    }


def load_instance_volume_map(
    canonical_volume: CanonicalAnatomicalVolume,
    instance_volume_directory: str | Path,
    containment_fit_directory: str | Path,
) -> InstanceVolumeMap:
    """Load one final B3 volume and build its exact Checkpoint 11-C map."""

    instance_directory = Path(instance_volume_directory).expanduser().resolve(
        strict=True
    )
    containment_directory = Path(containment_fit_directory).expanduser().resolve(
        strict=True
    )
    if not instance_directory.is_dir() or not containment_directory.is_dir():
        raise NotADirectoryError("instance and containment inputs must be directories")
    shoe_name = instance_directory.name
    if containment_directory.name != shoe_name:
        raise ValueError("instance volume and containment-fit shoe names disagree")

    json_path = instance_directory / "instance_volume.json"
    npz_path = instance_directory / "instance_volume.npz"
    vtk_path = instance_directory / "instance_volume.vtk"
    for path in (json_path, npz_path, vtk_path):
        if not path.is_file() or (path == vtk_path and path.stat().st_size == 0):
            raise FileNotFoundError(path)
    payload = _read_json_object(json_path)
    if (
        payload.get("schema_version") != 2
        or payload.get("stage") != "instance_volume_optimization"
        or payload.get("shoe_name") != shoe_name
        or payload.get("status") not in _VALID_STATUSES
        or payload.get("reached_beta") != 1.0
        or payload.get("configuration") != optimization_configuration()
    ):
        raise ValueError(f"{shoe_name}: instance volume is not a compatible final B3 result")

    digests = payload.get("digests")
    final = payload.get("final")
    if not isinstance(digests, dict) or not isinstance(final, dict):
        raise ValueError(f"{shoe_name}: instance metadata is incomplete")
    if digests.get("canonical_volume_topology_sha256") != canonical_volume.topology_digest:
        raise ValueError(f"{shoe_name}: canonical volume topology digest mismatch")
    if (
        final.get("inner_self_intersection_count") != 0
        or final.get("inner_outer_intersection_count") != 0
        or final.get("outer_boundary_exact") is not True
        or final.get("authoritative_anatomy_fidelity_passed") is not True
    ):
        raise ValueError(f"{shoe_name}: recorded final B3 acceptance is invalid")

    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != _INSTANCE_ARRAY_NAMES:
            raise ValueError(f"{shoe_name}: instance NPZ fields are invalid")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    vertices = np.asarray(arrays["volume_vertices"], dtype=np.float64)
    tetrahedra = np.asarray(arrays["tetrahedra"], dtype=np.int64)
    harmonic_r = np.asarray(arrays["harmonic_r"], dtype=np.float64)
    determinants = np.asarray(arrays["jacobian_determinants"], dtype=np.float64)
    singular_values = np.asarray(
        arrays["jacobian_singular_values"], dtype=np.float64
    )
    condition_numbers = np.asarray(arrays["condition_numbers"], dtype=np.float64)
    corrections = np.asarray(arrays["target_correction_vectors"], dtype=np.float64)
    cell_count = len(canonical_volume.tetrahedra)
    if (
        vertices.shape != canonical_volume.volume_vertices.shape
        or tetrahedra.shape != canonical_volume.tetrahedra.shape
        or harmonic_r.shape != canonical_volume.harmonic_r.shape
        or determinants.shape != (cell_count,)
        or singular_values.shape != (cell_count, 3)
        or condition_numbers.shape != (cell_count,)
        or corrections.shape
        != (len(canonical_volume.computational_inner_vertex_indices), 3)
        or not all(
            np.isfinite(array).all()
            for array in (
                vertices,
                harmonic_r,
                determinants,
                singular_values,
                condition_numbers,
                corrections,
            )
        )
        or not np.array_equal(tetrahedra, canonical_volume.tetrahedra)
        or not np.array_equal(harmonic_r, canonical_volume.harmonic_r)
        or not np.array_equal(
            vertices[canonical_volume.outer_vertex_indices],
            canonical_volume.volume_vertices[canonical_volume.outer_vertex_indices],
        )
    ):
        raise ValueError(f"{shoe_name}: instance volume arrays are invalid")
    vertices_digest = array_digest(vertices)
    if digests.get("instance_volume_vertices_sha256") != vertices_digest:
        raise ValueError(f"{shoe_name}: instance volume vertex digest mismatch")

    canonical_matrices = _tetrahedron_matrices(
        canonical_volume.volume_vertices, tetrahedra
    )
    canonical_inverse = np.linalg.inv(canonical_matrices)
    instance_matrices = _tetrahedron_matrices(vertices, tetrahedra)
    instance_inverse = np.linalg.inv(instance_matrices)
    deformation = instance_matrices @ canonical_inverse
    recomputed_determinants = np.linalg.det(deformation)
    recomputed_singular_values = np.linalg.svd(deformation, compute_uv=False)
    recomputed_condition = (
        recomputed_singular_values[:, 0] / recomputed_singular_values[:, 2]
    )
    if (
        not np.isfinite(instance_inverse).all()
        or not np.allclose(
            determinants, recomputed_determinants, atol=1.0e-12, rtol=1.0e-12
        )
        or not np.allclose(
            singular_values,
            recomputed_singular_values,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        or not np.allclose(
            condition_numbers,
            recomputed_condition,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        or float(np.min(recomputed_determinants)) < B3_FINAL_MINIMUM_DETERMINANT
        or float(np.min(recomputed_singular_values[:, 2]))
        < B3_FINAL_MINIMUM_SINGULAR_VALUE
        or float(np.max(recomputed_singular_values[:, 0]))
        > B3_FINAL_MAXIMUM_SINGULAR_VALUE
        or float(np.max(recomputed_condition))
        > B3_FINAL_MAXIMUM_CONDITION_NUMBER
    ):
        raise ValueError(f"{shoe_name}: recomputed B3 deformation quality is invalid")
    _validate_correction_diagnostics(final, corrections, shoe_name)

    containment_path = containment_directory / "containment_fit.json"
    if not containment_path.is_file():
        raise FileNotFoundError(containment_path)
    containment = _read_json_object(containment_path)
    if (
        containment.get("schema_version") != 5
        or containment.get("shoe_profile") != "normal"
        or containment.get("status") not in _VALID_CONTAINMENT_STATUSES
    ):
        raise ValueError(f"{shoe_name}: containment-fit metadata is incompatible")
    transforms = _load_transforms(containment.get("transforms"), shoe_name)

    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    instance_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    canonical_bounds = np.stack(
        (
            canonical_volume.volume_vertices.min(axis=0),
            canonical_volume.volume_vertices.max(axis=0),
        )
    )
    canonical_diagonal = float(np.linalg.norm(canonical_bounds[1] - canonical_bounds[0]))
    if instance_diagonal <= 0.0 or canonical_diagonal <= 0.0:
        raise ValueError(f"{shoe_name}: volume bounds are degenerate")
    spatial_tolerance = MAPPING_SPATIAL_TOLERANCE_RELATIVE * instance_diagonal
    round_trip_tolerance = MAPPING_ROUND_TRIP_TOLERANCE_RELATIVE * canonical_diagonal
    spatial_index = _build_tetrahedron_spatial_index(
        vertices, tetrahedra, spatial_tolerance
    )
    boundary_labels = _tetrahedron_boundary_labels(canonical_volume)
    inner_global_faces = canonical_volume.computational_inner_vertex_indices[
        canonical_volume.computational_inner_faces
    ]
    outer_faces = canonical_volume.boundary_faces[
        canonical_volume.boundary_labels == BOUNDARY_OUTER_ENVELOPE
    ]
    return InstanceVolumeMap(
        shoe_name=shoe_name,
        canonical_volume=canonical_volume,
        instance_vertices=vertices,
        instance_inverse_matrices=instance_inverse,
        spatial_index=spatial_index,
        boundary_face_labels=boundary_labels,
        inner_triangles=vertices[inner_global_faces],
        outer_triangles=vertices[outer_faces],
        transforms=transforms,
        spatial_tolerance=spatial_tolerance,
        round_trip_tolerance=round_trip_tolerance,
        instance_status=str(payload["status"]),
        instance_volume_json_digest=_file_digest(json_path),
        instance_volume_vertices_digest=vertices_digest,
        containment_fit_json_digest=_file_digest(containment_path),
    )


def _validate_coordinate_frame(frame: str) -> None:
    if frame not in {"posed_supr", "normalized_shoe", "original_shoe"}:
        raise ValueError(f"unsupported coordinate frame: {frame}")


def _tetrahedron_matrices(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)[tetrahedra]
    return np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )


def _build_tetrahedron_spatial_index(
    vertices: np.ndarray, tetrahedra: np.ndarray, tolerance: float
) -> _TetrahedronSpatialIndex:
    points = np.asarray(vertices, dtype=np.float64)[tetrahedra]
    minimum = points.min(axis=1) - tolerance
    maximum = points.max(axis=1) + tolerance
    origin = minimum.min(axis=0)
    upper = maximum.max(axis=0)
    maximum_extent = float(np.max(upper - origin))
    if not np.isfinite(maximum_extent) or maximum_extent <= 0.0:
        raise ValueError("instance volume has invalid spatial bounds")
    cell_size = maximum_extent / MAPPING_SPATIAL_AXIS_CELLS
    shape = np.maximum(1, np.ceil((upper - origin) / cell_size).astype(np.int64))
    bins: dict[tuple[int, int, int], list[int]] = {}
    broad: list[int] = []
    for tetrahedron_index, (lower, higher) in enumerate(zip(minimum, maximum)):
        first = np.clip(
            np.floor((lower - origin) / cell_size).astype(np.int64), 0, shape - 1
        )
        last = np.clip(
            np.floor((higher - origin) / cell_size).astype(np.int64), 0, shape - 1
        )
        counts = last - first + 1
        if int(np.prod(counts)) > MAPPING_MAXIMUM_CELLS_PER_TETRAHEDRON:
            broad.append(tetrahedron_index)
            continue
        for first_axis in range(int(first[0]), int(last[0]) + 1):
            for second_axis in range(int(first[1]), int(last[1]) + 1):
                for third_axis in range(int(first[2]), int(last[2]) + 1):
                    bins.setdefault(
                        (first_axis, second_axis, third_axis), []
                    ).append(tetrahedron_index)
    frozen = {
        cell: np.asarray(indices, dtype=np.int64) for cell, indices in bins.items()
    }
    return _TetrahedronSpatialIndex(
        minimum=minimum,
        maximum=maximum,
        origin=origin,
        cell_size=cell_size,
        shape=shape,
        bins=frozen,
        broad_tetrahedron_indices=np.asarray(broad, dtype=np.int64),
    )


def _tetrahedron_boundary_labels(
    canonical_volume: CanonicalAnatomicalVolume,
) -> np.ndarray:
    boundary = {
        tuple(sorted(int(value) for value in face)): int(label)
        for face, label in zip(
            canonical_volume.boundary_faces, canonical_volume.boundary_labels
        )
    }
    tetrahedra = canonical_volume.tetrahedra
    labels = np.zeros((len(tetrahedra), 4), dtype=np.int16)
    opposite_corners = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    for corner, other in enumerate(opposite_corners):
        for cell_index, face in enumerate(tetrahedra[:, other]):
            labels[cell_index, corner] = boundary.get(
                tuple(sorted(int(value) for value in face)), 0
            )
    return labels


def _status_from_boundary_labels(labels: np.ndarray) -> VolumePointStatus:
    values = set(int(value) for value in np.asarray(labels).ravel())
    if BOUNDARY_KNEE_TRUNCATION in values:
        return VolumePointStatus.VALID_KNEE_TRUNCATION
    if BOUNDARY_OUTER_ENVELOPE in values:
        return VolumePointStatus.VALID_OUTER_BOUNDARY
    if values.intersection(_ANATOMICAL_LABELS):
        return VolumePointStatus.VALID_ANATOMICAL_BOUNDARY
    return VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY


def _generalized_winding_numbers(
    points: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    queries = np.asarray(points, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.float64)
    result = np.zeros(len(queries), dtype=np.float64)
    point_chunk = 32
    triangle_chunk = 2_048
    for point_offset in range(0, len(queries), point_chunk):
        selected = queries[point_offset : point_offset + point_chunk]
        total = np.zeros(len(selected), dtype=np.float64)
        for triangle_offset in range(0, len(faces), triangle_chunk):
            current = faces[triangle_offset : triangle_offset + triangle_chunk]
            first = current[None, :, 0] - selected[:, None, :]
            second = current[None, :, 1] - selected[:, None, :]
            third = current[None, :, 2] - selected[:, None, :]
            first_norm = np.linalg.norm(first, axis=2)
            second_norm = np.linalg.norm(second, axis=2)
            third_norm = np.linalg.norm(third, axis=2)
            numerator = np.einsum(
                "pfi,pfi->pf", first, np.cross(second, third)
            )
            denominator = (
                first_norm * second_norm * third_norm
                + np.einsum("pfi,pfi->pf", first, second) * third_norm
                + np.einsum("pfi,pfi->pf", second, third) * first_norm
                + np.einsum("pfi,pfi->pf", third, first) * second_norm
            )
            total += np.sum(2.0 * np.arctan2(numerator, denominator), axis=1)
        result[point_offset : point_offset + len(selected)] = total / (4.0 * np.pi)
    return result


def _load_transforms(payload: Any, shoe_name: str) -> dict[str, np.ndarray]:
    if not isinstance(payload, dict):
        raise ValueError(f"{shoe_name}: containment transforms are missing")
    names = {
        name for pair in _FRAME_TRANSFORMS.values() for name in pair
    }
    result: dict[str, np.ndarray] = {}
    for name in names:
        matrix = np.asarray(payload.get(name), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{shoe_name}: transform {name} is invalid")
        result[name] = matrix
    for to_posed, from_posed in _FRAME_TRANSFORMS.values():
        if not (
            np.allclose(
                result[to_posed] @ result[from_posed],
                np.eye(4),
                atol=1.0e-12,
                rtol=1.0e-12,
            )
            and np.allclose(
                result[from_posed] @ result[to_posed],
                np.eye(4),
                atol=1.0e-12,
                rtol=1.0e-12,
            )
        ):
            raise ValueError(f"{shoe_name}: transform pair is not invertible")
    return result


def _validate_correction_diagnostics(
    final: dict[str, Any], corrections: np.ndarray, shoe_name: str
) -> None:
    resolution = final.get("surface_resolution")
    magnitude = final.get("target_correction_magnitude")
    area_ratio = final.get("corrected_triangle_area_ratio")
    if (
        not isinstance(resolution, (int, float))
        or not np.isfinite(resolution)
        or resolution <= 0.0
        or not isinstance(magnitude, dict)
        or not isinstance(area_ratio, dict)
    ):
        raise ValueError(f"{shoe_name}: final boundary diagnostics are invalid")
    maximum = float(np.max(np.linalg.norm(corrections, axis=1)))
    if (
        not np.isclose(
            magnitude.get("maximum"), maximum, atol=1.0e-12, rtol=1.0e-12
        )
        or maximum > B3_MAXIMUM_CORRECTION_RESOLUTIONS * float(resolution) + 1.0e-12
        or float(area_ratio.get("minimum", -np.inf))
        < B3_MINIMUM_TRIANGLE_AREA_RATIO
        or float(area_ratio.get("maximum", np.inf))
        > B3_MAXIMUM_TRIANGLE_AREA_RATIO
    ):
        raise ValueError(f"{shoe_name}: final boundary correction is invalid")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
