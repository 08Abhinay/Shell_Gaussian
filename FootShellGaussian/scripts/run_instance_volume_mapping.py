#!/usr/bin/env python3
"""Validate exact Checkpoint 11-C maps for final instance volumes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from foot_prior.anatomical_volume import (
    BOUNDARY_ANKLE_TRANSITION,
    BOUNDARY_FOOT_SKIN,
    BOUNDARY_KNEE_TRUNCATION,
    BOUNDARY_LOWER_LEG_SKIN,
    BOUNDARY_OUTER_ENVELOPE,
    ENVELOPE_CENTER,
    ENVELOPE_POWER,
    ENVELOPE_RADII,
    CanonicalAnatomicalVolume,
    load_canonical_anatomical_volume,
    map_volume_coordinates,
)
from foot_prior.instance_volume_mapping import (
    InstanceVolumeMap,
    VolumePointStatus,
    load_instance_volume_map,
    mapping_configuration,
)
from foot_prior.mesh import load_triangle_mesh


_INTERIOR_WEIGHTS = np.asarray(
    (
        (0.25, 0.25, 0.25, 0.25),
        (0.55, 0.15, 0.15, 0.15),
        (0.15, 0.55, 0.15, 0.15),
        (0.15, 0.15, 0.55, 0.15),
        (0.15, 0.15, 0.15, 0.55),
    ),
    dtype=np.float64,
)
_ANATOMICAL_LABELS = {
    BOUNDARY_FOOT_SKIN,
    BOUNDARY_ANKLE_TRANSITION,
    BOUNDARY_LOWER_LEG_SKIN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anatomical-volume-root", type=Path, required=True)
    parser.add_argument("--instance-volume-batch-root", type=Path, required=True)
    parser.add_argument("--containment-fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("shoes", nargs="*")
    return parser.parse_args()


_WORKER_CANONICAL: CanonicalAnatomicalVolume | None = None
_WORKER_BATCH_ROOT: Path | None = None
_WORKER_CONTAINMENT_ROOT: Path | None = None


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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selected_shoes(
    batch_root: Path, requested: list[str], excluded: list[str]
) -> tuple[list[str], dict[str, Any]]:
    manifest_path = batch_root / "batch_manifest.json"
    manifest = _read_json_object(manifest_path)
    shoes = manifest.get("shoes")
    if (
        manifest.get("stage") != "instance_volume_optimization_batch"
        or not isinstance(shoes, list)
        or not shoes
        or not all(isinstance(name, str) and name for name in shoes)
        or len(set(shoes)) != len(shoes)
    ):
        raise ValueError("instance-volume batch manifest is invalid")
    available = set(shoes)
    excluded_names = set(excluded)
    unknown_excluded = excluded_names.difference(available | {"sneaker_vibe"})
    if unknown_excluded:
        raise ValueError(f"unknown excluded shoes: {sorted(unknown_excluded)}")
    selected = requested if requested else list(shoes)
    if len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError("requested shoes are duplicated or absent from the B3 batch")
    selected = sorted(name for name in selected if name not in excluded_names)
    if "sneaker_vibe" in selected:
        raise ValueError("sneaker_vibe is excluded from Checkpoint 11-C")
    if not selected:
        raise ValueError("no instance volumes were selected")
    return selected, manifest


def _preflight_paths(
    names: list[str], batch_root: Path, containment_root: Path
) -> None:
    for name in names:
        instance_directory = batch_root / name
        containment_directory = containment_root / name
        if (
            not instance_directory.is_dir()
            or instance_directory.parent != batch_root
            or not containment_directory.is_dir()
            or containment_directory.parent != containment_root
        ):
            raise ValueError(f"{name}: input directories are missing or not direct children")
        instance_json = instance_directory / "instance_volume.json"
        containment_json = containment_directory / "containment_fit.json"
        for path in (
            instance_json,
            instance_directory / "instance_volume.npz",
            instance_directory / "instance_volume.vtk",
            containment_json,
            containment_directory / "foot_containment_fitted.ply",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        instance = _read_json_object(instance_json)
        containment = _read_json_object(containment_json)
        if (
            instance.get("schema_version") != 2
            or instance.get("stage") != "instance_volume_optimization"
            or instance.get("shoe_name") != name
            or instance.get("status")
            not in {"final_exact_target", "final_corrected_target"}
            or instance.get("reached_beta") != 1.0
        ):
            raise ValueError(f"{name}: B3 metadata is not final")
        if (
            containment.get("schema_version") != 5
            or containment.get("shoe_profile") != "normal"
        ):
            raise ValueError(f"{name}: containment metadata is incompatible")


def _summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("mapping error summary requires finite nonempty values")
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "p99": float(np.percentile(data, 99.0)),
        "maximum": float(np.max(data)),
    }


def _status_counts(*arrays: np.ndarray) -> dict[str, int]:
    values = np.concatenate([np.asarray(array, dtype=np.int16) for array in arrays])
    unique, counts = np.unique(values, return_counts=True)
    return {
        VolumePointStatus(int(code)).name.lower(): int(count)
        for code, count in zip(unique, counts)
    }


def _unique_tetrahedron_faces(tetrahedra: np.ndarray) -> np.ndarray:
    cells = np.asarray(tetrahedra, dtype=np.int64)
    faces = np.concatenate(
        (
            cells[:, (1, 2, 3)],
            cells[:, (0, 2, 3)],
            cells[:, (0, 1, 3)],
            cells[:, (0, 1, 2)],
        ),
        axis=0,
    )
    return np.unique(np.sort(faces, axis=1), axis=0)


def _boundary_label_lookup(volume: CanonicalAnatomicalVolume) -> dict[tuple[int, int, int], int]:
    return {
        tuple(sorted(int(value) for value in face)): int(label)
        for face, label in zip(volume.boundary_faces, volume.boundary_labels)
    }


def _expected_face_status(label: int) -> VolumePointStatus:
    if label == BOUNDARY_KNEE_TRUNCATION:
        return VolumePointStatus.VALID_KNEE_TRUNCATION
    if label == BOUNDARY_OUTER_ENVELOPE:
        return VolumePointStatus.VALID_OUTER_BOUNDARY
    if label in _ANATOMICAL_LABELS:
        return VolumePointStatus.VALID_ANATOMICAL_BOUNDARY
    return VolumePointStatus.VALID_INTERNAL_TETRAHEDRON_BOUNDARY


def _audit_normalized_anatomy_alignment(
    mapping: InstanceVolumeMap,
    instance_directory: Path,
    containment_directory: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    volume = mapping.canonical_volume
    instance_payload = _read_json_object(instance_directory / "instance_volume.json")
    fitted_path = Path(instance_payload.get("inputs", {}).get("fitted_surface", ""))
    if not fitted_path.is_file():
        raise FileNotFoundError(fitted_path)
    fitted = load_triangle_mesh(fitted_path)
    containment_foot = load_triangle_mesh(
        containment_directory / "foot_containment_fitted.ply"
    )
    if (
        len(fitted.vertices) < len(containment_foot.vertices)
        or not np.array_equal(
            fitted.vertices[: len(containment_foot.vertices)],
            containment_foot.vertices,
        )
    ):
        raise RuntimeError(
            f"{mapping.shoe_name}: containment foot was not preserved in fitted anatomy"
        )

    inner = mapping.instance_vertices[volume.computational_inner_vertex_indices]
    reconstructed = np.einsum(
        "ni,nij->nj",
        volume.canonical_to_computational_barycentric,
        inner[
            volume.computational_inner_faces[
                volume.canonical_to_computational_face_indices
            ]
        ],
    )
    distances = np.linalg.norm(reconstructed - fitted.vertices, axis=1)
    repair_zone = np.zeros(len(distances), dtype=bool)
    repair_zone[volume.repair_zone_canonical_vertex_indices] = True
    preserved = distances[~repair_zone]
    repaired = distances[repair_zone]
    resolution = float(instance_payload.get("final", {}).get("surface_resolution", np.nan))
    if (
        not np.isfinite(resolution)
        or resolution <= 0.0
        or float(np.percentile(preserved, 99.0)) > 0.5 * resolution
        or float(np.max(preserved)) > resolution
        or float(np.percentile(repaired, 99.0)) > 2.0 * resolution
        or float(np.max(repaired)) > 2.0 * resolution
    ):
        raise RuntimeError(
            f"{mapping.shoe_name}: normalized fitted anatomy and B3 boundary disagree"
        )

    containment = _read_json_object(containment_directory / "containment_fit.json")
    shoe_path = Path(containment.get("inputs", {}).get("normalized_shoe", ""))
    if not shoe_path.is_file():
        raise FileNotFoundError(shoe_path)
    shoe = load_triangle_mesh(shoe_path)
    count = min(4096, len(shoe.vertices))
    indices = np.linspace(0, len(shoe.vertices) - 1, count, dtype=np.int64)
    samples = shoe.vertices[indices]
    if not np.array_equal(
        mapping.convert_points(
            samples,
            input_frame="normalized_shoe",
            output_frame="normalized_shoe",
        ),
        samples,
    ):
        raise RuntimeError(f"{mapping.shoe_name}: normalized frame is not identity")
    shoe_result = mapping.instance_to_canonical(
        samples, input_frame="normalized_shoe"
    )
    return (
        {
            "surface_resolution": resolution,
            "all_reverse_distance": _summary(distances),
            "preserved_reverse_distance": _summary(preserved),
            "repair_zone_reverse_distance": _summary(repaired),
            "containment_native_vertices_preserved": True,
            "normalized_frame_identity": True,
            "fidelity_limits_passed": True,
        },
        _status_counts(shoe_result.status_codes),
    )


def _audit_mapping(
    mapping: InstanceVolumeMap,
    instance_directory: Path,
    containment_directory: Path,
) -> dict[str, Any]:
    volume = mapping.canonical_volume
    cell_count = len(volume.tetrahedra)
    cell_ids = np.repeat(np.arange(cell_count, dtype=np.int64), len(_INTERIOR_WEIGHTS))
    weights = np.tile(_INTERIOR_WEIGHTS, (cell_count, 1))
    physical = mapping.canonical_to_instance(cell_ids, weights)
    interior = mapping.instance_to_canonical(physical)
    expected_canonical = map_volume_coordinates(
        cell_ids, weights, volume.volume_vertices, volume.tetrahedra
    )
    recovered_physical = mapping.canonical_to_instance(
        interior.coordinates.tetrahedron_indices,
        interior.coordinates.barycentric_weights,
    )
    expected_r = np.einsum(
        "ni,ni->n", weights, volume.harmonic_r[volume.tetrahedra[cell_ids]]
    )
    interior_canonical_error = np.linalg.norm(
        interior.canonical_points - expected_canonical, axis=1
    )
    interior_physical_error = np.linalg.norm(recovered_physical - physical, axis=1)
    interior_weight_error = np.max(
        np.abs(interior.coordinates.barycentric_weights - weights), axis=1
    )
    interior_r_error = np.abs(interior.harmonic_r - expected_r)
    if (
        not np.all(
            interior.status_codes == int(VolumePointStatus.VALID_INTERIOR)
        )
        or not np.array_equal(interior.coordinates.tetrahedron_indices, cell_ids)
        or float(np.max(interior_canonical_error)) > mapping.round_trip_tolerance
        or float(np.max(interior_physical_error)) > mapping.round_trip_tolerance
        or float(np.max(interior_weight_error)) > 1.0e-9
        or float(np.max(interior_r_error)) > 1.0e-9
    ):
        raise RuntimeError(f"{mapping.shoe_name}: interior round-trip audit failed")

    vertex_result = mapping.instance_to_canonical(mapping.instance_vertices)
    vertex_canonical_error = np.linalg.norm(
        vertex_result.canonical_points - volume.volume_vertices, axis=1
    )
    if (
        not np.all(vertex_result.mappable_mask)
        or float(np.max(vertex_canonical_error)) > mapping.round_trip_tolerance
    ):
        raise RuntimeError(f"{mapping.shoe_name}: volume-vertex audit failed")

    unique_faces = _unique_tetrahedron_faces(volume.tetrahedra)
    physical_face_centroids = mapping.instance_vertices[unique_faces].mean(axis=1)
    canonical_face_centroids = volume.volume_vertices[unique_faces].mean(axis=1)
    face_result = mapping.instance_to_canonical(physical_face_centroids)
    face_canonical_error = np.linalg.norm(
        face_result.canonical_points - canonical_face_centroids, axis=1
    )
    boundary_lookup = _boundary_label_lookup(volume)
    labels = np.asarray(
        [boundary_lookup.get(tuple(int(value) for value in face), 0) for face in unique_faces],
        dtype=np.int16,
    )
    expected_face_status = np.asarray(
        [int(_expected_face_status(int(label))) for label in labels], dtype=np.int16
    )
    if (
        not np.all(face_result.mappable_mask)
        or not np.array_equal(face_result.status_codes, expected_face_status)
        or float(np.max(face_canonical_error)) > mapping.round_trip_tolerance
    ):
        raise RuntimeError(f"{mapping.shoe_name}: shared-face audit failed")

    representative_ids = np.unique(
        np.linspace(0, cell_count - 1, 256, dtype=np.int64)
    )
    representative_weights = np.broadcast_to(
        _INTERIOR_WEIGHTS[0], (len(representative_ids), 4)
    ).copy()
    frame_diagnostics: dict[str, Any] = {}
    for frame in ("normalized_shoe", "posed_supr", "original_shoe"):
        framed = mapping.canonical_to_instance(
            representative_ids, representative_weights, output_frame=frame
        )
        recovered = mapping.instance_to_canonical(framed, input_frame=frame)
        frame_physical = mapping.canonical_to_instance(
            recovered.coordinates.tetrahedron_indices,
            recovered.coordinates.barycentric_weights,
            output_frame=frame,
        )
        canonical_error = np.linalg.norm(
            recovered.canonical_points
            - map_volume_coordinates(
                representative_ids,
                representative_weights,
                volume.volume_vertices,
                volume.tetrahedra,
            ),
            axis=1,
        )
        physical_error = np.linalg.norm(frame_physical - framed, axis=1)
        if (
            not np.array_equal(
                recovered.coordinates.tetrahedron_indices, representative_ids
            )
            or float(np.max(canonical_error)) > mapping.round_trip_tolerance
            or float(np.max(physical_error)) > mapping.round_trip_tolerance
        ):
            raise RuntimeError(f"{mapping.shoe_name}: {frame} round-trip failed")
        frame_diagnostics[frame] = {
            "query_count": int(len(framed)),
            "canonical_round_trip_error": _summary(canonical_error),
            "physical_round_trip_error": _summary(physical_error),
        }

    outer_vertices = mapping.instance_vertices[volume.outer_vertex_indices]
    envelope_equation = np.sum(
        np.abs((outer_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    overlap_count = int(
        np.count_nonzero(
            np.concatenate(
                (interior.status_codes, vertex_result.status_codes, face_result.status_codes)
            )
            == int(VolumePointStatus.NONINJECTIVE_OVERLAP)
        )
    )
    if overlap_count:
        raise RuntimeError(f"{mapping.shoe_name}: noninjective overlap detected")

    anatomy_alignment, shoe_status_counts = _audit_normalized_anatomy_alignment(
        mapping, instance_directory, containment_directory
    )

    inner_face_count = int(np.count_nonzero(np.isin(labels, list(_ANATOMICAL_LABELS))))
    knee_face_count = int(np.count_nonzero(labels == BOUNDARY_KNEE_TRUNCATION))
    outer_face_count = int(np.count_nonzero(labels == BOUNDARY_OUTER_ENVELOPE))
    return {
        "schema_version": 2,
        "stage": "instance_volume_mapping_validation",
        "status": "mapping_valid",
        "shoe_name": mapping.shoe_name,
        "instance_volume_status": mapping.instance_status,
        "configuration": mapping_configuration(),
        "inputs": {
            "canonical_volume_topology_sha256": volume.topology_digest,
            "instance_volume_json_sha256": mapping.instance_volume_json_digest,
            "instance_volume_vertices_sha256": mapping.instance_volume_vertices_digest,
            "containment_fit_json_sha256": mapping.containment_fit_json_digest,
        },
        "coordinate_transforms": {
            name: matrix.tolist() for name, matrix in sorted(mapping.transforms.items())
        },
        "counts": {
            "tetrahedra": cell_count,
            "strictly_interior_queries": int(len(physical)),
            "volume_vertex_queries": int(len(mapping.instance_vertices)),
            "unique_tetrahedron_face_queries": int(len(unique_faces)),
            "anatomical_boundary_face_queries": inner_face_count,
            "knee_truncation_face_queries": knee_face_count,
            "outer_boundary_face_queries": outer_face_count,
            "frame_queries_per_frame": int(len(representative_ids)),
        },
        "query_status_counts": _status_counts(
            interior.status_codes, vertex_result.status_codes, face_result.status_codes
        ),
        "interior_round_trip": {
            "same_tetrahedron_ids": True,
            "canonical_error": _summary(interior_canonical_error),
            "physical_error": _summary(interior_physical_error),
            "barycentric_weight_error": _summary(interior_weight_error),
            "harmonic_r_error": _summary(interior_r_error),
        },
        "volume_vertices": {
            "all_mappable": True,
            "canonical_error": _summary(vertex_canonical_error),
        },
        "shared_faces": {
            "all_mappable": True,
            "boundary_labels_match": True,
            "canonical_error": _summary(face_canonical_error),
        },
        "coordinate_frames": frame_diagnostics,
        "normalized_anatomy_alignment": anatomy_alignment,
        "normalized_shoe_sample_status_counts": shoe_status_counts,
        "outer_envelope_analytical_cross_check": {
            "maximum_absolute_equation_error": float(
                np.max(np.abs(envelope_equation - 1.0))
            )
        },
        "noninjective_overlap_count": overlap_count,
        "stopping_reason": "all_exact_mapping_checks_passed",
    }


def _initialize_mapping_worker(
    volume_root: str,
    batch_root: str,
    containment_root: str,
) -> None:
    """Load immutable batch inputs once in each worker process."""

    global _WORKER_CANONICAL, _WORKER_BATCH_ROOT, _WORKER_CONTAINMENT_ROOT
    _WORKER_CANONICAL = load_canonical_anatomical_volume(Path(volume_root))
    _WORKER_BATCH_ROOT = Path(batch_root)
    _WORKER_CONTAINMENT_ROOT = Path(containment_root)


def _audit_mapping_worker(name: str) -> dict[str, Any]:
    """Audit one shoe without writing shared batch artifacts."""

    if (
        _WORKER_CANONICAL is None
        or _WORKER_BATCH_ROOT is None
        or _WORKER_CONTAINMENT_ROOT is None
    ):
        raise RuntimeError("11-C mapping worker was not initialized")
    mapping = load_instance_volume_map(
        _WORKER_CANONICAL,
        _WORKER_BATCH_ROOT / name,
        _WORKER_CONTAINMENT_ROOT / name,
    )
    return _audit_mapping(
        mapping,
        _WORKER_BATCH_ROOT / name,
        _WORKER_CONTAINMENT_ROOT / name,
    )


def _batch_summary(
    names: list[str], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ordered_results = {name: results[name] for name in names if name in results}
    finished = len(ordered_results) == len(names)
    accepted = finished and all(
        value["status"] == "mapping_valid" for value in ordered_results.values()
    )
    return {
        "schema_version": 2,
        "stage": "instance_volume_mapping_batch",
        "status": (
            "mapping_valid"
            if accepted
            else "failed_11_c"
            if finished
            else "running_or_failed"
        ),
        "total": len(names),
        "completed": len(ordered_results),
        "successful": sum(
            value["status"] == "mapping_valid"
            for value in ordered_results.values()
        ),
        "failed": sum(
            value["status"] == "failed_11_c"
            for value in ordered_results.values()
        ),
        "results": ordered_results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    volume_root = args.anatomical_volume_root.expanduser().resolve(strict=True)
    batch_root = args.instance_volume_batch_root.expanduser().resolve(strict=True)
    containment_root = args.containment_fit_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if not volume_root.is_dir() or not batch_root.is_dir() or not containment_root.is_dir():
        raise NotADirectoryError("all mapping input roots must be directories")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    names, source_manifest = _selected_shoes(
        batch_root, list(args.shoes), list(args.exclude)
    )
    _preflight_paths(names, batch_root, containment_root)

    known_outputs = [output_root / "mapping_manifest.json", output_root / "mapping_summary.json"]
    known_outputs.extend(output_root / name / "mapping_validation.json" for name in names)
    existing = [path for path in known_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"mapping artifacts already exist; pass --overwrite: {existing[0]}"
        )

    canonical = load_canonical_anatomical_volume(volume_root)
    manifest = {
        "schema_version": 2,
        "stage": "instance_volume_mapping_batch",
        "shoes": names,
        "shoe_count": len(names),
        "jobs": int(args.jobs),
        "excluded": sorted(set(args.exclude) | {"sneaker_vibe"}),
        "configuration": mapping_configuration(),
        "inputs": {
            "anatomical_volume_root": str(volume_root),
            "instance_volume_batch_root": str(batch_root),
            "containment_fit_root": str(containment_root),
            "source_batch_manifest_sha256": _file_digest(
                batch_root / "batch_manifest.json"
            ),
            "canonical_volume_topology_sha256": canonical.topology_digest,
            "source_batch_shoe_count": source_manifest.get("shoe_count"),
        },
    }
    _write_json_atomic(output_root / "mapping_manifest.json", manifest)
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    print(
        f"11-C batch: auditing {len(names)} shoes with {args.jobs} worker(s)",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, len(names)),
        initializer=_initialize_mapping_worker,
        initargs=(str(volume_root), str(batch_root), str(containment_root)),
    ) as executor:
        futures = {executor.submit(_audit_mapping_worker, name): name for name in names}
        for offset, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                report = future.result()
                status = "mapping_valid"
                error = None
            except Exception as exception:
                status = "failed_11_c"
                error = f"{type(exception).__name__}: {exception}"
                report = {
                    "schema_version": 2,
                    "stage": "instance_volume_mapping_validation",
                    "status": status,
                    "shoe_name": name,
                    "configuration": mapping_configuration(),
                    "failure": error,
                }
            _write_json_atomic(
                output_root / name / "mapping_validation.json", report
            )
            results[name] = {"status": status, "failure": error}
            print(
                f"[11-C {offset}/{len(names)} completed] {name}: {status}",
                flush=True,
            )
            summary = _batch_summary(names, results)
            _write_json_atomic(output_root / "mapping_summary.json", summary)

    summary = _batch_summary(names, results)
    _write_json_atomic(output_root / "mapping_summary.json", summary)
    print(
        f"11-C batch: {summary['successful']}/{len(names)} valid, "
        f"{summary['failed']} failed ({time.monotonic() - started:.1f}s)",
        flush=True,
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    if summary["status"] != "mapping_valid":
        sys.exit(1)


if __name__ == "__main__":
    main()
