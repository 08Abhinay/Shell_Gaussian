"""Deterministic robust similarity alignment for reconstructed meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations, product

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh_io import sample_surface


@dataclass(frozen=True)
class AlignmentConfig:
    """Controls coarse initialization and robust similarity-ICP refinement."""

    sample_count: int = 50_000
    coarse_sample_count: int = 5_000
    candidate_count: int = 4
    inlier_fraction: float = 0.8
    max_iterations: int = 100
    tolerance: float = 1e-7
    seed: int = 0

    def validate(self) -> None:
        if self.sample_count < 100:
            raise ValueError("sample_count must be at least 100")
        if not 100 <= self.coarse_sample_count <= self.sample_count:
            raise ValueError("coarse_sample_count must be between 100 and sample_count")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.5 <= self.inlier_fraction <= 1.0:
            raise ValueError("inlier_fraction must be in [0.5, 1.0]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class AlignmentResult:
    """Similarity transform and diagnostics for prediction-to-GT alignment."""

    transform: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    initialization: str
    converged: bool
    iterations: int
    before_error: float
    after_error: float
    inlier_fraction: float
    rotation_determinant: float
    candidate_errors: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["transform"] = self.transform.tolist()
        payload["rotation"] = self.rotation.tolist()
        payload["translation"] = self.translation.tolist()
        return payload


@dataclass(frozen=True)
class _Candidate:
    name: str
    transform: np.ndarray
    coarse_error: float


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to row-vector 3D points."""
    xyz = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def compose_similarity(
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Build a homogeneous matrix representing x' = scale * R * x + t."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def decompose_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Recover positive uniform scale, proper rotation, and translation."""
    matrix = np.asarray(transform, dtype=np.float64)
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Similarity transform has invalid scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Similarity transform contains nonuniform scale or shear")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-5):
        raise ValueError("Similarity transform contains a reflection")
    return scale, rotation, matrix[:3, 3].copy()


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Estimate the least-squares proper similarity transform using Umeyama."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity correspondences must have matching Nx3 shapes")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("Source correspondences are degenerate")

    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Estimated similarity has invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return compose_similarity(scale, rotation, translation)


def _trim_indices(distances: np.ndarray, fraction: float) -> np.ndarray:
    count = max(3, int(np.ceil(len(distances) * fraction)))
    if count >= len(distances):
        return np.arange(len(distances))
    return np.argpartition(distances, count - 1)[:count]


def symmetric_trimmed_error(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    inlier_fraction: float,
) -> float:
    """Return mean bidirectional nearest-neighbor distance after trimming."""
    transformed = apply_transform(source, transform)
    source_to_target = cKDTree(target).query(transformed, workers=1)[0]
    target_to_source = cKDTree(transformed).query(target, workers=1)[0]
    source_keep = _trim_indices(source_to_target, inlier_fraction)
    target_keep = _trim_indices(target_to_source, inlier_fraction)
    return 0.5 * (
        float(np.mean(source_to_target[source_keep]))
        + float(np.mean(target_to_source[target_keep]))
    )


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def _proper_axis_mappings() -> list[np.ndarray]:
    mappings: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            mapping = base @ np.diag(signs)
            if np.linalg.det(mapping) > 0.5:
                mappings.append(mapping)
    return mappings


def _robust_radius(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    radius = float(np.quantile(np.linalg.norm(centered, axis=1), 0.75))
    if not np.isfinite(radius) or radius <= np.finfo(np.float64).eps:
        raise ValueError("Point cloud has a degenerate robust radius")
    return radius


def _initial_candidates(source: np.ndarray, target: np.ndarray) -> list[tuple[str, np.ndarray]]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    initial_scale = _robust_radius(target) / _robust_radius(source)
    source_axes = _principal_axes(source)
    target_axes = _principal_axes(target)

    rotations: list[tuple[str, np.ndarray]] = [("existing_pose", np.eye(3, dtype=np.float64))]
    for index, mapping in enumerate(_proper_axis_mappings()):
        rotation = target_axes @ mapping @ source_axes.T
        rotations.append((f"pca_{index:02d}", rotation))

    candidates: list[tuple[str, np.ndarray]] = []
    seen: list[np.ndarray] = []
    for name, rotation in rotations:
        if any(np.allclose(rotation, previous, atol=1e-8) for previous in seen):
            continue
        translation = target_mean - initial_scale * (rotation @ source_mean)
        candidates.append((name, compose_similarity(initial_scale, rotation, translation)))
        seen.append(rotation)
    return candidates


def _correspondences(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    inlier_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = apply_transform(source, transform)
    target_tree = cKDTree(target)
    source_distances, source_matches = target_tree.query(transformed, workers=1)
    source_keep = _trim_indices(source_distances, inlier_fraction)

    transformed_tree = cKDTree(transformed)
    target_distances, target_matches = transformed_tree.query(target, workers=1)
    target_keep = _trim_indices(target_distances, inlier_fraction)

    source_pairs = np.concatenate(
        (source[source_keep], source[target_matches[target_keep]]),
        axis=0,
    )
    target_pairs = np.concatenate(
        (target[source_matches[source_keep]], target[target_keep]),
        axis=0,
    )
    return source_pairs, target_pairs


def _refine_similarity(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    config: AlignmentConfig,
) -> tuple[np.ndarray, float, int, bool]:
    current = initial.copy()
    best = current.copy()
    best_error = symmetric_trimmed_error(
        source, target, current, config.inlier_fraction
    )
    previous_error = best_error
    converged = False
    error_scale = max(
        float(np.linalg.norm(np.ptp(target, axis=0))),
        np.finfo(np.float64).eps,
    )

    for iteration in range(1, config.max_iterations + 1):
        source_pairs, target_pairs = _correspondences(
            source, target, current, config.inlier_fraction
        )
        candidate = estimate_similarity(source_pairs, target_pairs)
        candidate_error = symmetric_trimmed_error(
            source, target, candidate, config.inlier_fraction
        )
        if candidate_error < best_error:
            best = candidate.copy()
            best_error = candidate_error

        relative_change = abs(previous_error - candidate_error) / max(
            previous_error, np.finfo(np.float64).eps
        )
        current = candidate
        if (
            relative_change < config.tolerance
            or candidate_error / error_scale < config.tolerance
        ):
            converged = True
            return best, best_error, iteration, converged
        previous_error = candidate_error

    return best, best_error, config.max_iterations, converged


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Align prediction samples to ground-truth samples with robust similarity ICP."""
    settings = config or AlignmentConfig()
    settings.validate()
    source = np.ascontiguousarray(source, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape[1:] != (3,)
    ):
        raise ValueError("Alignment inputs must be Nx3 point arrays")
    if len(source) < settings.coarse_sample_count or len(target) < settings.coarse_sample_count:
        raise ValueError("Alignment inputs contain fewer points than coarse_sample_count")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Alignment inputs must be finite")

    source_coarse = source[: settings.coarse_sample_count]
    target_coarse = target[: settings.coarse_sample_count]
    target_diagonal = float(np.linalg.norm(np.ptp(target, axis=0)))
    if target_diagonal <= np.finfo(np.float64).eps:
        raise ValueError("Ground-truth samples are degenerate")

    before_error = symmetric_trimmed_error(
        source, target, np.eye(4), settings.inlier_fraction
    ) / target_diagonal

    ranked: list[_Candidate] = []
    for name, transform in _initial_candidates(source_coarse, target_coarse):
        error = symmetric_trimmed_error(
            source_coarse,
            target_coarse,
            transform,
            settings.inlier_fraction,
        )
        ranked.append(_Candidate(name=name, transform=transform, coarse_error=error))
    ranked.sort(key=lambda candidate: candidate.coarse_error)

    candidate_errors: dict[str, float] = {}
    refined: list[tuple[str, np.ndarray, float, int, bool]] = []
    for candidate in ranked[: settings.candidate_count]:
        transform, error, iterations, converged = _refine_similarity(
            source_coarse,
            target_coarse,
            candidate.transform,
            settings,
        )
        candidate_errors[candidate.name] = float(error / target_diagonal)
        refined.append((candidate.name, transform, error, iterations, converged))
    refined.sort(key=lambda item: item[2])

    initialization, coarse_transform, _, coarse_iterations, _ = refined[0]
    final_transform, final_error, fine_iterations, fine_converged = _refine_similarity(
        source,
        target,
        coarse_transform,
        settings,
    )
    scale, rotation, translation = decompose_similarity(final_transform)
    after_error = final_error / target_diagonal

    return AlignmentResult(
        transform=final_transform,
        rotation=rotation,
        translation=translation,
        scale=scale,
        initialization=initialization,
        converged=bool(fine_converged),
        iterations=int(coarse_iterations + fine_iterations),
        before_error=float(before_error),
        after_error=float(after_error),
        inlier_fraction=settings.inlier_fraction,
        rotation_determinant=float(np.linalg.det(rotation)),
        candidate_errors=candidate_errors,
    )


def align_meshes(
    prediction: trimesh.Trimesh,
    ground_truth: trimesh.Trimesh,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Sample and align a predicted triangle mesh to its ground truth."""
    settings = config or AlignmentConfig()
    settings.validate()
    prediction_points, _ = sample_surface(prediction, settings.sample_count, settings.seed)
    ground_truth_points, _ = sample_surface(
        ground_truth, settings.sample_count, settings.seed + 1
    )
    return align_point_clouds(prediction_points, ground_truth_points, settings)
