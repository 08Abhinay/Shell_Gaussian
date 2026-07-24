"""Command-line entry point for prediction-to-ground-truth mesh alignment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .alignment import AlignmentConfig, align_meshes
from .mesh_io import load_mesh, mesh_summary, transform_mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align a reconstructed mesh to a ground-truth mesh with robust similarity ICP."
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--coarse-samples", type=int, default=5_000)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--inlier-fraction", type=float, default=0.8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-aligned", action="store_true")
    return parser


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    prediction_path = args.prediction.expanduser().resolve()
    ground_truth_path = args.ground_truth.expanduser().resolve()
    output = args.output.expanduser().resolve()

    config = AlignmentConfig(
        sample_count=args.samples,
        coarse_sample_count=args.coarse_samples,
        candidate_count=args.candidates,
        inlier_fraction=args.inlier_fraction,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    prediction = load_mesh(prediction_path)
    ground_truth = load_mesh(ground_truth_path)
    result = align_meshes(prediction, ground_truth, config)

    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "prediction": str(prediction_path),
        "ground_truth": str(ground_truth_path),
        "configuration": {
            "sample_count": config.sample_count,
            "coarse_sample_count": config.coarse_sample_count,
            "candidate_count": config.candidate_count,
            "inlier_fraction": config.inlier_fraction,
            "max_iterations": config.max_iterations,
            "tolerance": config.tolerance,
            "seed": config.seed,
            "allow_reflection": False,
            "allow_nonuniform_scale": False,
        },
        "prediction_mesh": mesh_summary(prediction),
        "ground_truth_mesh": mesh_summary(ground_truth),
        "alignment": result.to_dict(),
    }
    _atomic_json(output / "alignment.json", payload)

    if args.save_aligned:
        aligned = transform_mesh(prediction, result.transform)
        aligned.export(output / "aligned_prediction.ply")

    print(f"Alignment written to {output / 'alignment.json'}")
    print(f"Initialization: {result.initialization}")
    print(f"Uniform scale: {result.scale:.8f}")
    print(f"Normalized error: {result.before_error:.8f} -> {result.after_error:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
