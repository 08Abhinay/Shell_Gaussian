#!/usr/bin/env python3
"""Time the existing NumPy search fitter against the differentiable one.

Both are timed on the same shoes from the same Checkpoint 5 starting point.
Neither writes artifacts; this measures fitting only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from anatomical_coordinates.config import FitConfig
from anatomical_coordinates.fitter import ShoeFootFitter
from anatomical_coordinates.losses import millimetres
from anatomical_coordinates.shoe_data import SUPR_MODEL_PATH, load_shoe_case
from anatomical_coordinates.shoe_field import build_cavity_field
from anatomical_coordinates.supr_torch import TorchSuprFoot
from foot_prior.alignment import identify_supr_contact_regions
from foot_prior.containment import build_containment_foot_fit
from foot_prior.supr_foot import load_neutral_supr_foot, load_posable_supr_foot

OUTPUT = Path("/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoes", nargs="*", default=["crocs", "pb129_shoe_low"])
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--skip-numpy", action="store_true")
    args = parser.parse_args()

    neutral = load_neutral_supr_foot(SUPR_MODEL_PATH)
    torch_model = TorchSuprFoot(SUPR_MODEL_PATH, num_betas=10)
    config = FitConfig()
    fitter = ShoeFootFitter(torch_model, neutral, config)
    regions = identify_supr_contact_regions(neutral)
    contact = {name: regions.vertex_regions[name] for name in ("heel", "forefoot")}
    records = []

    for name in args.shoes:
        case = load_shoe_case(name)
        started = time.time()
        field = build_cavity_field(
            case.normalized_shoe, case.normalized_footbed,
            case.footbed_source_face_indices, case.normalized_centerline_xz,
            case.baseline_foot.bounds,
            target_spacing=millimetres(config.field_spacing_mm),
            margin=config.field_margin,
        )
        torch.cuda.synchronize()
        bake = time.time() - started

        restarts = max(1, args.restarts)
        generator = np.random.default_rng(config.seed)
        betas = np.concatenate(
            [np.zeros((1, 10))]
            + [generator.uniform(-1.5, 1.5, size=(1, 10)) for _ in range(restarts - 1)]
        )
        translation = np.tile(
            np.asarray(case.support_fit["placement"]["translation"])[None], (restarts, 1)
        )
        ankle = np.repeat(case.baseline_ankle_degrees, restarts)
        midfoot = np.repeat(case.baseline_midfoot_degrees, restarts)

        # warm-up so the timing excludes CUDA context and kernel autotuning
        fitter.fit(
            field, translation[:1], ankle[:1], midfoot[:1], betas[:1],
            contact, regions.plantar_vertex_indices,
            case.support_compression_allowance,
            stages=config.stages[:1],
        )
        torch.cuda.synchronize()
        started = time.time()
        result = fitter.fit(
            field, translation, ankle, midfoot, betas, contact,
            regions.plantar_vertex_indices, case.support_compression_allowance,
        )
        torch.cuda.synchronize()
        torch_seconds = time.time() - started

        numpy_seconds = None
        if not args.skip_numpy:
            posable = load_posable_supr_foot(SUPR_MODEL_PATH, num_betas=10)
            pose = np.zeros(39)
            pose[3] = np.deg2rad(case.baseline_ankle_degrees)
            pose[6] = np.deg2rad(case.baseline_midfoot_degrees)
            started = time.time()
            try:
                build_containment_foot_fit(
                    supr_model=posable,
                    neutral_foot_mesh=neutral,
                    normalized_shoe_mesh=case.normalized_shoe,
                    normalized_support_mesh=case.normalized_footbed,
                    normalized_centerline_xz=case.normalized_centerline_xz,
                    footbed_source_face_indices=case.footbed_source_face_indices,
                    shoe_to_normalized=case.shoe_to_normalized,
                    normalized_to_shoe=case.normalized_to_shoe,
                    support_grid_cell_spacing=case.support_grid_cell_spacing,
                    initial_pose_parameters=pose,
                    initial_betas=np.zeros(10),
                    baseline_fitted_foot=case.baseline_foot,
                    support_compression_allowance=case.support_compression_allowance,
                )
                numpy_seconds = time.time() - started
            except Exception as error:  # noqa: BLE001
                numpy_seconds = f"failed after {time.time() - started:.1f}s: {error}"

        record = {
            "shoe": name,
            "field_bake_seconds": bake,
            "torch_fit_seconds": torch_seconds,
            "torch_restarts": restarts,
            "torch_seconds_per_restart": torch_seconds / restarts,
            "torch_peak_memory_mb": result.peak_memory_bytes / 1e6,
            "numpy_fit_seconds": numpy_seconds,
        }
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "runtime_benchmark.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(f"wrote {OUTPUT / 'runtime_benchmark.json'}")


if __name__ == "__main__":
    main()
