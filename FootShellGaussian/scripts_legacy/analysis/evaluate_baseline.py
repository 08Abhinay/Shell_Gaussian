#!/usr/bin/env python3
"""Experiment A: judge the stored NumPy-fitter output with the same evaluator.

Read-only over ``golden_set_evaluation``; writes one JSON into the runs folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anatomical_coordinates.evaluate import build_evaluator, exact_metrics
from anatomical_coordinates.shoe_data import GOLDEN_SET_ROOT, available_shoes, load_shoe_case
from foot_prior.mesh import load_triangle_mesh

OUTPUT = Path("/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs")


def main() -> None:
    payload: dict[str, dict] = {}
    for name in available_shoes():
        case = load_shoe_case(name)
        evaluator = build_evaluator(case)
        faces = case.baseline_foot.faces
        checkpoint5 = exact_metrics(
            evaluator, case, case.baseline_foot.vertices, faces,
            np.zeros(10), case.baseline_ankle_degrees,
            case.baseline_midfoot_degrees,
        )
        fitted = GOLDEN_SET_ROOT / "containment_fit" / name
        mesh = load_triangle_mesh(fitted / "foot_containment_fitted.ply")
        record = json.loads((fitted / "containment_fit.json").read_text())
        angles = record["supr"]["selected_angles_degrees"]
        numpy_fit = exact_metrics(
            evaluator, case, mesh.vertices, mesh.faces,
            np.asarray(record["supr"]["betas"]),
            angles["ankle_pitch"], angles["midfoot_pitch"],
        )
        payload[name] = {"checkpoint5": checkpoint5, "numpy_fitter": numpy_fit}
        print(
            f"[{name:<38}] C5 coll {checkpoint5['collision_area_fraction']:.4f} "
            f"-> A coll {numpy_fit['collision_area_fraction']:.4f}",
            flush=True,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "baseline_numpy.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {OUTPUT / 'baseline_numpy.json'}")


if __name__ == "__main__":
    main()
