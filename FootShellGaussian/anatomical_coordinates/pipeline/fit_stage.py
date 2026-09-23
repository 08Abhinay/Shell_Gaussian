"""Stage 1: fit every shoe's foot on the GPU and publish containment artifacts.

One process owns one GPU and works through a slice of the shoe list. The model
is tiny, so parallelism is across shoes rather than inside a fit.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from foot_prior.alignment import identify_supr_contact_regions
from foot_prior.supr_foot import load_neutral_supr_foot

from ..config import FitConfig
from ..evaluate import build_evaluator, exact_metrics, selection_metrics
from ..foot_fitter import ShoeFootFitter
from ..losses import millimetres
from ..shoe_data import SUPR_MODEL_PATH, ShoeCase
from ..cavity_field import build_cavity_field
from ..supr_torch import TorchSuprFoot
from .fit_artifacts import write_containment_fit
from .shoe_set import UNIFIED_INPUT_ROOT, load_case


def _contact_regions(regions) -> dict[str, np.ndarray]:
    return {name: regions.vertex_regions[name] for name in ("heel", "forefoot")}


def fit_one(
    fitter: ShoeFootFitter,
    case: ShoeCase,
    contact: dict[str, np.ndarray],
    faces: np.ndarray,
    restarts: int,
    config: FitConfig,
) -> dict[str, Any]:
    """Fit one shoe and return everything the artifact writer needs."""

    started = time.time()
    field = build_cavity_field(
        case.normalized_shoe,
        case.normalized_footbed,
        case.footbed_source_face_indices,
        case.normalized_centerline_xz,
        case.baseline_foot.bounds,
        target_spacing=millimetres(config.field_spacing_mm),
        margin=config.field_margin,
        longitudinal_sign=config.use_longitudinal_sign,
    )
    translation = np.asarray(
        case.support_fit["placement"]["translation"], dtype=np.float64
    )[None]
    ankle = np.asarray([case.baseline_ankle_degrees])
    midfoot = np.asarray([case.baseline_midfoot_degrees])
    betas = np.zeros((1, fitter.model.num_betas))
    if restarts > 1:
        # Restart 0 stays the deterministic zero-beta start, so adding starts
        # can only help; the rest are a fixed seeded spread.
        generator = np.random.default_rng(config.seed)
        betas = np.concatenate(
            [betas]
            + [
                generator.uniform(-1.5, 1.5, size=(1, fitter.model.num_betas))
                for _ in range(restarts - 1)
            ]
        )
        translation = np.tile(translation, (restarts, 1))
        ankle = np.tile(ankle, restarts)
        midfoot = np.tile(midfoot, restarts)

    result = fitter.fit(
        field,
        translation,
        ankle,
        midfoot,
        betas,
        contact,
        case.plantar_vertex_indices,
        case.support_compression_allowance,
    )

    # The winner is chosen by the exact evaluator, never by the training loss.
    evaluator = build_evaluator(case)
    scores = [
        selection_metrics(evaluator, result.vertices[slot], faces)
        for slot in range(result.vertices.shape[0])
    ]
    order = min(
        range(len(scores)),
        key=lambda slot: (
            scores[slot]["collision_area_fraction"]
            + scores[slot]["outside_area_fraction"]
        ),
    )
    metrics = exact_metrics(
        evaluator,
        case,
        result.vertices[order],
        faces,
        betas=result.betas[order],
        ankle_degrees=float(result.ankle_degrees[order]),
        midfoot_degrees=float(result.midfoot_degrees[order]),
    )
    return {
        "vertices": result.vertices[order],
        "betas": result.betas[order],
        "pose": result.pose[order],
        "translation": result.translation[order],
        "scale": result.scale,
        "ankle_degrees": float(result.ankle_degrees[order]),
        "midfoot_degrees": float(result.midfoot_degrees[order]),
        "metrics": metrics,
        "restart_scores": scores,
        "selected_restart": int(order),
        "seconds": time.time() - started,
    }


def run_fit_stage(
    names: list[str],
    output_root: Path,
    config: FitConfig | None = None,
    restarts: int = 8,
    input_root: Path = UNIFIED_INPUT_ROOT,
    device: str = "cuda",
    containment_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit every named shoe and write ``containment_fit/<shoe>/``."""

    config = config or FitConfig()
    torch_device = torch.device(device)
    neutral = load_neutral_supr_foot(SUPR_MODEL_PATH)
    model = TorchSuprFoot(SUPR_MODEL_PATH, num_betas=10)
    fitter = ShoeFootFitter(model, neutral, config)
    contact = _contact_regions(identify_supr_contact_regions(neutral))
    faces = np.asarray(model.faces.cpu().numpy(), dtype=np.int64)

    containment_root = (
        Path(containment_dir) if containment_dir is not None
        else Path(output_root) / "containment_fit"
    )
    containment_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"shoes": {}}
    for name in names:
        try:
            case = load_case(name, input_root)
            fit = fit_one(fitter, case, contact, faces, restarts, config)
            write_containment_fit(
                containment_root / name,
                case,
                fit["vertices"],
                faces,
                fit["betas"],
                fit["pose"],
                fit["scale"],
                fit["translation"],
                fit["ankle_degrees"],
                fit["midfoot_degrees"],
                fit["metrics"],
                fit["seconds"],
                config.to_dict(),
            )
            after = fit["metrics"]
            report["shoes"][name] = {
                "status": after["status"],
                "collision_area_fraction": after["collision_area_fraction"],
                "outside_area_fraction": after["outside_area_fraction"],
                "toe_allowance_mm": after["toe_allowance_mm"],
                "heel_x_mm": after.get("heel_x_mm"),
                "plantar_minimum_gap_mm": after["plantar_minimum_gap_mm"],
                "beta_l2_norm": after["beta_l2_norm"],
                "seconds": fit["seconds"],
                "selected_restart": fit["selected_restart"],
            }
            print(
                f"[fit] {name:46s} {after['status']:20s} "
                f"coll {after['collision_area_fraction']:.4f} "
                f"out {after['outside_area_fraction']:.4f} "
                f"toe {after['toe_allowance_mm']:5.1f}mm  {fit['seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - report, never silently skip
            report["shoes"][name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"[fit] {name:46s} FAILED: {error}", flush=True)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--containment-dir", type=Path, default=None,
        help="where the fitted feet go; defaults to <output-root>/containment_fit",
    )
    parser.add_argument("--input-root", type=Path, default=UNIFIED_INPUT_ROOT)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--max-beta-norm", type=float, default=None,
        help="anatomy budget; smaller keeps the fit closer to canonical",
    )
    args = parser.parse_args()

    from .shoe_set import all_shoes

    names = args.shoes or all_shoes(args.input_root)
    config = FitConfig() if args.max_beta_norm is None else FitConfig(
        max_beta_norm=args.max_beta_norm
    )
    report = run_fit_stage(
        names, args.output_root, config=config,
        restarts=args.restarts, input_root=args.input_root,
        containment_dir=args.containment_dir,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
