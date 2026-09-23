#!/usr/bin/env python3
"""Fit SUPR into prepared golden-set shoes with the differentiable fitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from anatomical_coordinates.config import FitConfig, StageConfig
from anatomical_coordinates.evaluate import (
    build_evaluator,
    exact_metrics,
    selection_metrics,
)
from anatomical_coordinates.fitter import ShoeFootFitter
from anatomical_coordinates.losses import millimetres
from anatomical_coordinates.shoe_data import (
    GOLDEN_SET_ROOT,
    SUPR_MODEL_PATH,
    available_shoes,
    load_shoe_case,
)
from anatomical_coordinates.shoe_field import build_cavity_field
from anatomical_coordinates.supr_torch import TorchSuprFoot
from foot_prior.alignment import identify_supr_contact_regions
from foot_prior.mesh import save_triangle_mesh, TriangleMesh
from foot_prior.supr_foot import load_neutral_supr_foot


DEFAULT_OUTPUT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs"
)
# Checkpoint 5 is the same starting point for every run, and the exact
# evaluator is expensive, so its metrics are read from the baseline artifact
# when one exists instead of being recomputed per experiment.
BASELINE_CACHE = DEFAULT_OUTPUT / "baseline_numpy.json"

# Named variable subsets for the experiment matrix. "translation" alone is
# experiment B, adding "pose" is C, adding "betas" is D.
VARIANTS: dict[str, tuple[str, ...]] = {
    "translation": ("translation",),
    "pose": ("translation", "pose"),
    "full": ("translation", "pose", "betas"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument(
        "--golden-root", type=Path, default=GOLDEN_SET_ROOT,
        help="root holding shoe_preparation/ and support_fit/ artifacts",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="torch_fit")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="full",
        help="which parameter groups the optimizer may move",
    )
    parser.add_argument("--w-support", type=float, default=None)
    parser.add_argument("--w-beta", type=float, default=None)
    parser.add_argument("--w-length", type=float, default=None)
    parser.add_argument("--support-safety-mm", type=float, default=None)
    parser.add_argument("--w-heel", type=float, default=None)
    parser.add_argument("--length-scale-mm", type=float, default=None)
    parser.add_argument("--field-spacing-mm", type=float, default=1.0)
    parser.add_argument("--w-containment", type=float, default=None)
    parser.add_argument("--containment-margin-mm", type=float, default=None)
    parser.add_argument(
        "--no-ankle-exemption", action="store_true",
        help="score the ankle exit region as outside (ablation)",
    )
    parser.add_argument(
        "--subdivision-levels", type=int, choices=(0, 1, 2), default=1,
        help="containment sampling density; the SUPR forward stays at 266 verts",
    )
    parser.add_argument(
        "--restarts", type=int, default=8,
        help=(
            "fit this many beta initializations simultaneously as one batch and "
            "keep the best by the exact evaluator; the model is tiny, so extra "
            "restarts are nearly free"
        ),
    )
    parser.add_argument(
        "--longitudinal-sign",
        action="store_true",
        help="enable the +/-X ray family that signs the distance channel",
    )
    parser.add_argument(
        "--no-longitudinal-sign",
        action="store_true",
        help="ablate the +/-X ray family (the default is already off)",
    )
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument(
        "--batched", action="store_true",
        help="fit every requested shoe in one batch instead of sequentially",
    )
    return parser.parse_args()


def _stage_schedule(variant: str, config: FitConfig) -> tuple[StageConfig, ...]:
    """Drop stages whose variables this variant is not allowed to move."""

    allowed = set(VARIANTS[variant])
    stages: list[StageConfig] = []
    for stage in config.stages:
        active = tuple(name for name in stage.active if name in allowed)
        if not active:
            continue
        stages.append(
            StageConfig(
                stage.name,
                stage.steps,
                {name: stage.learning_rates[name] for name in active},
                active,
            )
        )
    return tuple(stages)


def _contact_regions(regions) -> dict[str, np.ndarray]:
    """The load-bearing plantar regions, kept separate.

    Heel and forefoot each need their own contact term; the arch is excluded
    because a real arch does not touch, and the toes only lightly.
    """

    return {
        name: regions.vertex_regions[name] for name in ("heel", "forefoot")
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    names = args.shoes or available_shoes(args.golden_root)
    overrides: dict[str, Any] = {
        "field_spacing_mm": args.field_spacing_mm,
        "use_ankle_exemption": not args.no_ankle_exemption,
    }
    for name, value in (
        ("w_support", args.w_support),
        ("w_beta", args.w_beta),
        ("w_length", args.w_length),
        ("w_containment", args.w_containment),
        ("containment_margin_mm", args.containment_margin_mm),
        ("containment_subdivision_levels", args.subdivision_levels),
        ("support_safety_mm", args.support_safety_mm),
        ("w_heel", args.w_heel),
        ("length_scale_mm", args.length_scale_mm),
    ):
        if value is not None:
            overrides[name] = value
    # The longitudinal sign defaults to off in FitConfig, so only an explicit
    # flag may override it in either direction.
    if args.longitudinal_sign and args.no_longitudinal_sign:
        raise SystemExit("pass at most one of --longitudinal-sign/--no-...")
    if args.longitudinal_sign:
        overrides["use_longitudinal_sign"] = True
    elif args.no_longitudinal_sign:
        overrides["use_longitudinal_sign"] = False
    config = FitConfig(**overrides)

    neutral = load_neutral_supr_foot(SUPR_MODEL_PATH)
    model = TorchSuprFoot(SUPR_MODEL_PATH, num_betas=10)
    fitter = ShoeFootFitter(model, neutral, config)
    regions = identify_supr_contact_regions(neutral)
    contact = _contact_regions(regions)
    faces = np.asarray(model.faces.cpu().numpy(), dtype=np.int64)
    stages = _stage_schedule(args.variant, config)

    cached_before: dict[str, Any] = {}
    if BASELINE_CACHE.is_file():
        cached_before = {
            name: record["checkpoint5"]
            for name, record in json.loads(BASELINE_CACHE.read_text()).items()
        }

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    cases = []
    fields = []
    for name in names:
        try:
            case = load_shoe_case(name, args.golden_root)
        except Exception as error:  # noqa: BLE001 - report, do not silently skip
            results.append({"shoe": name, "error": f"load failed: {error}"})
            print(f"[{name}] LOAD FAILED: {error}", flush=True)
            continue
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
        torch.cuda.synchronize()
        bake_seconds = time.time() - started
        cases.append((case, bake_seconds))
        fields.append(field)

    groups: list[list[int]] = (
        [list(range(len(cases)))] if args.batched
        else [[index] for index in range(len(cases))]
    )

    for group in groups:
        if not group:
            continue
        if len(group) > 1:
            field = _stack_fields([fields[index] for index in group])
        else:
            field = fields[group[0]]
        members = [cases[index][0] for index in group]
        translation = np.stack(
            [
                np.asarray(item.support_fit["placement"]["translation"], dtype=np.float64)
                for item in members
            ]
        )
        ankle = np.asarray([item.baseline_ankle_degrees for item in members])
        midfoot = np.asarray([item.baseline_midfoot_degrees for item in members])
        betas = np.zeros((len(members), model.num_betas))
        # Restarts perturb the initial betas, so they are only meaningful when
        # betas are actually being optimized. For the translation-only and
        # pose-only variants every restart would otherwise start from a random
        # shape it can never correct, which silently changes what those
        # experiments measure.
        restarts = max(1, int(args.restarts))
        if "betas" not in VARIANTS[args.variant]:
            restarts = 1
        if args.batched and restarts > 1:
            # Restarts and batching both consume the batch axis, and one field
            # copy per (shoe, restart) would be ~8.7 GB for the full set. The
            # batched run exists to test whether batching changes the answer,
            # which is isolated with a single start on both sides.
            print(
                "note: --batched forces --restarts 1; compare against a "
                "sequential run with --restarts 1",
                flush=True,
            )
            restarts = 1
        if restarts > 1:
            # Restart 0 is always the deterministic zero-beta start, so adding
            # restarts can only help. The rest are a fixed, seeded spread.
            generator = np.random.default_rng(config.seed)
            betas = np.concatenate(
                [betas]
                + [
                    generator.uniform(-1.5, 1.5, size=(len(members), model.num_betas))
                    for _ in range(restarts - 1)
                ]
            )
            translation = np.tile(translation, (restarts, 1))
            ankle = np.tile(ankle, restarts)
            midfoot = np.tile(midfoot, restarts)
        allowance = float(np.mean([item.support_compression_allowance for item in members]))

        result = fitter.fit(
            field,
            translation,
            ankle,
            midfoot,
            betas,
            contact,
            regions.plantar_vertex_indices,
            allowance,
            stages=stages,
            lbfgs_steps=args.lbfgs_steps,
        )

        for slot, index in enumerate(group):
            case, bake_seconds = cases[index]
            evaluator = build_evaluator(case)
            if restarts > 1:
                # Pick the restart the *exact* judge prefers, never the one with
                # the smallest differentiable loss.
                options = [
                    (slot + attempt * len(group)) for attempt in range(restarts)
                ]
                scored = []
                for option in options:
                    metrics = selection_metrics(
                        evaluator, result.vertices[option], faces
                    )
                    scored.append(
                        (
                            metrics["collision_area_fraction"]
                            + metrics["outside_area_fraction"],
                            option,
                        )
                    )
                scored.sort(key=lambda item: item[0])
                record_restarts = [
                    {"option": int(o), "score": float(v)} for v, o in scored
                ]
                slot = scored[0][1]
            before = cached_before.get(case.name)
            if before is None:
                before = exact_metrics(
                    evaluator, case, case.baseline_foot.vertices, faces,
                    np.zeros(model.num_betas), case.baseline_ankle_degrees,
                    case.baseline_midfoot_degrees,
                )
            after = exact_metrics(
                evaluator, case, result.vertices[slot], faces,
                result.betas[slot], float(result.ankle_degrees[slot]),
                float(result.midfoot_degrees[slot]),
            )
            record = {
                "shoe": case.name,
                "variant": args.variant,
                "batched": bool(args.batched),
                "batch_size": len(group),
                "field": {
                    "shape": list(fields[index].shape),
                    "bake_seconds": bake_seconds,
                    "valid_fraction": fields[index].metadata["valid_fraction"],
                    "spacing_mm": config.field_spacing_mm,
                },
                "fit_seconds": result.seconds,
                "peak_memory_bytes": result.peak_memory_bytes,
                "final_losses": result.final_losses,
                "parameters": {
                    "betas": result.betas[slot].tolist(),
                    "ankle_pitch_degrees": float(result.ankle_degrees[slot]),
                    "midfoot_pitch_degrees": float(result.midfoot_degrees[slot]),
                    "translation": result.translation[slot].tolist(),
                    "scale": result.scale,
                },
                "before": before,
                "after": after,
                **(
                    {"restart_scores": record_restarts}
                    if restarts > 1 else {}
                ),
            }
            results.append(record)
            print(
                f"[{case.name:<38}] {before['status']:<22} -> {after['status']:<22} "
                f"coll {before['collision_area_fraction']:.4f}->{after['collision_area_fraction']:.4f}  "
                f"out {before['outside_area_fraction']:.4f}->{after['outside_area_fraction']:.4f}  "
                f"toe {after['toe_allowance_mm']:.1f}mm  |b| {after['beta_l2_norm']:.2f}  "
                f"{result.seconds:.1f}s",
                flush=True,
            )
            if args.save_meshes:
                save_triangle_mesh(
                    run_dir / f"{case.name}_fitted.ply",
                    TriangleMesh(result.vertices[slot], faces),
                )
        if args.batched or len(group) == 1:
            history_name = (
                "history_batched.json" if args.batched
                else f"history_{members[0].name}.json"
            )
            (run_dir / history_name).write_text(
                json.dumps(result.history, indent=2) + "\n"
            )

    payload = {
        "run_name": args.run_name,
        "variant": args.variant,
        "batched": bool(args.batched),
        "config": config.to_dict(),
        "stages": [
            {"name": s.name, "steps": s.steps, "active": list(s.active),
             "learning_rates": s.learning_rates}
            for s in stages
        ],
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "results": results,
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nwrote {run_dir / 'results.json'}")


def _extend_axis(axis: "torch.Tensor", count: int) -> "torch.Tensor":
    """Continue a uniform 1-D lattice to ``count`` samples at the same spacing."""

    if axis.numel() >= count:
        return axis
    step = axis[1] - axis[0] if axis.numel() > 1 else axis.new_ones(())
    extra = axis[-1] + step * torch.arange(
        1, count - axis.numel() + 1, device=axis.device, dtype=axis.dtype
    )
    return torch.cat((axis, extra))


def _stack_fields(fields: list) -> Any:
    """Pad several shoe fields onto a common lattice for batched fitting."""

    from anatomical_coordinates.shoe_field import ShoeCavityField

    shapes = np.asarray([field.shape for field in fields])
    target = shapes.max(axis=0)
    clearance = []
    valid = []
    distance = []
    for field in fields:
        pad = [
            0, int(target[2] - field.shape[2]),
            0, int(target[1] - field.shape[1]),
            0, int(target[0] - field.shape[0]),
        ]
        clearance.append(
            torch.nn.functional.pad(field.clearance, pad, mode="replicate")
        )
        valid.append(torch.nn.functional.pad(field.valid, pad, mode="replicate"))
        distance.append(
            torch.nn.functional.pad(field.distance, pad, mode="replicate")
        )
    # Padding adds cells at the high end of each axis, so every padded shoe's
    # domain box has to grow with it: grid_sample maps lower..upper onto the
    # full extent of the *padded* tensor, and leaving `upper` alone would
    # silently rescale that shoe's world coordinates.
    uppers = [
        field.lower + field.spacing * (
            torch.as_tensor(target, device=field.lower.device, dtype=field.lower.dtype) - 1.0
        )
        for field in fields
    ]
    return ShoeCavityField(
        clearance=torch.cat(clearance, dim=0),
        valid=torch.cat(valid, dim=0),
        distance=torch.cat(distance, dim=0),
        lower=torch.stack([field.lower for field in fields]),
        upper=torch.stack(uppers),
        spacing=torch.stack([field.spacing for field in fields]),
        footbed_x=torch.stack(
            [_extend_axis(field.footbed_x, int(target[0])) for field in fields]
        ),
        footbed_z=torch.stack(
            [_extend_axis(field.footbed_z, int(target[2])) for field in fields]
        ),
        footbed_y=torch.stack(
            [
                torch.nn.functional.pad(
                    field.footbed_y[None, None],
                    (0, int(target[2] - field.footbed_y.shape[1]),
                     0, int(target[0] - field.footbed_y.shape[0])),
                    mode="replicate",
                )[0, 0]
                for field in fields
            ]
        ),
        open_above=torch.stack(
            [
                torch.nn.functional.pad(
                    field.open_above[None, None],
                    (0, int(target[2] - field.open_above.shape[1]),
                     0, int(target[0] - field.open_above.shape[0])),
                    mode="constant",
                    value=0.0,
                )[0, 0]
                for field in fields
            ]
        ),
        metadata={"batched": True, "members": len(fields)},
    )


if __name__ == "__main__":
    main()
