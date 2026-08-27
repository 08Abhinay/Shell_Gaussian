#!/usr/bin/env python3
"""Build SUPR-derived pseudo-lasts from foot-aware alignment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOOTSHELL_ROOT = Path(__file__).resolve().parents[1]
for path in [PROJECT_ROOT, FOOTSHELL_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from FootShellGaussian.foot_prior import (  # noqa: E402
    FootSDFBuildConfig,
    PseudoLastConfig,
    build_and_save_signed_sdf_from_obj,
    build_pseudo_last,
    load_triangle_mesh,
    save_pseudo_last_artifacts,
)


DEFAULT_ALIGNMENT_ROOT = FOOTSHELL_ROOT / "output/foot_aware_alignment"
DEFAULT_OUTPUT_ROOT = FOOTSHELL_ROOT / "output/pseudo_last"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shoe-name", action="append", help="Shoe folder name. Repeatable.")
    parser.add_argument("--max-shoes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-normal", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="Device for optional SDF generation.")
    parser.add_argument("--no-sdf", action="store_true", help="Skip pseudo_last_sdf.npz generation.")
    parser.add_argument("--sdf-resolution", type=int, default=128)
    parser.add_argument("--sdf-padding", type=float, default=0.03)
    parser.add_argument("--builder-mode", choices=["section_loft", "surface_offset"], default="section_loft")
    parser.add_argument("--n-x", type=int, default=128)
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument("--eta-s", type=float, default=0.93)
    parser.add_argument("--toe-allowance-ratio", type=float, default=0.05)
    parser.add_argument("--arch-strength", type=float, default=0.75)
    parser.add_argument("--heel-hold-strength", type=float, default=0.05)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> PseudoLastConfig:
    return PseudoLastConfig(
        builder_mode=args.builder_mode,
        n_x=args.n_x,
        n_theta=args.n_theta,
        eta_s=args.eta_s,
        toe_allowance_ratio=args.toe_allowance_ratio,
        arch_strength=args.arch_strength,
        heel_hold_strength=args.heel_hold_strength,
    )


def discover_shoes(alignment_root: Path, shoe_names: Optional[Iterable[str]], max_shoes: Optional[int]) -> list[str]:
    if shoe_names:
        shoes = sorted(dict.fromkeys(str(name) for name in shoe_names))
    else:
        shoes = sorted(
            path.name
            for path in alignment_root.iterdir()
            if path.is_dir() and (path / "foot_aligned_optimized.obj").exists()
        )
    return shoes[:max_shoes] if max_shoes is not None else shoes


def load_style_mode(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return "unknown"
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    shoe_style = metrics.get("shoe_style", {})
    return str(shoe_style.get("mode", "unknown"))


def process_shoe(args: argparse.Namespace, shoe_name: str) -> dict[str, object]:
    shoe_dir = args.alignment_root / shoe_name
    support_dir = shoe_dir / "support"
    foot_obj = shoe_dir / "foot_aligned_optimized.obj"
    support_json = support_dir / "support_footprint.json"
    footbed_npz = support_dir / "pseudo_footbed_heightmap.npz"
    fit_metrics_json = shoe_dir / "fit_metrics.json"
    output_dir = args.output_root / shoe_name
    row: dict[str, object] = {
        "shoe_name": shoe_name,
        "status": "pending",
        "alignment_dir": str(shoe_dir),
        "output_dir": str(output_dir),
        "foot_obj": str(foot_obj),
        "support_json": str(support_json),
        "footbed_npz": str(footbed_npz),
        "fit_metrics_json": str(fit_metrics_json),
    }

    required = [
        ("aligned foot OBJ", foot_obj),
        ("support footprint JSON", support_json),
        ("pseudo-footbed NPZ", footbed_npz),
        ("fit metrics JSON", fit_metrics_json),
    ]
    for label, path in required:
        if not path.exists():
            row.update(status="missing_input", error=f"Missing {label}: {path}")
            return row

    style_mode = load_style_mode(fit_metrics_json)
    row["style_mode"] = style_mode
    if style_mode != "normal" and not args.allow_non_normal:
        row.update(status="skipped_non_normal", error=f"style_mode={style_mode}; rerun with --allow-non-normal to force")
        return row

    if output_dir.exists() and (output_dir / "pseudo_last.obj").exists() and not args.overwrite:
        row.update(status="skipped_existing")
        return row

    output_dir.mkdir(parents=True, exist_ok=True)
    stale_error = output_dir / "error.txt"
    if stale_error.exists():
        stale_error.unlink()

    foot_mesh = load_triangle_mesh(foot_obj)
    result = build_pseudo_last(
        foot_mesh,
        support_json,
        footbed_npz,
        config=make_config(args),
    )
    artifacts = save_pseudo_last_artifacts(result, output_dir, foot_mesh=foot_mesh)

    sdf_status = "skipped"
    sdf_error = ""
    sdf_path = output_dir / "pseudo_last_sdf.npz"
    if not args.no_sdf:
        sdf_config = FootSDFBuildConfig(
            resolution=args.sdf_resolution,
            padding=args.sdf_padding,
            device=args.device,
        )
        build_and_save_signed_sdf_from_obj(artifacts["pseudo_last_obj"], str(sdf_path), config=sdf_config)
        sdf_status = "ok"
        artifacts["pseudo_last_sdf_npz"] = str(sdf_path)

    result.metrics.update(
        {
            "shoe_name": shoe_name,
            "style_mode": style_mode,
            "inputs": {
                "alignment_dir": str(shoe_dir),
                "foot_obj": str(foot_obj),
                "support_json": str(support_json),
                "footbed_npz": str(footbed_npz),
                "fit_metrics_json": str(fit_metrics_json),
            },
            "artifacts": artifacts,
            "sdf_status": sdf_status,
            "sdf_error": sdf_error,
        }
    )
    result.save_metrics_json(output_dir / "pseudo_last_metrics.json")

    row.update(
        status="ok",
        vertex_count=int(result.metrics["vertex_count"]),
        face_count=int(result.metrics["face_count"]),
        method=str(result.metrics.get("method", "")),
        boundary_edge_count=int(result.metrics["boundary_edge_count"]),
        section_count=int(result.metrics["section_count"]),
        support_length=float(result.metrics["support_length"]),
        foot_length=float(result.metrics["foot_length"]),
        toe_extension_length=float(result.metrics["toe_extension_length"]),
        max_width_violation=float(result.metrics["max_width_violation"]),
        bottom_footbed_distance_p95_abs=float(result.metrics["bottom_footbed_distance_p95_abs"]),
        plantar_rmse_to_footbed=float(result.metrics.get("plantar_rmse_to_footbed", 0.0)),
        support_conformity_ratio_max=float(result.metrics.get("support_conformity_ratio_max", 0.0)),
        toe_component_count_after_s_box=int(result.metrics.get("toe_component_count_after_s_box", 0)),
        slice_fallback_count=int(result.metrics.get("slice_fallback_count", 0)),
        sdf_status=sdf_status,
        pseudo_last_obj=artifacts["pseudo_last_obj"],
        pseudo_last_bottom_surface_obj=artifacts["pseudo_last_bottom_surface_obj"],
        pseudo_last_sdf_npz=str(sdf_path) if not args.no_sdf else "",
        pseudo_last_sections_npz=artifacts["pseudo_last_sections_npz"],
        pseudo_last_metrics_json=artifacts["pseudo_last_metrics_json"],
        section_overlays_png=artifacts["section_overlays_png"],
        pseudo_last_overlay_png=artifacts["pseudo_last_overlay_png"],
    )
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    fieldnames = [
        "shoe_name",
        "status",
        "style_mode",
        "vertex_count",
        "face_count",
        "method",
        "boundary_edge_count",
        "section_count",
        "support_length",
        "foot_length",
        "toe_extension_length",
        "max_width_violation",
        "bottom_footbed_distance_p95_abs",
        "plantar_rmse_to_footbed",
        "support_conformity_ratio_max",
        "toe_component_count_after_s_box",
        "slice_fallback_count",
        "sdf_status",
        "pseudo_last_obj",
        "pseudo_last_bottom_surface_obj",
        "pseudo_last_sdf_npz",
        "pseudo_last_sections_npz",
        "pseudo_last_metrics_json",
        "section_overlays_png",
        "pseudo_last_overlay_png",
        "error",
    ]
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.alignment_root.exists():
        raise FileNotFoundError(f"Missing alignment root: {args.alignment_root}")
    shoes = discover_shoes(args.alignment_root, args.shoe_name, args.max_shoes)
    if not shoes:
        raise RuntimeError(f"No aligned shoes found in {args.alignment_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, shoe_name in enumerate(shoes, start=1):
        print(f"[{index}/{len(shoes)}] {shoe_name}")
        try:
            row = process_shoe(args, shoe_name)
            print(f"  status={row.get('status')} style={row.get('style_mode', '')}")
        except Exception as exc:
            output_dir = args.output_root / shoe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "error.txt").write_text(traceback.format_exc())
            row = {
                "shoe_name": shoe_name,
                "status": "failed",
                "output_dir": str(output_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {row['error']}")
        rows.append(row)
        write_summary(rows, args.output_root)

    write_summary(rows, args.output_root)
    print(f"Wrote {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
