#!/usr/bin/env python3
"""Prepare external shoe assets into a render-ready preprocessed dataset.

This script keeps the flow intentionally simple:

1. Walk each top-level shoe folder recursively.
2. Choose the model file to use for that shoe.
3. Normalize the asset into a consistent render-friendly layout.
4. Write a prepared manifest that Blender can use directly.

The output looks like:

    <output_root>/
      source/
        <shoe_name>/
          source/
            ...
      manifests/
        external_source_preprocessed_render_manifest.json
      reports/
        preparation_summary.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import normalize_external_shoe_assets as normalize_assets


RENDERABLE_MODEL_SUFFIXES = frozenset({".dae", ".fbx", ".glb", ".gltf", ".obj"})
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".webp", ".exr"}
)
MATERIAL_SUFFIXES = frozenset({".mtl"})
ARCHIVE_SUFFIXES = frozenset({".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"})
MODEL_BASE_SCORE = {
    ".glb": 500,
    ".gltf": 480,
    ".fbx": 460,
    ".dae": 430,
    ".obj": 400,
}
DEFAULT_INPUT_MANIFEST = Path(
    "/storage/Abhinay/Shell_Gaussian/FootShellGaussian/configs/external_source_normalized_all_shoes_render_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/external_source_preprocessed"
)
DEFAULT_OUTPUT_MANIFEST_NAME = "external_source_preprocessed_render_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to populate the preprocessed dataset.",
    )
    parser.add_argument("--shoe", action="append", default=None, help="Prepare only this shoe name.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def manifest_shoe_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = normalize_assets.load_json(path)
    shoes = payload.get("shoes", [])
    if not isinstance(shoes, list):
        raise ValueError(f"Invalid manifest shoes list: {path}")

    mapping: dict[str, dict[str, Any]] = {}
    for shoe_cfg in shoes:
        if not isinstance(shoe_cfg, dict):
            continue
        name = str(shoe_cfg.get("name", "")).strip()
        if name:
            mapping[name] = dict(shoe_cfg)
    return mapping


def resolve_output_manifest_path(output_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    return output_root / "manifests" / DEFAULT_OUTPUT_MANIFEST_NAME


def relative_paths_with_suffixes(root: Path, suffixes: set[str] | frozenset[str]) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def model_candidate_score(
    path: Path,
    *,
    shoe_root: Path,
    source_root: Path,
    configured_model_rel: str | None,
) -> int:
    rel_from_source = path.relative_to(source_root).as_posix()
    rel_from_shoe = path.relative_to(shoe_root)
    score = MODEL_BASE_SCORE.get(path.suffix.lower(), 0)

    if configured_model_rel and rel_from_source == configured_model_rel:
        score += 120
    if path.parent == shoe_root / "source":
        score += 30
    if "source" in {part.lower() for part in rel_from_shoe.parts}:
        score += 15
    if path.stem.lower() in {"model", "mesh"}:
        score += 5

    score -= len(rel_from_shoe.parts)
    return score


def discover_model_candidates(
    *,
    shoe_root: Path,
    source_root: Path,
    configured_model_rel: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(shoe_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in RENDERABLE_MODEL_SUFFIXES:
            continue

        relative_path = path.relative_to(source_root).as_posix()
        rows.append(
            {
                "relative_path": relative_path,
                "basename": path.name,
                "suffix": path.suffix.lower(),
                "configured_match": relative_path == configured_model_rel,
                "score": model_candidate_score(
                    path,
                    shoe_root=shoe_root,
                    source_root=source_root,
                    configured_model_rel=configured_model_rel,
                ),
                "size_bytes": path.stat().st_size,
            }
        )

    rows.sort(key=lambda row: (-int(row["score"]), str(row["relative_path"])))
    return rows


def inspect_shoe_folder(
    *,
    shoe_root: Path,
    source_root: Path,
    configured_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    configured_model = None
    if configured_cfg is not None:
        configured_model = str(configured_cfg.get("model") or "").strip() or None

    model_candidates = discover_model_candidates(
        shoe_root=shoe_root,
        source_root=source_root,
        configured_model_rel=configured_model,
    )
    selected_model = model_candidates[0]["relative_path"] if model_candidates else None
    if selected_model is None:
        selection_reason = "no_renderable_model_found"
    elif model_candidates[0]["configured_match"]:
        selection_reason = "configured_model"
    else:
        selection_reason = "best_discovered_candidate"

    return {
        "shoe": shoe_root.name,
        "configured_model": configured_model,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "manifest_present": configured_cfg is not None,
        "model_candidates": model_candidates,
        "texture_count": len(relative_paths_with_suffixes(shoe_root, IMAGE_SUFFIXES)),
        "material_file_count": len(relative_paths_with_suffixes(shoe_root, MATERIAL_SUFFIXES)),
        "archive_file_count": len(relative_paths_with_suffixes(shoe_root, ARCHIVE_SUFFIXES)),
    }


def build_shoe_cfg(
    shoe_name: str,
    configured_cfg: dict[str, Any] | None,
    selected_model_rel: str,
) -> dict[str, Any]:
    shoe_cfg = dict(configured_cfg or {})
    shoe_cfg["name"] = shoe_name
    shoe_cfg["model"] = selected_model_rel
    shoe_cfg.setdefault("source_axes", "auto")
    return shoe_cfg


def remove_obsolete_outputs(output_root: Path) -> None:
    stale_paths = [
        output_root / "reports" / "scan_report.json",
        output_root / "reports" / "per_shoe",
    ]
    for path in stale_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def clean_output_root(output_root: Path, overwrite: bool, selected_shoes: set[str]) -> None:
    if output_root.exists() and overwrite and not selected_shoes:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs(output_root)


def write_preparation_summary(
    path: Path,
    *,
    source_root: Path,
    input_manifest: Path | None,
    output_root: Path,
    output_manifest: Path,
    copy_mode: str,
    prepared_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "source_root": str(source_root),
        "input_manifest": str(input_manifest) if input_manifest is not None else None,
        "output_root": str(output_root),
        "output_manifest": str(output_manifest),
        "copy_mode": copy_mode,
        "prepared_count": len(prepared_rows),
        "skipped_count": len(skipped_rows),
        "prepared_shoes": [
            {
                "shoe": row["shoe"],
                "selected_model": row["selected_model"],
                "configured_model": row["configured_model"],
                "selection_reason": row["selection_reason"],
                "manifest_present": row["manifest_present"],
                "texture_count": row["texture_count"],
                "material_file_count": row["material_file_count"],
                "archive_file_count": row["archive_file_count"],
                "model_candidates": row["model_candidates"],
            }
            for row in prepared_rows
        ],
        "skipped_shoes": skipped_rows,
    }
    normalize_assets.write_json(path, payload)


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    selected_shoes = set(args.shoe or [])
    configured_map = manifest_shoe_map(args.input_manifest if args.input_manifest.is_file() else None)

    clean_output_root(args.output_root, overwrite=args.overwrite, selected_shoes=selected_shoes)
    reports_root = args.output_root / "reports"
    prepared_source_root = args.output_root / "source"
    output_manifest = resolve_output_manifest_path(args.output_root, args.output_manifest)

    prepared_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for shoe_root in sorted(path for path in args.source_root.iterdir() if path.is_dir()):
        if selected_shoes and shoe_root.name not in selected_shoes:
            continue

        row = inspect_shoe_folder(
            shoe_root=shoe_root,
            source_root=args.source_root,
            configured_cfg=configured_map.get(shoe_root.name),
        )
        if row["selected_model"] is None:
            skipped_rows.append(
                {
                    "shoe": row["shoe"],
                    "configured_model": row["configured_model"],
                    "selection_reason": row["selection_reason"],
                    "model_candidates": row["model_candidates"],
                }
            )
            print(f"[skip] {row['shoe']} (no renderable model found)")
            continue

        row["shoe_cfg"] = build_shoe_cfg(
            shoe_name=row["shoe"],
            configured_cfg=configured_map.get(shoe_root.name),
            selected_model_rel=str(row["selected_model"]),
        )
        prepared_rows.append(row)

    if not prepared_rows:
        raise SystemExit("No renderable shoe assets were discovered during preparation.")

    normalized_shoes: list[dict[str, Any]] = []
    for row in prepared_rows:
        normalized_cfg, _report = normalize_assets.normalize_one_shoe(
            shoe_cfg=row["shoe_cfg"],
            source_root=args.source_root,
            output_root=prepared_source_root,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
        )
        normalized_shoes.append(normalized_cfg)
        print(
            f"[ok] prepared {row['shoe']} "
            f"({row['selection_reason']}: {row['selected_model']})"
        )

    normalized_manifest = {
        "description": (
            "Preprocessed external shoe assets produced by prepare_external_shoe_assets.py. "
            "Each shoe folder was inspected recursively, the selected model was normalized, "
            "and this manifest points Blender at the prepared asset layout."
        ),
        "source_root": str(args.source_root),
        "source_manifest": str(args.input_manifest) if args.input_manifest.is_file() else None,
        "prepared_source_root": str(prepared_source_root),
        "shoes": normalized_shoes,
    }
    normalize_assets.write_json(output_manifest, normalized_manifest)
    write_preparation_summary(
        reports_root / "preparation_summary.json",
        source_root=args.source_root,
        input_manifest=args.input_manifest if args.input_manifest.is_file() else None,
        output_root=args.output_root,
        output_manifest=output_manifest,
        copy_mode=args.copy_mode,
        prepared_rows=prepared_rows,
        skipped_rows=skipped_rows,
    )

    manifest_backed_count = sum(1 for row in prepared_rows if row["manifest_present"])
    discovered_count = len(prepared_rows) - manifest_backed_count
    print(
        "Preparation complete:",
        f"prepared={len(prepared_rows)}",
        f"skipped={len(skipped_rows)}",
        f"manifest_backed={manifest_backed_count}",
        f"auto_discovered={discovered_count}",
    )
    print(f"Wrote prepared manifest: {output_manifest}")


if __name__ == "__main__":
    main()
