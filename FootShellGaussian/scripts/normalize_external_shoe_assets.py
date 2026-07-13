#!/usr/bin/env python3
"""Normalize external shoe assets into a consistent render-friendly layout.

This script keeps the original external/source tree untouched and writes a
normalized copy rooted by manifest shoe name:

    <output_root>/<shoe_name>/
      source/
        <model file>
        materials.mtl                  # generated or patched for OBJ assets
        <texture aliases next to model>
        textures/
          <texture aliases>
      normalization_report.json

It also writes a companion manifest that reuses the existing render metadata
(`source_axes`, `selection`, etc.) while rewriting `model` paths to the
normalized copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".webp", ".exr"}
)
MODEL_SUFFIXES = frozenset({".obj", ".fbx", ".dae", ".glb", ".gltf"})
PSEUDO_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".webp", ".exr"}
)
FALLBACK_MANIFEST = Path(
    "/storage/Abhinay/Shell_Gaussian/FootShellGaussian/configs/external_source_all_shoes_render_manifest.json"
)


@dataclass(frozen=True)
class TextureEntry:
    source_path: Path
    canonical_name: str
    aliases: tuple[str, ...]
    digest: str
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, default=FALLBACK_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to populate the normalized dataset.",
    )
    parser.add_argument("--shoe", action="append", default=None, help="Normalize only this shoe name.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def relative_source_folder(model_rel: str) -> str:
    return Path(model_rel).parts[0]


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path exists; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_like(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_texture_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if any(token in stem for token in ("normal", "nrm", "norm")):
        return "normal"
    if any(token in stem for token in ("rough", "gloss")):
        return "roughness"
    if any(token in stem for token in ("metal", "metallic", "metalness", "metallness")):
        return "metallic"
    if any(token in stem for token in ("ao", "occlusion", "ambient")):
        return "ao"
    if any(token in stem for token in ("opacity", "alpha", "transparency")):
        return "opacity"
    if any(token in stem for token in ("height", "disp", "displace")):
        return "height"
    if any(token in stem for token in ("basecolor", "base_color", "albedo", "diffuse", "color")):
        return "diffuse"
    return "other"


def canonical_texture_sort_key(path: Path, model_path: Path) -> tuple[int, int, int, str]:
    path_parts = {part.lower() for part in path.parts}
    score = 100
    if "textures" in path_parts:
        score -= 30
    if model_path.parent in path.parents:
        score -= 20
    if "source" in path_parts:
        score -= 10
    return (score, len(path.parts), len(path.name), path.as_posix().lower())


def alias_variants(filename: str) -> set[str]:
    variants = {filename}
    suffixes = Path(filename).suffixes
    if suffixes:
        last = suffixes[-1].lower()
        stem_without_last = filename[: -len(suffixes[-1])]
        if last == ".jpeg":
            variants.add(stem_without_last + ".jpg")
        if last == ".jpg":
            variants.add(stem_without_last + ".jpeg")
        if len(suffixes) >= 2 and suffixes[-2].lower() in PSEUDO_IMAGE_SUFFIXES:
            variants.add(stem_without_last)

    current = list(variants)
    for value in current:
        variants.add(value.replace("_", " "))
        variants.add(value.replace(" ", "_"))
    return {value for value in variants if value}


def collect_textures(shoe_root: Path, model_path: Path) -> list[TextureEntry]:
    candidates = sorted(
        [path for path in shoe_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: canonical_texture_sort_key(path, model_path),
    )

    chosen_by_name: dict[str, Path] = {}
    for path in candidates:
        chosen_by_name.setdefault(path.name.lower(), path)

    entries: list[TextureEntry] = []
    for lower_name, path in sorted(chosen_by_name.items()):
        aliases = sorted(alias_variants(path.name))
        entries.append(
            TextureEntry(
                source_path=path,
                canonical_name=path.name,
                aliases=tuple(aliases),
                digest=file_digest(path),
                role=classify_texture_name(path.name),
            )
        )
    return entries


def texture_lookup(entries: list[TextureEntry]) -> dict[str, TextureEntry]:
    lookup: dict[str, TextureEntry] = {}
    for entry in entries:
        for alias in entry.aliases:
            lookup.setdefault(alias.lower(), entry)
    return lookup


def link_texture_aliases(
    entries: list[TextureEntry],
    source_dir: Path,
    mode: str,
) -> tuple[dict[str, str], list[str]]:
    alias_map: dict[str, str] = {}
    collisions: list[str] = []
    used_targets: dict[Path, TextureEntry] = {}
    lookup = texture_lookup(entries)

    alias_destinations: list[tuple[str, Path]] = []
    for entry in entries:
        for alias in entry.aliases:
            alias_destinations.append((alias, source_dir / "textures" / alias))
            alias_destinations.append((alias, source_dir / alias))

    for alias, dst in alias_destinations:
        entry = lookup.get(alias.lower())
        if entry is None:
            continue
        existing = used_targets.get(dst)
        if existing is not None and existing.digest != entry.digest:
            collisions.append(f"{dst}: {existing.source_path.name} vs {entry.source_path.name}")
            continue
        used_targets[dst] = entry
        copy_like(entry.source_path, dst, mode)
        alias_map.setdefault(alias.lower(), os.path.relpath(dst, source_dir))
    return alias_map, collisions


def parse_obj_material_names(obj_text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in obj_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("usemtl "):
            continue
        name = normalize_material_name(line.split(None, 1)[1])
        if name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        names.append("material_0")
    return names


def normalize_material_name(name: str) -> str:
    stripped = name.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        return stripped[1:-1]
    return stripped


def rewrite_obj_text(obj_text: str) -> str:
    lines = obj_text.splitlines()
    output: list[str] = []
    inserted_mtllib = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            if not inserted_mtllib:
                output.append("mtllib materials.mtl")
                inserted_mtllib = True
            continue
        if stripped.startswith("usemtl "):
            mat_name = normalize_material_name(stripped.split(None, 1)[1])
            output.append(f"usemtl {mat_name}")
            continue
        output.append(line)
    if not inserted_mtllib:
        insertion_index = 0
        while insertion_index < len(output) and output[insertion_index].startswith("#"):
            insertion_index += 1
        output.insert(insertion_index, "mtllib materials.mtl")
    return "\n".join(output) + "\n"


def find_existing_mtl(obj_path: Path, shoe_root: Path) -> Path | None:
    obj_text = obj_path.read_text(errors="ignore")
    for raw_line in obj_text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("mtllib "):
            continue
        name = stripped.split(None, 1)[1].strip()
        candidate = obj_path.parent / name
        if candidate.is_file():
            return candidate
        basename = Path(name).name.lower()
        for path in shoe_root.rglob("*"):
            if path.is_file() and path.name.lower() == basename:
                return path
    return None


def fallback_texture_by_role(entries: list[TextureEntry], role: str) -> TextureEntry | None:
    for entry in entries:
        if entry.role == role:
            return entry
    if role == "diffuse":
        non_pbr = [entry for entry in entries if entry.role not in {"normal", "roughness", "metallic", "ao", "height"}]
        if len(non_pbr) == 1:
            return non_pbr[0]
        if non_pbr:
            return non_pbr[0]
        if len(entries) == 1 and entries[0].role == "other":
            return entries[0]
    return None


def patch_mtl_text(
    mtl_text: str,
    entries: list[TextureEntry],
    alias_map: dict[str, str],
) -> str:
    role_fallbacks = {
        "map_kd": fallback_texture_by_role(entries, "diffuse"),
        "map_ka": fallback_texture_by_role(entries, "diffuse"),
        "map_d": fallback_texture_by_role(entries, "opacity"),
        "map_bump": fallback_texture_by_role(entries, "normal"),
        "bump": fallback_texture_by_role(entries, "normal"),
        "norm": fallback_texture_by_role(entries, "normal"),
        "map_pr": fallback_texture_by_role(entries, "roughness"),
        "map_pm": fallback_texture_by_role(entries, "metallic"),
    }
    patched_lines: list[str] = []
    for raw_line in mtl_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            patched_lines.append(raw_line)
            continue
        tokens = stripped.split()
        key = tokens[0].lower()
        if key in role_fallbacks and len(tokens) >= 2:
            original_ref = tokens[-1]
            normalized_ref = alias_map.get(Path(original_ref).name.lower())
            if normalized_ref is None:
                fallback = role_fallbacks[key]
                if fallback is not None:
                    normalized_ref = f"textures/{fallback.canonical_name}"
            if normalized_ref is not None:
                tokens[-1] = normalized_ref
                patched_lines.append(" ".join(tokens))
                continue
        patched_lines.append(raw_line)
    return "\n".join(patched_lines) + "\n"


def build_generated_mtl(material_names: list[str], entries: list[TextureEntry]) -> str:
    diffuse = fallback_texture_by_role(entries, "diffuse")
    normal = fallback_texture_by_role(entries, "normal")
    opacity = fallback_texture_by_role(entries, "opacity")
    roughness = fallback_texture_by_role(entries, "roughness")
    metallic = fallback_texture_by_role(entries, "metallic")

    lines = ["# Generated by normalize_external_shoe_assets.py", ""]
    for name in material_names:
        lines.extend(
            [
                f"newmtl {name}",
                "Ns 10.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.450000",
                "d 1.000000",
                "illum 2",
            ]
        )
        if diffuse is not None:
            lines.append(f"map_Kd textures/{diffuse.canonical_name}")
        if opacity is not None:
            lines.append(f"map_d textures/{opacity.canonical_name}")
        if normal is not None:
            lines.append(f"map_Bump textures/{normal.canonical_name}")
        if roughness is not None:
            lines.append(f"map_Pr textures/{roughness.canonical_name}")
        if metallic is not None:
            lines.append(f"map_Pm textures/{metallic.canonical_name}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def normalize_obj_asset(
    model_path: Path,
    shoe_root: Path,
    output_source_dir: Path,
    entries: list[TextureEntry],
    alias_map: dict[str, str],
) -> dict[str, Any]:
    obj_text = model_path.read_text(errors="ignore")
    material_names = parse_obj_material_names(obj_text)
    output_obj_name = model_path.name
    output_obj_path = output_source_dir / output_obj_name
    output_obj_path.write_text(rewrite_obj_text(obj_text))

    existing_mtl = find_existing_mtl(model_path, shoe_root)
    if existing_mtl is not None:
        mtl_text = existing_mtl.read_text(errors="ignore")
        output_mtl_text = patch_mtl_text(mtl_text, entries, alias_map)
        mtl_source = str(existing_mtl)
    else:
        output_mtl_text = build_generated_mtl(material_names, entries)
        mtl_source = None

    output_mtl_path = output_source_dir / "materials.mtl"
    output_mtl_path.write_text(output_mtl_text)
    return {
        "output_model": output_obj_name,
        "generated_material_file": str(output_mtl_path),
        "source_material_file": mtl_source,
        "material_names": material_names,
    }


def normalize_one_shoe(
    shoe_cfg: dict[str, Any],
    source_root: Path,
    output_root: Path,
    copy_mode: str,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shoe_name = str(shoe_cfg["name"])
    model_rel = Path(str(shoe_cfg["model"]))
    model_path = source_root / model_rel
    original_shoe_root = source_root / relative_source_folder(str(model_rel))
    if not model_path.is_file():
        raise FileNotFoundError(f"{shoe_name}: model file not found: {model_path}")

    output_shoe_root = output_root / shoe_name
    ensure_empty_dir(output_shoe_root, overwrite)
    output_source_dir = output_shoe_root / "source"
    output_source_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_textures(original_shoe_root, model_path)
    alias_map, collisions = link_texture_aliases(entries, output_source_dir, copy_mode)

    if model_path.suffix.lower() == ".obj":
        model_info = normalize_obj_asset(
            model_path=model_path,
            shoe_root=original_shoe_root,
            output_source_dir=output_source_dir,
            entries=entries,
            alias_map=alias_map,
        )
        output_model_name = model_info["output_model"]
    else:
        output_model_name = model_path.name
        copy_like(model_path, output_source_dir / output_model_name, copy_mode)
        model_info = {
            "output_model": output_model_name,
            "generated_material_file": None,
            "source_material_file": None,
            "material_names": None,
        }

    report = {
        "shoe": shoe_name,
        "original_shoe_root": str(original_shoe_root),
        "source_model": str(model_path),
        "output_model": str(output_source_dir / output_model_name),
        "texture_count": len(entries),
        "texture_alias_count": len(alias_map),
        "texture_collisions": collisions,
        "textures": [
            {
                "canonical_name": entry.canonical_name,
                "source_path": str(entry.source_path),
                "role": entry.role,
                "aliases": list(entry.aliases),
            }
            for entry in entries
        ],
        "obj_material_handling": model_info,
    }
    write_json(output_shoe_root / "normalization_report.json", report)

    normalized_cfg = dict(shoe_cfg)
    normalized_cfg["model"] = f"{shoe_name}/source/{output_model_name}"
    return normalized_cfg, report


def main() -> None:
    args = parse_args()
    manifest = load_json(args.input_manifest)
    shoes = manifest.get("shoes", [])
    if not isinstance(shoes, list):
        raise ValueError(f"Invalid manifest shoes list: {args.input_manifest}")

    selected = set(args.shoe or [])
    normalized_shoes: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    args.output_root.mkdir(parents=True, exist_ok=True)
    for shoe_cfg in shoes:
        shoe_name = str(shoe_cfg["name"])
        if selected and shoe_name not in selected:
            continue
        normalized_cfg, report = normalize_one_shoe(
            shoe_cfg=shoe_cfg,
            source_root=args.source_root,
            output_root=args.output_root,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
        )
        normalized_shoes.append(normalized_cfg)
        reports.append(report)
        print(f"[ok] normalized {shoe_name}")

    normalized_manifest = {
        "description": (
            "Normalized external shoe assets produced by normalize_external_shoe_assets.py. "
            "Model choices and render metadata were copied from the original external-source manifest."
        ),
        "source_manifest": str(args.input_manifest),
        "normalized_source_root": str(args.output_root),
        "shoes": normalized_shoes,
    }
    write_json(args.output_manifest, normalized_manifest)
    write_json(args.output_root / "normalization_summary.json", {"reports": reports})
    print(f"Wrote normalized manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
