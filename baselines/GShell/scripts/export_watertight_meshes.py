#!/usr/bin/env python
"""Export watertight meshes from completed GShell checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.gshell_tets_geometry import GShellTetsGeometry  # noqa: E402
from render import obj  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "canonical-512-768",
        help="Root containing completed GShell output directories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "shoes_mc_normfix_512_768.json",
        help="Training config used to create the checkpoints.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Export only this output directory name, e.g. Foo_canonical. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_flags(config_path: Path) -> SimpleNamespace:
    flags = SimpleNamespace(
        boxscale=[1, 1, 1],
        use_tanh_deform=False,
        use_sdf_mlp=True,
        use_msdf_mlp=False,
        use_eikonal=True,
        use_mesh_msdf_reg=True,
        sphere_init=False,
        sphere_init_norm=0.5,
        sdf_mlp_pretrain_steps=1,
        n_hidden=6,
        d_hidden=256,
        n_freq=6,
        skip_in=[3],
        use_float16=False,
        visualize_watertight=True,
        gshell_grid=128,
        mesh_scale=1.0,
    )
    with config_path.open("r") as f:
        payload = json.load(f)
    for key, value in payload.items():
        setattr(flags, key, value)

    # One pretrain step is enough to initialize the module before loading the
    # trained checkpoint. The loaded state fully replaces these initial weights.
    flags.sdf_mlp_pretrain_steps = 1
    flags.visualize_watertight = True
    return flags


def resolve_scene_dirs(output_root: Path, scene_names: list[str] | None) -> list[Path]:
    if scene_names:
        return [output_root / name for name in scene_names]
    return sorted(path for path in output_root.iterdir() if path.is_dir() and (path / "mesh" / "model.pt").exists())


def export_one(scene_dir: Path, flags: SimpleNamespace, overwrite: bool) -> dict[str, object]:
    model_path = scene_dir / "mesh" / "model.pt"
    out_dir = scene_dir / "mesh_watertight"
    out_mesh = out_dir / "mesh.obj"

    if not model_path.exists():
        return {"scene": scene_dir.name, "status": "missing_model", "output": str(out_mesh)}
    if out_mesh.exists() and not overwrite:
        return {"scene": scene_dir.name, "status": "exists", "output": str(out_mesh)}

    geometry = GShellTetsGeometry(flags.gshell_grid, flags.mesh_scale, flags)
    state = torch.load(model_path, map_location="cpu")
    geometry.load_state_dict(state)
    geometry.eval()

    with torch.no_grad():
        mesh_result = geometry.getMesh(None)
        if "imesh_watertight" not in mesh_result:
            raise RuntimeError(f"{scene_dir.name}: geometry did not return imesh_watertight")
        out_dir.mkdir(parents=True, exist_ok=True)
        obj.write_obj(str(out_dir) + "/", mesh_result["imesh_watertight"], save_material=False)

    vertex_count = int(mesh_result["imesh_watertight"].v_pos.shape[0])
    face_count = int(mesh_result["imesh_watertight"].t_pos_idx.shape[0])
    return {
        "scene": scene_dir.name,
        "status": "exported",
        "vertices": vertex_count,
        "faces": face_count,
        "output": str(out_mesh),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GShell watertight export")
    if not args.output_root.exists():
        raise FileNotFoundError(f"Output root not found: {args.output_root}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    flags = load_flags(args.config)
    scene_dirs = resolve_scene_dirs(args.output_root, args.scene)
    if not scene_dirs:
        raise RuntimeError(f"No completed GShell scene directories found under {args.output_root}")

    rows = []
    for index, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{index}/{len(scene_dirs)}] {scene_dir.name}", flush=True)
        row = export_one(scene_dir, flags, args.overwrite)
        rows.append(row)
        if row["status"] == "exported":
            print(
                f"  exported vertices={row['vertices']} faces={row['faces']} -> {row['output']}",
                flush=True,
            )
        else:
            print(f"  {row['status']} -> {row['output']}", flush=True)

    summary_path = args.output_root / "watertight_export_summary.json"
    with summary_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
