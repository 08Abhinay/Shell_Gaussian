"""Build the canonical SUPR-Foot signed SDF grid.

Default input:
    baselines/SUPR/output/debug_playground/supr_male_right_foot_neutral.obj

Default output:
    FootShellGaussian/data/foot_prior/supr_male_right_foot_sdf.npz
"""

import argparse
import sys
from pathlib import Path


def _repo_paths() -> tuple[Path, Path]:
    footshell_root = Path(__file__).resolve().parents[1]
    project_root = footshell_root.parent
    return footshell_root, project_root


def main() -> None:
    footshell_root, project_root = _repo_paths()
    if str(footshell_root) not in sys.path:
        sys.path.insert(0, str(footshell_root))

    from foot_prior.foot_sdf import FootSDFBuildConfig, build_and_save_signed_sdf_from_obj

    default_input = (
        project_root
        / "baselines"
        / "SUPR"
        / "output"
        / "debug_playground"
        / "supr_male_right_foot_neutral.obj"
    )
    default_output = footshell_root / "data" / "foot_prior" / "supr_male_right_foot_sdf.npz"

    parser = argparse.ArgumentParser(description="Build a signed SDF grid from SUPR-Foot OBJ.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input foot OBJ path.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output SDF .npz path.")
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution per axis.")
    parser.add_argument("--padding", type=float, default=0.03, help="Padding around mesh bounds.")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Point chunk size.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for SDF generation.")
    parser.add_argument(
        "--no-cap-boundary",
        action="store_true",
        help="Do not close the ankle boundary before computing signs.",
    )
    args = parser.parse_args()

    config = FootSDFBuildConfig(
        resolution=args.resolution,
        padding=args.padding,
        chunk_size=args.chunk_size,
        cap_boundary=not args.no_cap_boundary,
        device=args.device,
    )
    sdf, bounds_min, bounds_max = build_and_save_signed_sdf_from_obj(
        obj_path=str(args.input),
        output_path=str(args.output),
        config=config,
    )

    print(f"Saved foot SDF: {args.output}")
    print(f"  source:     {args.input}")
    print(f"  resolution: {sdf.shape}")
    print(f"  bounds_min: {bounds_min}")
    print(f"  bounds_max: {bounds_max}")
    print(f"  sdf min:    {sdf.min():.6f}")
    print(f"  sdf max:    {sdf.max():.6f}")
    print(f"  sdf mean:   {sdf.mean():.6f}")


if __name__ == "__main__":
    main()
