#!/usr/bin/env python3
"""Evaluate finished masked-black-background SuGaR runs.

For each run this script writes image pairs and metrics for:
  - vanilla 3DGS, using the bundled gaussian_splatting/render.py + metrics.py
  - refined SuGaR, using the refined Gaussian-on-surface renderer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import open3d as o3d
import torch
import torchvision
from tqdm import tqdm


SUGAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
from gaussian_splatting.utils.image_utils import psnr  # noqa: E402
from gaussian_splatting.utils.loss_utils import ssim  # noqa: E402
from sugar_scene.gs_model import GaussianSplattingWrapper  # noqa: E402
from sugar_scene.sugar_model import SuGaR  # noqa: E402
from sugar_utils.spherical_harmonics import SH2RGB  # noqa: E402


RUN_RE = re.compile(
    r"^(?P<shoe>.+)_masked_blackbg_gs(?P<gs_iter>\d+)_(?P<reg>[^_]+)_v(?P<vertices>\d+)_g(?P<gpf>\d+)_r(?P<ref_iter>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=SUGAR_ROOT / "output" / "sugar_runs",
        help="Directory containing *_masked_blackbg_* run folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific run directory to evaluate. May be passed multiple times.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Logical CUDA device to use.")
    parser.add_argument("--pattern", default="*_masked_blackbg_*", help="Run glob when --run-dir is not provided.")
    parser.add_argument("--skip-3dgs", action="store_true", help="Skip vanilla 3DGS render.py + metrics.py.")
    parser.add_argument("--skip-sugar", action="store_true", help="Skip refined SuGaR evaluation.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if metric files already exist.")
    parser.add_argument("--shoe", help="Shoe name for a generic --run-dir.")
    parser.add_argument("--gs-iteration", type=int, default=7_000)
    parser.add_argument("--refined-iteration", type=int, default=15_000)
    parser.add_argument("--gaussians-per-triangle", type=int, default=1)
    parser.add_argument(
        "--white-background",
        action="store_true",
        help="Render and score against the white-background dataset contract.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUGAR_ROOT / "output" / "sugar_runs" / "blackbg_eval_summary.tsv",
        help="TSV summary path.",
    )
    return parser.parse_args()


def abelde_env() -> dict[str, str]:
    env = os.environ.copy()
    home = Path("/storage/Abhinay/home_ab5298")
    cache = home / ".cache"
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(cache)
    env["TORCH_HOME"] = str(cache / "torch")
    env["PIP_CACHE_DIR"] = str(cache / "pip")
    env["CUDA_CACHE_PATH"] = str(cache / "nv" / "ComputeCache")
    env["TMPDIR"] = str(home / "tmp")
    for value in ["XDG_CACHE_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR"]:
        Path(env[value]).mkdir(parents=True, exist_ok=True)
    return env


def storage_home_path(path: Path) -> Path:
    text = str(path)
    home_prefix = "/home/ab5298"
    if text == home_prefix or text.startswith(home_prefix + os.sep):
        return Path("/storage/Abhinay/home_ab5298" + text[len(home_prefix):])
    return path


def get_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path.resolve() for path in args.run_dir]
    else:
        runs = sorted(path.resolve() for path in args.runs_root.glob(args.pattern) if path.is_dir())
    return [run for run in runs if RUN_RE.match(run.name)]


def read_scene_path(run_dir: Path) -> Path:
    result_path = run_dir / "refinement_run_result.json"
    if result_path.is_file():
        return storage_home_path(
            Path(
                json.loads(
                    result_path.read_text(encoding="utf-8")
                )["scene_path"]
            )
        )
    log_path = run_dir / "logs" / "pipeline.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing pipeline log: {log_path}")
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("Scene: "):
            return storage_home_path(
                Path(line.split("Scene: ", 1)[1].strip())
            )
    raise ValueError(f"Could not find scene path in {log_path}")


def expected_test_view_count(scene_path: Path) -> int:
    split_path = scene_path / "transforms_test.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing test split: {split_path}")
    frames = json.loads(split_path.read_text(encoding="utf-8")).get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Test split contains no frames: {split_path}")
    return len(frames)


def required_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_one(pattern: str, root: Path, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {label} matching {root / pattern}")
    if len(matches) > 1:
        print(f"Warning: multiple {label} matches, using {matches[0]}", flush=True)
    return matches[0]


def read_refinement_artifacts(run_dir: Path, ref_iter: int) -> tuple[Path, Path]:
    """Resolve completed refinement outputs without assuming directory depth."""
    result_path = run_dir / "refinement_run_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(
                f"Refinement is not complete according to {result_path}: "
                f"{result.get('status')!r}"
            )
        manifest_iter = int(result.get("refinement_iterations", -1))
        if manifest_iter != ref_iter:
            raise ValueError(
                f"Refinement iteration mismatch in {result_path}: "
                f"expected {ref_iter}, found {manifest_iter}"
            )

        try:
            refined_model = storage_home_path(Path(result["refined_model_path"]))
            coarse_mesh = storage_home_path(Path(result["accepted_coarse_mesh_path"]))
        except KeyError as error:
            raise ValueError(
                f"Missing artifact path {error.args[0]!r} in {result_path}"
            ) from error

        required_path(refined_model, "refined model")
        required_path(coarse_mesh, "accepted coarse mesh")
        if refined_model.name != f"{ref_iter}.pt":
            raise ValueError(
                f"Expected refined checkpoint {ref_iter}.pt, found {refined_model}"
            )
        return refined_model, coarse_mesh

    # Older outputs predate the result manifest and may use varying nesting depths.
    refined_model = find_one(
        f"**/{ref_iter}.pt", run_dir / "refined", "refined model"
    )
    coarse_mesh = find_one("**/*.ply", run_dir / "coarse_mesh", "coarse mesh")
    return refined_model, coarse_mesh


def read_3dgs_results(gs_dir: Path, iteration: int) -> dict[str, float]:
    results_path = gs_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing 3DGS results: {results_path}")
    data = json.loads(results_path.read_text())
    method = f"ours_{iteration}"
    if method not in data:
        method = next(iter(data))
    values = data[method]
    return {
        "psnr": float(values["PSNR"]),
        "ssim": float(values["SSIM"]),
        "lpips": float(values["LPIPS"]),
    }


def run_official_3dgs_eval(
    run_dir: Path,
    scene_path: Path,
    gs_iter: int,
    gpu: int,
    overwrite: bool,
    white_background: bool,
) -> dict[str, object]:
    gs_dir = required_path(run_dir / "vanilla_3dgs", "3DGS output")
    renders_dir = gs_dir / "test" / f"ours_{gs_iter}" / "renders"
    gt_dir = gs_dir / "test" / f"ours_{gs_iter}" / "gt"
    results_path = gs_dir / "results.json"

    if overwrite or not (results_path.is_file() and renders_dir.is_dir() and gt_dir.is_dir()):
        env = abelde_env()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
        render_cmd = [
            sys.executable,
            "render.py",
            "-s",
            str(scene_path),
            "-m",
            str(gs_dir),
            "--iteration",
            str(gs_iter),
            "--skip_train",
            "--eval",
            "--quiet",
        ]
        if white_background:
            render_cmd.append("-w")
        metrics_cmd = [sys.executable, "metrics.py", "-m", str(gs_dir)]
        subprocess.run(render_cmd, cwd=SUGAR_ROOT / "gaussian_splatting", env=env, check=True)
        subprocess.run(metrics_cmd, cwd=SUGAR_ROOT / "gaussian_splatting", env=env, check=True)

    metrics = read_3dgs_results(gs_dir, gs_iter)
    return {
        "method": "3dgs",
        **metrics,
        "views": len(list(renders_dir.glob("*.png"))),
        "renders": str(renders_dir),
        "gt": str(gt_dir),
    }


def tensor_to_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(tensor.detach().clamp(0.0, 1.0).cpu(), path)


def evaluate_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    lpips_model: LPIPS,
) -> dict[str, float]:
    return {
        "psnr": float(psnr(render, gt).mean().item()),
        "ssim": float(ssim(render, gt).mean().item()),
        "lpips": float(lpips_model(render, gt).mean().item()),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in ["psnr", "ssim", "lpips"]
    }


def run_sugar_eval(
    run_dir: Path,
    scene_path: Path,
    gs_iter: int,
    ref_iter: int,
    gpf: int,
    gpu: int,
    overwrite: bool,
    white_background: bool,
) -> dict[str, object]:
    eval_dir = run_dir / "eval" / "sugar_gaussian_rasterizer"
    renders_dir = eval_dir / "renders"
    gt_dir = eval_dir / "gt"
    metrics_path = eval_dir / "metrics.json"
    per_view_path = eval_dir / "per_view.json"

    if not overwrite and metrics_path.is_file() and renders_dir.is_dir() and gt_dir.is_dir():
        values = json.loads(metrics_path.read_text())
        return {
            "method": "sugar",
            "psnr": float(values["PSNR"]),
            "ssim": float(values["SSIM"]),
            "lpips": float(values["LPIPS"]),
            "views": int(values["views"]),
            "renders": str(renders_dir),
            "gt": str(gt_dir),
        }

    torch.cuda.set_device(gpu)
    gs_dir = required_path(run_dir / "vanilla_3dgs", "3DGS output")
    refined_model, coarse_mesh = read_refinement_artifacts(run_dir, ref_iter)

    nerfmodel = GaussianSplattingWrapper(
        source_path=str(scene_path),
        output_path=str(gs_dir) + os.sep,
        iteration_to_load=gs_iter,
        load_gt_images=True,
        eval_split=True,
        eval_split_interval=8,
        background=[1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        white_background=white_background,
    )

    o3d_mesh = o3d.io.read_triangle_mesh(str(coarse_mesh))
    if len(o3d_mesh.triangles) == 0:
        raise ValueError(f"Coarse mesh has no triangles: {coarse_mesh}")

    checkpoint = torch.load(refined_model, map_location=nerfmodel.device)
    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint["state_dict"]["_points"],
        colors=SH2RGB(checkpoint["state_dict"]["_sh_coordinates_dc"][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode="average",
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=gpf,
    )
    refined_sugar.load_state_dict(checkpoint["state_dict"])
    refined_sugar.eval()

    lpips_model = LPIPS(net_type="vgg", version="0.1").to(nerfmodel.device).eval()
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        device=nerfmodel.device,
    )
    sh_deg = nerfmodel.gaussians.active_sh_degree

    per_view = []
    with torch.no_grad():
        for cam_idx in tqdm(range(len(nerfmodel.test_cameras)), desc=f"SuGaR eval {run_dir.name}"):
            gt = nerfmodel.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
            render = refined_sugar.render_image_gaussian_rasterizer(
                nerf_cameras=nerfmodel.test_cameras,
                camera_indices=cam_idx,
                bg_color=bg,
                sh_deg=sh_deg,
                compute_color_in_rasterizer=True,
            ).clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)

            if render.shape != gt.shape:
                raise ValueError(f"Render/GT shape mismatch for {run_dir.name} view {cam_idx}: {render.shape} vs {gt.shape}")

            view_name = nerfmodel.test_cam_list[cam_idx].image_name
            metrics = evaluate_pair(render, gt, lpips_model)
            per_view.append({"index": cam_idx, "image_name": view_name, **metrics})
            tensor_to_png(render[0], renders_dir / f"{cam_idx:05d}_{view_name}.png")
            tensor_to_png(gt[0], gt_dir / f"{cam_idx:05d}_{view_name}.png")

    avg = mean_metrics(per_view)
    metrics_out = {
        "method": "sugar_gaussian_rasterizer",
        "PSNR": avg["psnr"],
        "SSIM": avg["ssim"],
        "LPIPS": avg["lpips"],
        "views": len(per_view),
        "scene_path": str(scene_path),
        "gs_iteration": gs_iter,
        "refined_iteration": ref_iter,
        "refined_model": str(refined_model),
        "coarse_mesh": str(coarse_mesh),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_out, indent=2) + "\n")
    per_view_path.write_text(json.dumps(per_view, indent=2) + "\n")

    return {
        "method": "sugar",
        "psnr": avg["psnr"],
        "ssim": avg["ssim"],
        "lpips": avg["lpips"],
        "views": len(per_view),
        "renders": str(renders_dir),
        "gt": str(gt_dir),
    }


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["shoe", "run_id", "method", "PSNR", "SSIM", "LPIPS", "views", "renders", "gt"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "shoe": row["shoe"],
                    "run_id": row["run_id"],
                    "method": row["method"],
                    "PSNR": f"{row['psnr']:.12f}",
                    "SSIM": f"{row['ssim']:.12f}",
                    "LPIPS": f"{row['lpips']:.12f}",
                    "views": row["views"],
                    "renders": row["renders"],
                    "gt": row["gt"],
                }
            )


def main() -> None:
    args = parse_args()
    torch.hub.set_dir(
        "/storage/Abhinay/home_ab5298/.cache/torch/hub"
    )
    runs = get_runs(args)
    if not runs:
        raise SystemExit(f"No runs found under {args.runs_root} matching {args.pattern}")

    rows: list[dict[str, object]] = []
    for run_dir in runs:
        match = RUN_RE.match(run_dir.name)
        if match is not None:
            shoe = match["shoe"]
            gs_iter = int(match["gs_iter"])
            ref_iter = int(match["ref_iter"])
            gpf = int(match["gpf"])
        else:
            shoe = args.shoe or run_dir.name
            gs_iter = args.gs_iteration
            ref_iter = args.refined_iteration
            gpf = args.gaussians_per_triangle
        scene_path = read_scene_path(run_dir)
        required_path(scene_path / "images", "masked images")
        required_path(scene_path / "sparse", "COLMAP sparse reconstruction")
        required_path(run_dir / "vanilla_3dgs" / "point_cloud" / f"iteration_{gs_iter}" / "point_cloud.ply", "3DGS point cloud")
        required_path(run_dir / "refined", "refined model directory")
        required_path(run_dir / "coarse_mesh", "coarse mesh directory")

        print(f"\n=== Evaluating {run_dir.name} ===", flush=True)
        print(f"Scene: {scene_path}", flush=True)
        expected_views = expected_test_view_count(scene_path)
        print(f"Expected held-out views: {expected_views}", flush=True)

        if not args.skip_3dgs:
            row = run_official_3dgs_eval(
                run_dir,
                scene_path,
                gs_iter,
                args.gpu,
                args.overwrite,
                args.white_background,
            )
            if row["views"] != expected_views:
                raise ValueError(
                    f"Expected {expected_views} held-out 3DGS views, found {row['views']}"
                )
            row.update({"shoe": shoe, "run_id": run_dir.name})
            rows.append(row)
            print(f"3DGS: PSNR={row['psnr']:.4f} SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}", flush=True)

        if not args.skip_sugar:
            row = run_sugar_eval(
                run_dir,
                scene_path,
                gs_iter,
                ref_iter,
                gpf,
                args.gpu,
                args.overwrite,
                args.white_background,
            )
            if row["views"] != expected_views:
                raise ValueError(
                    f"Expected {expected_views} held-out SuGaR views, found {row['views']}"
                )
            row.update({"shoe": shoe, "run_id": run_dir.name})
            rows.append(row)
            print(f"SuGaR: PSNR={row['psnr']:.4f} SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}", flush=True)

        write_summary(rows, args.summary)

    write_summary(rows, args.summary)
    print(f"\nSummary written to {args.summary}", flush=True)


if __name__ == "__main__":
    main()
