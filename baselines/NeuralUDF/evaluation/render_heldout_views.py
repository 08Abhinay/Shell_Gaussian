#!/usr/bin/env python3
"""Render NeuralUDF predictions for the held-out shoe cameras."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import torch
from PIL import Image


NEURALUDF_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NEURALUDF_ROOT.parents[1]
sys.path.insert(0, str(NEURALUDF_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_tools_blender.core import (  # noqa: E402
    GSHELL_LOADER_LEFT_ROTATION,
    RESOLUTION,
)
from dataset_tools_blender.neuraludf.pipeline import (  # noqa: E402
    neuraludf_intrinsic,
    normalized_neuraludf_pose,
)
from exp_runner_blending import Runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe", required=True)
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--source-views", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--maximum-views",
        type=int,
        help="Render only the first N held-out views for a smoke check.",
    )
    return parser.parse_args()


def runner_args(checkpoint: Path, chunk_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        learning_rate=0.0,
        learning_rate_geo=0.0,
        sparse_weight=0.0,
        is_finetune=False,
        reg_weights_schedule=False,
        vis_ray=False,
        init_checkpoint=str(checkpoint.resolve()),
        batch_size=chunk_size,
    )


def read_test_frames(source_scene: Path) -> list[tuple[int, str, np.ndarray]]:
    payload = json.loads(
        (source_scene / "transforms_test.json").read_text(encoding="utf-8")
    )
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("transforms_test.json contains no held-out frames")

    result = []
    for frame in frames:
        source_name = Path(str(frame.get("file_path", ""))).name
        if not source_name.startswith("img") or Path(source_name).suffix.lower() != ".jpg":
            raise ValueError(f"Unexpected held-out frame: {frame.get('file_path')!r}")
        try:
            source_index = int(Path(source_name).stem.removeprefix("img")) - 1
        except ValueError as error:
            raise ValueError(f"Unexpected held-out frame: {source_name!r}") from error
        saved_c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if saved_c2w.shape != (4, 4) or not np.isfinite(saved_c2w).all():
            raise ValueError(f"Invalid held-out pose for {source_name}")
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        result.append((source_index, source_name, effective_c2w))
    return result


def camera_rays(
    intrinsic: torch.Tensor,
    query_c2w: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=query_c2w.device, dtype=query_c2w.dtype),
        torch.arange(width, device=query_c2w.device, dtype=query_c2w.dtype),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    local = torch.matmul(
        torch.linalg.inv(intrinsic[:3, :3]), pixels[..., None]
    ).squeeze(-1)
    local = torch.nn.functional.normalize(local, dim=-1)
    directions = torch.matmul(
        query_c2w[:3, :3], local[..., None]
    ).squeeze(-1)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    origins = query_c2w[:3, 3].expand_as(directions)
    return origins, directions


def render_query(
    runner: Runner,
    query_c2w: np.ndarray,
    chunk_size: int,
    source_views: int,
) -> np.ndarray:
    query = torch.from_numpy(query_c2w.astype(np.float32)).to(runner.device)
    intrinsic = runner.dataset.intrinsics_all[0]
    rays_o, rays_d = camera_rays(
        intrinsic, query, runner.dataset.H, runner.dataset.W
    )
    ref_c2w, src_c2ws, src_intrinsics, src_images, _ = (
        runner.dataset.get_nearest_src_info(query, num=source_views)
    )
    background = torch.ones((1, 3), device=runner.device)
    colors = []
    for rays_o_batch, rays_d_batch in zip(
        rays_o.reshape(-1, 3).split(chunk_size),
        rays_d.reshape(-1, 3).split(chunk_size),
    ):
        near, far = runner.dataset.near_far_from_sphere(
            rays_o_batch, rays_d_batch
        )
        # The renderer differentiates the UDF with respect to 3D samples to
        # obtain normals, so point gradients must remain enabled at inference.
        result = runner.renderer.render(
            rays_o_batch,
            rays_d_batch,
            near,
            far,
            color_maps=src_images,
            w2cs=torch.inverse(src_c2ws),
            intrinsics=src_intrinsics,
            query_c2w=ref_c2w,
            cos_anneal_ratio=1.0,
            background_rgb=background,
        )
        if result.get("color") is None:
            raise RuntimeError("NeuralUDF renderer did not return color")
        colors.append(result["color"].detach().cpu())
    image = torch.cat(colors, dim=0).reshape(
        runner.dataset.H, runner.dataset.W, 3
    )
    return np.clip(image.numpy(), 0.0, 1.0)


def white_ground_truth(
    source_scene: Path, source_name: str
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(
        Image.open(source_scene / "image" / source_name).convert("RGB"),
        dtype=np.float32,
    ) / 255.0
    mask_name = f"{Path(source_name).stem}.png"
    mask = np.asarray(
        Image.open(source_scene / "mask" / mask_name).convert("L"),
        dtype=np.uint8,
    )
    foreground = mask >= 128
    image = np.where(foreground[..., None], image, 1.0)
    return image, np.where(foreground, 255, 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.source_views <= 0:
        raise ValueError("chunk-size and source-views must be positive")
    for path, label in (
        (args.conf, "configuration"),
        (args.checkpoint, "checkpoint"),
        (args.source_scene, "source scene"),
        (args.prepared_scene, "prepared scene"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    output = args.output.resolve()
    render_dir = output / "renders"
    gt_dir = output / "gt"
    mask_dir = output / "masks"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if output.exists():
        shutil.rmtree(output)
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # The trained camera archive defines the normalization used by the network.
    with np.load(args.prepared_scene / "cameras_sphere.npz") as cameras:
        scale_matrix = cameras["scale_mat_0"].astype(np.float64)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    runtime_args = runner_args(args.checkpoint, args.chunk_size)
    runner = Runner(
        str(args.conf.resolve()),
        mode="evaluate_heldout",
        case=args.shoe,
        is_continue=False,
        args=runtime_args,
    )
    if runner.dataset.H != RESOLUTION[1] or runner.dataset.W != RESOLUTION[0]:
        raise ValueError(
            f"Unexpected training resolution: {runner.dataset.W}x{runner.dataset.H}"
        )

    frames = read_test_frames(args.source_scene)
    if args.maximum_views is not None:
        if args.maximum_views <= 0:
            raise ValueError("maximum-views must be positive")
        frames = frames[: args.maximum_views]

    manifest_frames = []
    for ordinal, (source_index, source_name, effective_c2w) in enumerate(frames):
        query_c2w = normalized_neuraludf_pose(effective_c2w, scale_matrix)
        predicted_bgr = render_query(
            runner, query_c2w, args.chunk_size, args.source_views
        )
        ground_truth_rgb, mask = white_ground_truth(args.source_scene, source_name)
        output_name = f"{Path(source_name).stem}.png"
        cv.imwrite(
            str(render_dir / output_name),
            np.rint(predicted_bgr * 255.0).astype(np.uint8),
        )
        Image.fromarray(
            np.rint(ground_truth_rgb * 255.0).astype(np.uint8), mode="RGB"
        ).save(gt_dir / output_name)
        Image.fromarray(mask, mode="L").save(mask_dir / output_name)
        manifest_frames.append(
            {
                "ordinal": ordinal,
                "source_view_index": source_index,
                "source_name": source_name,
                "output_name": output_name,
            }
        )
        print(
            f"[rendered] {args.shoe} {ordinal + 1}/{len(frames)} {output_name}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "method": "NeuralUDF (masked open-surface configuration)",
        "shoe": args.shoe,
        "checkpoint": str(args.checkpoint.resolve()),
        "source_scene": str(args.source_scene.resolve()),
        "prepared_scene": str(args.prepared_scene.resolve()),
        "query_view_count": len(frames),
        "training_view_count": runner.dataset.n_images,
        "color_source_views": "nearest views from the training split only",
        "source_view_count": args.source_views,
        "chunk_size": args.chunk_size,
        "width": runner.dataset.W,
        "height": runner.dataset.H,
        "frames": manifest_frames,
    }
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
