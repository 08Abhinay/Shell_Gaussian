#!/usr/bin/env python3
"""Compute PSNR, SSIM, and LPIPS for saved NeuralUDF held-out renders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-network", choices=("alex", "vgg"), default="vgg")
    return parser.parse_args()


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def main() -> None:
    args = parse_args()
    evaluation = args.evaluation.resolve()
    manifest_path = evaluation / "render_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("Render manifest contains no frames")

    device = torch.device(args.device)
    perceptual = lpips.LPIPS(net=args.lpips_network).to(device).eval()
    rows = []
    for frame in frames:
        name = frame["output_name"]
        prediction = image_tensor(evaluation / "renders" / name, device)
        reference = image_tensor(evaluation / "gt" / name, device)
        with torch.inference_mode():
            mse = float(torch.mean((prediction - reference) ** 2).item())
            psnr = -10.0 * math.log10(max(mse, 1e-12))
            ssim_value = float(
                ssim(
                    prediction,
                    reference,
                    data_range=1.0,
                    size_average=True,
                ).item()
            )
            lpips_value = float(
                perceptual(prediction * 2.0 - 1.0, reference * 2.0 - 1.0)
                .mean()
                .item()
            )
        rows.append(
            {
                "source_view_index": int(frame["source_view_index"]),
                "image": name,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim_value,
                "lpips": lpips_value,
            }
        )
        print(
            f"[scored] {name} PSNR={psnr:.4f} "
            f"SSIM={ssim_value:.6f} LPIPS={lpips_value:.6f}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "method": manifest["method"],
        "shoe": manifest["shoe"],
        "views": len(rows),
        "metric_domain": "full-resolution white-background RGB",
        "lpips_network": args.lpips_network,
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
        "per_view": rows,
    }
    (evaluation / "image_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (evaluation / "image_metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        "AVERAGE_TEST "
        f"PSNR={summary['psnr']:.6f} "
        f"SSIM={summary['ssim']:.6f} "
        f"LPIPS={summary['lpips']:.6f} "
        f"views={summary['views']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
