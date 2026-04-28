#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from PIL import Image


def parse_color(value):
    names = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }
    if value.lower() in names:
        return names[value.lower()]
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use black, white, or R,G,B.")
    rgb = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255].")
    return rgb


def replace_link_or_dir(link_path, target_path):
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target_path, link_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create a COLMAP scene with mask-composited RGBA images."
    )
    parser.add_argument("--shoe-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--background", default="black", type=parse_color)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shoe_root = args.shoe_root.resolve()
    source_scene = shoe_root / "undistorted"
    source_images = source_scene / "images"
    source_sparse = source_scene / "sparse"
    source_masks = shoe_root / "masks"

    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing images directory: {source_images}")
    if not source_sparse.is_dir():
        raise FileNotFoundError(f"Missing sparse directory: {source_sparse}")
    if not source_masks.is_dir():
        raise FileNotFoundError(f"Missing masks directory: {source_masks}")

    output_scene = args.output_root.resolve() / shoe_root.name / "undistorted"
    output_images = output_scene / "images"
    output_sparse = output_scene / "sparse"

    if output_scene.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_scene}")

    output_images.mkdir(parents=True, exist_ok=True)
    replace_link_or_dir(output_sparse, source_sparse)

    image_paths = sorted(
        path for path in source_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {source_images}")

    for image_path in image_paths:
        mask_path = source_masks / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.BILINEAR)
        background = Image.new("RGB", image.size, args.background)
        composited = Image.composite(image, background, mask)
        composited.putalpha(mask)

        # Keep COLMAP image basenames/extensions unchanged. PIL writes PNG data
        # even if the suffix is .jpg, and PIL readers use the file signature.
        composited.save(output_images / image_path.name, format="PNG")

    print(f"Masked scene written to: {output_scene}")
    print(f"Images: {len(image_paths)}")
    print(f"Sparse symlink: {output_sparse} -> {source_sparse}")


if __name__ == "__main__":
    main()
