"""Camera and held-out asset loading for the direct Blender evaluation dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EvaluationCamera:
    frame_name: str
    image_path: Path
    mask_path: Path
    invdepth_path: Path
    effective_c2w: np.ndarray
    width: int
    height: int
    fov_x_rad: float
    radius: float
    ring_index: int
    elevation_deg: float
    azimuth_deg: float


def project_rotation_x(angle_rad: float) -> np.ndarray:
    """Match the row-sign convention used by dataset_tools_blender."""
    sine, cosine = math.sin(angle_rad), math.cos(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, sine, 0.0],
            [0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_test_cameras(scene_root: str | Path) -> list[EvaluationCamera]:
    """Load held-out cameras in effective GShell world coordinates."""
    root = Path(scene_root).expanduser().resolve()
    payload = _read_json(root / "transforms_test.json")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms_test.json contains no held-out frames")
    if payload.get("pose_convention") != "legacy_gshell_saved_c2w_for_fixed_loader":
        raise ValueError("Unexpected saved camera-pose convention")

    canonicalization_path = root / "blender_canonicalization.json"
    turntable_path = root / "turntable_manifest.json"
    if canonicalization_path.is_file():
        canonicalization = _read_json(canonicalization_path)
        camera_contract = canonicalization.get("camera_contract")
        if not isinstance(camera_contract, dict):
            raise ValueError("blender_canonicalization.json has no camera_contract")
        resolution = camera_contract.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError("Camera resolution must contain width and height")
        width, height = (int(resolution[0]), int(resolution[1]))
        fov_x_rad = math.radians(float(camera_contract["fov_x_deg"]))
        radius = float(camera_contract["radius"])
    elif turntable_path.is_file():
        turntable = _read_json(turntable_path)
        camera_contract = turntable.get("camera")
        if not isinstance(camera_contract, dict):
            raise ValueError("turntable_manifest.json has no camera contract")
        first_image = root / Path(str(frames[0]["file_path"]))
        with Image.open(first_image) as image:
            width, height = image.size
        fov_x_rad = math.radians(float(camera_contract["horizontal_fov_degrees"]))
        radius = float(camera_contract["radius"])
    else:
        raise FileNotFoundError("Scene has no Blender or turntable camera manifest")

    loader_rotation = project_rotation_x(math.pi / 2.0)
    cameras: list[EvaluationCamera] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Held-out frame entry must be an object")
        image_relative = Path(str(frame["file_path"]))
        invdepth_relative = Path(str(frame["invdepth_path"]))
        frame_name = image_relative.stem
        saved_c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if saved_c2w.shape != (4, 4) or not np.all(np.isfinite(saved_c2w)):
            raise ValueError(f"Invalid camera matrix for {frame_name}")
        effective_c2w = loader_rotation @ saved_c2w
        rotation = effective_c2w[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"Non-rigid camera rotation for {frame_name}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(f"Improper camera rotation for {frame_name}")
        actual_radius = float(np.linalg.norm(effective_c2w[:3, 3]))
        if not np.isclose(actual_radius, radius, atol=1e-6):
            raise ValueError(f"Unexpected camera radius for {frame_name}: {actual_radius}")

        image_path = root / image_relative
        mask_path = root / "mask" / f"{frame_name}.png"
        invdepth_path = root / invdepth_relative
        for asset in (image_path, mask_path, invdepth_path):
            if not asset.is_file():
                raise FileNotFoundError(f"Held-out asset is missing: {asset}")
        cameras.append(
            EvaluationCamera(
                frame_name=frame_name,
                image_path=image_path,
                mask_path=mask_path,
                invdepth_path=invdepth_path,
                effective_c2w=effective_c2w,
                width=width,
                height=height,
                fov_x_rad=fov_x_rad,
                radius=radius,
                ring_index=int(frame["ring_index"]),
                elevation_deg=float(frame["elevation_deg"]),
                azimuth_deg=float(frame["azimuth_deg"]),
            )
        )
    return cameras
