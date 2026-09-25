"""What a StockX-style ring of cameras can see of a shoe.

The capture is 36 views, 10 degrees apart in azimuth, from one fixed
elevation; the processed StockX turntables put every camera within about 5
degrees of horizontal. The report (section 2.1) calls the result ring-observed
reconstruction: the sole, the top-down view of the opening and the interior
are never seen, and more views on the same ring never fix it.

``Ring`` answers two questions, with no rendering of images:

    seen    how many cameras see a surface point: nothing in front of it
            along the ray, and not viewed edge-on
    free    whether a point in space is known to be empty: some camera's ray
            passes through it before reaching any surface, or misses the
            shoe entirely. This is the evidence a ring really provides away
            from the surface - the visual hull, plus depth.

Occlusion uses a z-buffer splatted from dense surface samples. Each buffer is
min-filtered over 3x3 pixels, so a gap between splats never lets a hidden point
show through. Density matters: below about three splats per covered pixel the
bottom of a box read as seen from above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

#: Normalized shoe frame: +Y points down, so "up" is -Y.
UP = np.array([0.0, -1.0, 0.0])


def ring_cameras(centre: np.ndarray, radius: float, elevation_deg: float,
                 views: int = 36) -> np.ndarray:
    """Camera positions on a circle around ``centre``, ``elevation_deg`` up."""

    azimuth = np.deg2rad(np.arange(views) * (360.0 / views))
    e = np.deg2rad(elevation_deg)
    offset = np.stack((np.cos(e) * np.cos(azimuth),
                       np.full(views, -np.sin(e)),
                       np.cos(e) * np.sin(azimuth)), axis=1)
    return centre[None] + radius * offset


@dataclass
class _View:
    eye: torch.Tensor
    forward: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    nearest: torch.Tensor   # (res*res,) eroded front depth; 1e9 where empty


class Ring:
    def __init__(self, splats: np.ndarray, centre: np.ndarray, radius: float,
                 elevation_deg: float, views: int = 36, resolution: int = 768,
                 device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.resolution = resolution
        extent = float(np.linalg.norm(np.ptp(splats, axis=0)))
        # Field of view that just holds the shoe.
        self.focal = (resolution / 2) / (0.6 * extent / radius)
        target = torch.as_tensor(centre, dtype=torch.float32, device=self.device)
        up_world = torch.as_tensor(UP, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(splats, dtype=torch.float32, device=self.device)
        self.views: list[_View] = []
        for eye in ring_cameras(np.asarray(centre, dtype=np.float64), radius, elevation_deg, views):
            eye_t = torch.as_tensor(eye, dtype=torch.float32, device=self.device)
            forward = (target - eye_t) / (target - eye_t).norm()
            right = torch.linalg.cross(forward, up_world)
            right = right / right.norm()
            up = torch.linalg.cross(right, forward)
            view = _View(eye_t, forward, right, up, torch.empty(0))
            depth, x, y, ok = self._project(view, s)
            buffer = torch.full((resolution * resolution,), 1e9, device=self.device)
            buffer.scatter_reduce_(0, y[ok] * resolution + x[ok], depth[ok], reduce="amin")
            buffer = buffer.view(1, 1, resolution, resolution)
            view.nearest = (-functional.max_pool2d(-buffer, 3, stride=1, padding=1)).view(-1)
            self.views.append(view)

    def _project(self, view: _View, p: torch.Tensor):
        rel = p - view.eye
        depth = rel @ view.forward
        x = ((rel @ view.right) / depth * self.focal + self.resolution / 2).long()
        y = ((rel @ view.up) / depth * self.focal + self.resolution / 2).long()
        ok = (x >= 0) & (x < self.resolution) & (y >= 0) & (y < self.resolution) & (depth > 0)
        return depth, x, y, ok

    def _front(self, view: _View, p: torch.Tensor):
        depth, x, y, ok = self._project(view, p)
        index = y.clamp(0, self.resolution - 1) * self.resolution + x.clamp(0, self.resolution - 1)
        return depth, view.nearest[index], ok

    def seen(self, points: np.ndarray, normals: np.ndarray,
             depth_tolerance: float = 1.5 / 262.5, grazing_cos: float = 0.15) -> np.ndarray:
        """(N,) how many cameras see each surface point."""

        p = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        n = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
        count = torch.zeros(len(p), dtype=torch.int32, device=self.device)
        for view in self.views:
            depth, nearest, ok = self._front(view, p)
            direction = p - view.eye
            direction = direction / direction.norm(dim=1, keepdim=True)
            facing = (n * direction).sum(1).abs() >= grazing_cos
            count += (ok & (depth <= nearest + depth_tolerance) & facing).int()
        return count.cpu().numpy()

    def free(self, points: np.ndarray, margin: float = 1.5 / 262.5) -> np.ndarray:
        """(N,) True where some camera's ray passes through the point empty."""

        p = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        known = torch.zeros(len(p), dtype=torch.bool, device=self.device)
        for view in self.views:
            depth, nearest, ok = self._front(view, p)
            # In front of the first surface on its ray, by a margin, or on a
            # ray that meets no surface at all.
            known |= ok & (depth < nearest - margin)
        return known.cpu().numpy()
