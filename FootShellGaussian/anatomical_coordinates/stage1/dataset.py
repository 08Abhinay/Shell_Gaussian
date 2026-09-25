"""The Stage 1 training set: one file of samples per CAD shoe.

The report's make-or-break test (section 6.1) trains identical auto-decoders
that differ only in their input coordinates:

    A    object-centric: the CAD file's own frame, centred and scaled to the
         unit sphere per shoe
    A+   aligned: the prepare stage's normalized shoe frame - ground plane,
         heel-to-toe axis and functional length, no anatomy
    B    anatomical: the address (s, r) - the canonical skin point a sample's
         fiber lands on, and canonical arclength out along it (negative
         inside the fitted anatomy)

Every sample carries all three, so the models see exactly the same points and
targets. The target is the unsigned distance to the shoe surface (see
``geometry`` for why not the signed one); the winding number is stored too,
with a per-shoe flag saying whether it can be trusted as a sign.

For the blind-zone test each surface sample is labelled by what a StockX-style
ring would show of it. Elevation 0 is the StockX case; 15 and 30 degrees are
the report's controllable ablation.

    visible   seen by at least one of the 36 cameras
    interior  unseen, and the first surface out from the skin along its fiber
              (cavity lining, footbed top, inside of the tongue)
    sole      unseen, on the plantar side, not interior (midsole, outsole)
    top       any other unseen surface

    python -m anatomical_coordinates.stage1.dataset --shard 0 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import common
from .base_envelope import region_of
from .geometry import DistanceField, sample_surface
from .mesh_audit import sign_ready
from .ring import Ring
from .winding import inside, orientation
from ..coordinate_mapping.address import ADDRESSED, INSIDE_ANATOMY

SURFACE, NEAR, VOLUME = 0, 1, 2
VISIBLE, INTERIOR, SOLE, TOP = 0, 1, 2, 3
REGION_NAMES = ("visible", "interior", "sole", "top")
ELEVATIONS = (0, 15, 30)
MM = common.MILLIMETRES


def _object_centric(name: str, points: np.ndarray, mesh_vertices: np.ndarray) -> np.ndarray:
    """Model A's input: the CAD frame, per-shoe unit sphere."""

    prep = json.loads((common.PIPELINE_OUTPUT / "inputs" / "shoe_preparation" / name
                       / "shoe_preparation.json").read_text())
    matrix = np.asarray(prep["normalization"]["normalized_to_shoe"], dtype=np.float64)

    def apply(p):
        return p @ matrix[:3, :3].T + matrix[:3, 3]

    original = apply(mesh_vertices)
    centre = 0.5 * (original.min(0) + original.max(0))
    scale = np.linalg.norm(original - centre, axis=1).max()
    return (apply(points) - centre) / scale


def build(name: str, flows, semantics, tracer, sign_trust: dict,
          counts=(30000, 60000, 20000), seed: int = 0) -> dict:
    started = time.time()
    shoe = common.shoe_mesh(name)
    vertices, faces = shoe.vertices, shoe.faces
    n_surface, n_near, n_volume = counts
    rng = np.random.default_rng(seed)

    surface, _, normal = sample_surface(vertices, faces, n_surface, seed=seed)
    source = rng.integers(0, n_surface, n_near)
    sigma = np.where(np.arange(n_near) < n_near // 2, 0.5, 3.0) / MM
    near = surface[source] + rng.normal(size=(n_near, 3)) * sigma[:, None]
    lower, upper = vertices.min(0) - 15 / MM, vertices.max(0) + 15 / MM
    volume = rng.uniform(lower, upper, size=(n_volume, 3))
    points = np.concatenate((surface, near, volume))
    kind = np.concatenate((np.full(n_surface, SURFACE), np.full(n_near, NEAR),
                           np.full(n_volume, VOLUME))).astype(np.int8)
    origin = np.concatenate((np.arange(n_surface), source, np.full(n_volume, -1))).astype(np.int32)

    distance = DistanceField(vertices, faces)
    udf = np.concatenate((np.zeros(n_surface), distance(near), distance(volume)))

    sv, sf, _ = sign_ready(vertices, faces, flows.device)
    winding = inside(points, sv, sf, flows.device, sign=orientation(sv, sf, flows.device))

    # Anatomical address of every sample (exact tracing, as the address stage).
    book = common.address_book(semantics, tracer, flows, name)
    address = book.query(points, exact=True, verify=False, step=0.002, max_steps=1500)
    usable = ((address.outcome == ADDRESSED) | (address.outcome == INSIDE_ANATOMY)) & (address.face >= 0)
    s = address.surface_point(tracer)
    r = np.where(usable, address.arclength, np.nan)

    # Ring views: splats dense enough that hidden points cannot see through.
    splats, _, _ = sample_surface(vertices, faces, 6_000_000, seed=seed + 1)
    centre = 0.5 * (vertices.min(0) + vertices.max(0))
    radius = 3.0 * float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    seen = {}
    free = {}
    for elevation in ELEVATIONS:
        ring = Ring(splats, centre, radius, elevation, device=flows.device)
        seen[elevation] = ring.seen(surface, normal)
        free[elevation] = ring.free(points)
        del ring

    # Regions of the surface samples, from the elevation-0 ring.
    mat = common.material(name)
    names = [str(x) for x in mat["label_names"]]
    site_region = region_of(mat["site_point"], mat["site_label"], names)
    site = address.face[:n_surface]
    ok = usable[:n_surface]
    delta_in = np.where(mat["covered"], np.maximum(mat["signed_delta_in"], 0.0), np.nan)
    r_surface = r[:n_surface]
    interior = ok & np.isfinite(delta_in[np.maximum(site, 0)]) & (r_surface <= delta_in[np.maximum(site, 0)] + 2 / MM)
    plantar = ok & (site_region[np.maximum(site, 0)] == "plantar")
    region = np.full(n_surface, TOP, dtype=np.int8)
    region[plantar] = SOLE
    region[interior] = INTERIOR
    region[seen[0] > 0] = VISIBLE
    # Near samples inherit their source surface point's region.
    region_all = np.concatenate((region, region[source], np.full(n_volume, -1, dtype=np.int8)))

    out = common.STAGE1_OUTPUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{name}.npz",
        points=points.astype(np.float32),
        kind=kind, origin=origin,
        udf=udf.astype(np.float32),
        winding=winding.astype(np.float32),
        coords_a=_object_centric(name, points, vertices).astype(np.float32),
        coords_s=s.astype(np.float32),
        coords_r=r.astype(np.float32),
        b_valid=usable,
        region=region_all,
        **{f"seen{e}": seen[e].astype(np.int8) for e in ELEVATIONS},
        **{f"free{e}": free[e] for e in ELEVATIONS},
    )
    record = {
        "shoe": name,
        "samples": int(len(points)),
        "b_valid_fraction": float(usable.mean()),
        "sign_trusted": bool(sign_trust.get(name, False)),
        "region_fractions_elev0": {REGION_NAMES[k]: float((region == k).mean()) for k in range(4)},
        "surface_seen_fraction": {str(e): float((seen[e] > 0).mean()) for e in ELEVATIONS},
        "free_fraction_of_volume": {str(e): float(free[e][kind == VOLUME].mean()) for e in ELEVATIONS},
        "seconds": time.time() - started,
    }
    (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument("--shard", type=int, nargs=2, default=None, metavar=("INDEX", "TOTAL"))
    args = parser.parse_args()
    flows = common.load_flows()
    semantics, tracer, _ = common.load_canonical(device=flows.device)
    trust = {}
    for path in (common.STAGE1_OUTPUT / "signs").glob("*.json"):
        row = json.loads(path.read_text())
        agreement = row.get("agreement_even_confident_without_start_fix")
        trust[row["shoe"]] = agreement is not None and agreement >= 0.95
    names = args.shoes or flows.names
    if args.shard is not None:
        names = names[args.shard[0]::args.shard[1]]
    for name in names:
        record = build(name, flows, semantics, tracer, trust)
        f = record["region_fractions_elev0"]
        print(f"{name[:40]:40s} B-valid {record['b_valid_fraction']*100:5.1f}%  "
              f"visible {f['visible']*100:5.1f}%  interior {f['interior']*100:5.1f}%  "
              f"sole {f['sole']*100:5.1f}%  top {f['top']*100:5.1f}%  "
              f"seen@15 {record['surface_seen_fraction']['15']*100:5.1f}%  "
              f"seen@30 {record['surface_seen_fraction']['30']*100:5.1f}%  {record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
