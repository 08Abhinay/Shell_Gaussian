"""Can a training sample be given a sign, and where does the sign go wrong?

Two independent ways to say whether a point is inside shoe material:

    parity   walk out along a fiber from the skin; every crossing of the shoe
             surface toggles inside/outside. Needs the walk to start outside
             material and the mesh to be closed along that fiber.
    winding  the generalized winding number of the whole mesh at the point.
             Needs consistently oriented faces, tolerates holes.

They fail in different ways, so where they agree the sign is trustworthy, and
where they disagree the reason is informative. This also answers a question
the material stage cannot: which fibers *start* inside material, because the
foot was fitted slightly into the shoe. Those fibers have their parity
inverted, and their first crossing is an exit, not ``delta_in``.

    python -m anatomical_coordinates.stage1.signs --shoes sneaker_1 crocs
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import common
from .mesh_audit import sign_ready
from .winding import inside, orientation

AMBIGUOUS = (0.25, 0.75)


def _along(arclength: np.ndarray, curves: np.ndarray, site: np.ndarray,
           at: np.ndarray) -> np.ndarray:
    """Canonical positions at arclength ``at`` along fiber ``site``.

    Every fiber's arclength increases monotonically, so offsetting each row by
    its index makes one global sorted array and one ``searchsorted`` serves all
    queries.
    """

    count, samples = arclength.shape
    finite = np.isfinite(arclength)
    last = np.maximum(finite.sum(axis=1) - 1, 0)
    reach = arclength[np.arange(count), last]
    # Past a fiber's end, repeat its last sample (with a tiny rising
    # arclength so each row stays strictly sorted and below the next row).
    length = np.where(finite, arclength,
                      reach[:, None] + 1e-9 * np.arange(samples)[None, :])
    curves = np.where(finite[:, :, None], curves,
                      curves[np.arange(count), last][:, None, :])
    offset = 10.0 * np.arange(count)[:, None]
    flat = (length + offset).ravel()
    key = at + 10.0 * site
    upper = np.clip(np.searchsorted(flat, key), 1, flat.size - 1)
    # Never step across a row boundary.
    row_start = site * samples
    upper = np.clip(upper, row_start + 1, row_start + samples - 1)
    lower = upper - 1
    span = flat[upper] - flat[lower]
    t = np.where(np.isfinite(span) & (span > 0), (key - flat[lower]) / np.where(span > 0, span, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    points = curves.reshape(-1, 3)
    return points[lower] + t[:, None] * (points[upper] - points[lower])


def study(name: str, flows, paths, device="cuda") -> dict:
    mat = common.material(name)
    shoe = common.shoe_mesh(name)
    curves = np.asarray(paths["curves"], dtype=np.float64)
    arclength = np.asarray(paths["arclength"], dtype=np.float64)
    reach = np.nanmax(arclength, axis=1)
    count = len(arclength)

    site = np.asarray(mat["crossing_site"], dtype=np.int64)
    r = np.asarray(mat["crossing_r"], dtype=np.float64)
    layers = np.bincount(site, minlength=count)

    # Query points: the fiber's first sample off the skin, then the midpoint
    # of every gap - skin to first crossing, between crossings, and a little
    # beyond the last one.
    q_site, q_at, q_gap = [np.arange(count)], [np.minimum(1e-3, 0.5 * reach)], [np.full(count, -1)]
    starts = np.searchsorted(site, np.arange(count))
    for s in np.nonzero(layers)[0]:
        mine = r[starts[s]:starts[s] + layers[s]]
        edges = np.concatenate(([0.0], mine, [min(mine[-1] + 0.02, reach[s])]))
        mids = 0.5 * (edges[:-1] + edges[1:])
        q_site.append(np.full(len(mids), s))
        q_at.append(mids)
        q_gap.append(np.arange(len(mids)))
    q_site = np.concatenate(q_site)
    q_at = np.concatenate(q_at)
    q_gap = np.concatenate(q_gap)

    canonical = _along(arclength, curves, q_site, q_at)
    placed = flows.to_shoe(canonical, name)
    # Twins removed and pieces oriented: without this a double-sided export
    # reads 0 everywhere and every fiber looks as if it starts outside.
    vertices, faces, _ = sign_ready(shoe.vertices, shoe.faces, device)
    sign = orientation(vertices, faces, device)
    w = inside(placed, vertices, faces, device, sign=sign)

    is_start = q_gap < 0
    start_w = np.full(count, np.nan)
    start_w[q_site[is_start]] = w[is_start]
    starts_inside = start_w > 0.5

    gap = ~is_start
    gs, gj, gw = q_site[gap], q_gap[gap], w[gap]
    # Parity from the skin, flipped where the winding number says the fiber
    # starts inside material.
    parity = (gj % 2 == 1) ^ starts_inside[gs]
    confident = (gw < AMBIGUOUS[0]) | (gw > AMBIGUOUS[1])
    even = layers[gs] % 2 == 0
    agree = parity == (gw > 0.5)
    plain_parity = gj % 2 == 1

    penetration = np.asarray(mat["penetration_mm"]) > 0
    covered = layers > 0
    return {
        "shoe": name,
        "faces": int(len(shoe.faces)),
        "orientation": sign,
        "sites": int(count),
        "covered_sites": int(covered.sum()),
        "gap_points": int(gap.sum()),
        "winding_ambiguous_fraction": float((~confident).mean()),
        "agreement_even_confident": float(agree[even & confident].mean()) if (even & confident).any() else None,
        "agreement_even_confident_without_start_fix": (
            float((plain_parity == (gw > 0.5))[even & confident].mean())
            if (even & confident).any() else None),
        "agreement_odd_confident": float(agree[~even & confident].mean()) if (~even & confident).any() else None,
        "odd_site_fraction_of_covered": float((layers[covered] % 2 == 1).mean()) if covered.any() else None,
        "starts_inside_fraction_of_covered": float(starts_inside[covered].mean()) if covered.any() else None,
        "starts_inside_sites": int(starts_inside.sum()),
        "penetration_sites": int(penetration.sum()),
        "starts_inside_and_penetration_seen": int((starts_inside & penetration).sum()),
        "starts_inside_missed_by_penetration": int((starts_inside & ~penetration).sum()),
        "_per_site": {
            "start_winding": start_w.astype(np.float32),
            "starts_inside": starts_inside,
            "layers": layers.astype(np.int16),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument("--shard", type=int, nargs=2, default=None,
                        metavar=("INDEX", "TOTAL"), help="process every TOTAL-th shoe from INDEX")
    args = parser.parse_args()

    out = common.STAGE1_OUTPUT / "signs"
    out.mkdir(parents=True, exist_ok=True)
    flows = common.load_flows()
    _, _, paths = common.load_canonical(device=flows.device)
    names = args.shoes or flows.names
    if args.shard is not None:
        index, total = args.shard
        names = names[index::total]
    for name in names:
        started = time.time()
        record = study(name, flows, paths, flows.device)
        per_site = record.pop("_per_site")
        np.savez_compressed(out / f"{name}.npz", **per_site)
        record["seconds"] = time.time() - started
        (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name[:40]:40s} ambiguous {record['winding_ambiguous_fraction']*100:5.1f}%  "
              f"agree(even) {100*(record['agreement_even_confident'] or 0):5.1f}% "
              f"[no start fix {100*(record['agreement_even_confident_without_start_fix'] or 0):5.1f}%]  "
              f"agree(odd) {100*(record['agreement_odd_confident'] or 0):5.1f}%  "
              f"starts-inside {record['starts_inside_sites']:5d} "
              f"(penetration saw {record['starts_inside_and_penetration_seen']})  "
              f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
