"""How much of each shoe does the measured base envelope already explain?

representation.pdf section 1.2 puts the anatomical prior into a base shell,

    B(u, v, r) = max{ g, delta_in - r, r - delta_out },

and a residual R for everything the shell cannot say. Before anything is
learned, the measured per-site g, delta_in and delta_out already define B for
every training shoe, so the premise can be checked directly: evaluate B at
each shoe's addressed surface samples (area-uniform, so fractions are area
fractions of the shoe surface) and sort each sample into

    on the shell     |B| <= tolerance: the envelope's inner or outer face
    inside the shell B < -tolerance: a surface inside the solid slab B
                     makes - an air gap or layer R would have to carve
    outside          B > tolerance, or no material measured at that site:
                     material B misses, which R would have to add

    python -m anatomical_coordinates.stage1.base_envelope
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import common
from ..coordinate_mapping.address import ADDRESSED, INSIDE_ANATOMY

TOLERANCES_MM = (1.0, 2.0)


def region_of(sites_point: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """A coarse anatomical region per site: plantar, dorsal/sides, leg, other."""

    region = np.full(len(labels), "other", dtype=object)
    foot = labels == names.index("foot_skin")
    # Canonical frame: +Y points down, the sole plane is y = 0.
    plantar = foot & (sites_point[:, 1] > -0.02)
    region[foot & ~plantar] = "dorsal_sides"
    region[plantar] = "plantar"
    region[labels == names.index("lower_leg_skin")] = "leg"
    region[labels == names.index("ankle_transition")] = "leg"
    return region


def measure(name: str) -> dict:
    mat = common.material(name)
    addr = np.load(common.PIPELINE_OUTPUT / "addresses" / name / "addresses.npz")
    names = [str(x) for x in mat["label_names"]]
    region = region_of(mat["site_point"], mat["site_label"], names)

    outcome = addr["outcome"]
    usable = ((outcome == ADDRESSED) | (outcome == INSIDE_ANATOMY)) & (addr["face"] >= 0)
    site = addr["face"][usable]
    r = addr["arclength"][usable].astype(np.float64)
    covered = mat["covered"][site]
    # Where the fiber starts inside material, B's inner face is below the skin.
    delta_in = mat["signed_delta_in"][site].astype(np.float64)
    delta_out = mat["delta_out"][site].astype(np.float64)
    b = np.where(covered, np.maximum(delta_in - r, r - delta_out), np.inf)

    record = {"shoe": name, "samples": int(usable.sum()),
              "unaddressed_fraction": float(1 - usable.mean())}
    for tol_mm in TOLERANCES_MM:
        tol = tol_mm / common.MILLIMETRES
        on = np.abs(b) <= tol
        inner = b < -tol
        outer = b > tol
        key = f"{tol_mm:g}mm"
        record[key] = {
            "on_shell": float(on.mean()),
            "inside_shell": float(inner.mean()),
            "outside_shell": float(outer.mean()),
            "by_region": {
                reg: {
                    "share_of_surface": float((region[site] == reg).mean()),
                    "on_shell": float(on[region[site] == reg].mean()) if (region[site] == reg).any() else None,
                }
                for reg in ("plantar", "dorsal_sides", "leg", "other")
            },
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shoes", nargs="*", default=None)
    args = parser.parse_args()
    out = common.STAGE1_OUTPUT / "base_envelope"
    out.mkdir(parents=True, exist_ok=True)
    names = args.shoes or json.loads(
        (common.PIPELINE_OUTPUT / "coordinates" / "summary.json").read_text())["names"]
    rows = [measure(n) for n in names]
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{'shoe':40s} {'on shell':>9s} {'inside':>7s} {'outside':>8s}   (1 mm)   on-shell plantar / dorsal+sides / leg")
    for row in rows:
        t = row["1mm"]
        g = t["by_region"]
        fmt = lambda v: "   -" if v is None else f"{v*100:4.0f}"
        print(f"{row['shoe'][:40]:40s} {t['on_shell']*100:8.1f}% {t['inside_shell']*100:6.1f}% "
              f"{t['outside_shell']*100:7.1f}%            {fmt(g['plantar']['on_shell'])} / "
              f"{fmt(g['dorsal_sides']['on_shell'])} / {fmt(g['leg']['on_shell'])}")
    for tol in TOLERANCES_MM:
        key = f"{tol:g}mm"
        print(f"median over shoes at {key}: on shell {np.median([r[key]['on_shell'] for r in rows])*100:.1f}%, "
              f"inside {np.median([r[key]['inside_shell'] for r in rows])*100:.1f}%, "
              f"outside {np.median([r[key]['outside_shell'] for r in rows])*100:.1f}%")


if __name__ == "__main__":
    main()
