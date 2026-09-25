"""The anatomy-only cavity: how well does "typical clearance" predict it?

The report's central argument (section 2.2) is that the regions a ring never
sees are the ones the foot determines most. The cleanest test of that needs
no learning at all. Take the shoes a model would train on, and at every
anatomical site take the median clearance - the median delta_in across the
shoes that cover it. For a shoe outside that set, place the cavity wall at
that distance along its fibers, and measure how far its true interior surface
lies from the wall.

This is mu_clr of representation.pdf 1.2 in its simplest form: a per-site
median, with no conditioning on the foot and no residual. It uses nothing from
the test shoe except its fitted foot (through the fibers), so it is the
anatomical prior with the latent switched off.

Scored on the same held-out interior samples as ``train``'s ring test, with the
same numbers - surface_mm and recall - so the two can be compared side by side.
The distance is measured along the fiber, which is never shorter than the true
distance to the predicted wall, so this errs against the prior.

    python -m anatomical_coordinates.stage1.cavity_prior
"""

from __future__ import annotations

import json

import numpy as np

from . import common
from .base_envelope import region_of
from .dataset import INTERIOR, SURFACE
from .train import SPLITS, load

MM = common.MILLIMETRES


def clearance(names: list[str]) -> np.ndarray:
    """Median delta_in per site over ``names``, in mm; NaN where none covers."""

    stack = []
    for n in names:
        mat = common.material(n)
        stack.append(np.where(mat["covered"], np.maximum(mat["signed_delta_in"], 0.0) * MM, np.nan))
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(stack), axis=0)


def score(name: str, prior: np.ndarray, tracer, plantar_site: np.ndarray) -> dict:
    shoe = load(name, "B")
    pick = shoe["held"] & (shoe["kind"] == SURFACE) & (shoe["region"] == INTERIOR)
    s = shoe["x"][pick, :3].astype(np.float64)
    r = shoe["r_mm"][pick].astype(np.float64)
    # The site each sample's fiber lands in: the canonical triangle under s.
    site, _, _ = tracer.project_to_surface(s)
    wall = prior[site]
    ok = np.isfinite(wall)
    gap = np.abs(r - wall)

    def summary(mask):
        mask = mask & ok
        return {
            "samples": int(mask.sum()),
            "surface_mm": float(gap[mask].mean()) if mask.any() else None,
            "recall_1mm": float((gap[mask] <= 1.0).mean()) if mask.any() else None,
            "recall_2mm": float((gap[mask] <= 2.0).mean()) if mask.any() else None,
        }

    # The footbed lies under the foot, where the exterior says least about it;
    # the lining of the upper sits just behind a visible wall.
    plantar = plantar_site[site]
    return {"with_prior": float(ok.mean()), "all": summary(np.ones_like(ok)),
            "plantar": summary(plantar), "dorsal_sides": summary(~plantar)}


def main() -> None:
    _, tracer, _ = common.load_canonical()
    everyone = json.loads((common.PIPELINE_OUTPUT / "coordinates" / "summary.json").read_text())["names"]
    reference = common.material(everyone[0])
    names = [str(x) for x in reference["label_names"]]
    plantar_site = region_of(reference["site_point"], reference["site_label"], names) == "plantar"
    results = {}
    for split, held_out in SPLITS.items():
        if not held_out:
            continue
        prior = clearance([n for n in everyone if n not in held_out])
        results[split] = {n: score(n, prior, tracer, plantar_site) for n in held_out}
    out = common.STAGE1_OUTPUT / "runs" / "cavity_prior.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{'split':8s} {'shoe':38s} {'all':>14s} {'plantar (footbed)':>18s} {'dorsal/sides':>14s}")
    fmt = lambda r: "      -" if r["surface_mm"] is None else f"{r['surface_mm']:5.2f}mm {r['recall_2mm']*100:3.0f}%"
    for split, shoes in results.items():
        for n, r in shoes.items():
            print(f"{split:8s} {n[:38]:38s} {fmt(r['all']):>14s} {fmt(r['plantar']):>18s} {fmt(r['dorsal_sides']):>14s}")


if __name__ == "__main__":
    main()
