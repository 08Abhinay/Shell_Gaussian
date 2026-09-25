"""The report's controllable ablation, and a finer split of the blind zones.

Real StockX captures are fixed near 0 degrees; the CAD set can be re-observed
from 15 and 30 degrees (report, section 6.3). For every held-out shoe and
saved model this re-fits the shoe's code to what each ring observes and
scores it on the surfaces a 0-degree ring never sees, split finer than
``train`` does:

    interior_plantar   the footbed: under the foot, where the exterior
                       constrains least
    interior_dorsal    the lining of the upper, just behind a visible wall
    sole, top          as in ``dataset``

The split is the one ``cavity_prior`` reports for the anatomy-only clearance
prior, so the two can be read side by side.

    python -m anatomical_coordinates.stage1.elevation --splits sandals crocs
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import common
from .base_envelope import region_of
from .dataset import ELEVATIONS, INTERIOR, NEAR, SOLE, SURFACE, TOP, VISIBLE, VOLUME
from .train import (CLAMP_MM, SPLITS, VARIANTS, Normalizer, _to_device, build_model,
                    fit_code, load, predict)

GROUPS = ("visible", "interior_plantar", "interior_dorsal", "sole", "top")


def _groups(shoe: dict, plantar: np.ndarray) -> dict[str, np.ndarray]:
    region = shoe["region"]
    return {
        "visible": region == VISIBLE,
        "interior_plantar": (region == INTERIOR) & plantar,
        "interior_dorsal": (region == INTERIOR) & ~plantar,
        "sole": region == SOLE,
        "top": region == TOP,
    }


@torch.no_grad()
def _predict_all(model, shoe: dict, code: torch.Tensor) -> np.ndarray:
    chunks = []
    for i in range(0, len(shoe["x"]), 65536):
        x = shoe["x"][i:i + 65536]
        chunks.append(predict(model, x, code.expand(len(x), -1), shoe["r_mm"][i:i + 65536])[0])
    return torch.cat(chunks).clamp(0, CLAMP_MM).cpu().numpy()


def _metrics(pred, raw: dict, groups: dict) -> dict:
    held, kind, target = raw["held"], raw["kind"], raw["target"]
    out = {}
    for name, mask in groups.items():
        surf = held & (kind == SURFACE) & mask
        near = held & (kind == NEAR) & mask
        out[name] = {
            "samples": int(surf.sum()),
            "surface_mm": float(pred[surf].mean()) if surf.any() else None,
            "recall_2mm": float((pred[surf] <= 2.0).mean()) if surf.any() else None,
            "field_mm": float(np.abs(pred[near] - target[near]).mean()) if near.any() else None,
        }
    far = held & (kind == VOLUME) & (target > 5.0)
    out["false_surface"] = float((pred[far] < 1.0).mean()) if far.any() else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=[s for s in SPLITS if s != "all"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    device = torch.device("cuda")
    _, tracer, _ = common.load_canonical(device=device)
    everyone = json.loads((common.PIPELINE_OUTPUT / "coordinates" / "summary.json").read_text())["names"]
    reference = common.material(everyone[0])
    plantar_site = region_of(reference["site_point"], reference["site_label"],
                             [str(x) for x in reference["label_names"]]) == "plantar"

    results = {}
    for split in args.splits:
        for shoe_name in SPLITS[split]:
            raw_b = load(shoe_name, "B")
            site, _, _ = tracer.project_to_surface(raw_b["x"][:, :3].astype(np.float64))
            groups = _groups(raw_b, plantar_site[site])
            data = np.load(common.STAGE1_OUTPUT / "dataset" / f"{shoe_name}.npz")
            keep = data["b_valid"]
            for variant in args.variants:
                path = common.STAGE1_OUTPUT / "runs" / split / f"{variant}_seed{args.seed}.pt"
                if not path.is_file():
                    continue
                saved = torch.load(path, map_location=device, weights_only=False)
                norm = Normalizer([load(n, variant) for n in saved["names"]])
                raw = load(shoe_name, variant)
                model = build_model(variant, raw["x"].shape[1]).to(device)
                model.load_state_dict(saved["model"])
                shoe = _to_device(raw, norm, device)
                for elevation in ELEVATIONS:
                    visible = np.zeros(len(keep), dtype=bool)
                    visible[data["kind"] == SURFACE] = data[f"seen{elevation}"] > 0
                    evidence = dict(shoe)
                    # fit_code reads "visible surface" as region 0 and the
                    # observed-empty samples from free0.
                    evidence["region"] = torch.as_tensor(
                        np.where(visible[keep], 0, 1).astype(np.int8), device=device)
                    evidence["free0"] = torch.as_tensor(data[f"free{elevation}"][keep], device=device)
                    code = fit_code(model, evidence, "ring", device, seed=args.seed)
                    # Always scored on the 0-degree regions: the same surfaces a
                    # StockX ring never sees, whatever this ring saw.
                    results.setdefault(split, {}).setdefault(variant, {}).setdefault(
                        shoe_name, {})[str(elevation)] = _metrics(
                            _predict_all(model, shoe, code), raw, groups)
            print(f"{split}/{shoe_name} done", flush=True)

    # Named by splits *and* variants, so a run over a subset of variants never
    # overwrites the results of another; ``table`` merges them all.
    variants = "" if list(args.variants) == list(VARIANTS[:4]) else "_" + "_".join(args.variants)
    out = common.STAGE1_OUTPUT / "runs" / f"elevation_{'_'.join(args.splits)}{variants}_seed{args.seed}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite it")
    out.write_text(json.dumps(results, indent=2) + "\n")


def table(paths=None) -> str:
    """Mean over every held-out shoe in the saved elevation results."""

    paths = paths or sorted((common.STAGE1_OUTPUT / "runs").glob("elevation_*_seed*.json"))
    merged = {}
    for path in paths:
        for split, variants in json.loads(path.read_text()).items():
            for variant, shoes in variants.items():
                for shoe, by_elev in shoes.items():
                    merged.setdefault(variant, {}).setdefault(shoe, {}).update(by_elev)
    lines = [f"{'variant':8s} {'elev':>4s} | " + " | ".join(f"{g:>16s}" for g in GROUPS) + " | false"]
    for variant in VARIANTS:
        shoes = merged.get(variant, {})
        for elevation in map(str, ELEVATIONS):
            rows = [s[elevation] for s in shoes.values() if elevation in s]
            if not rows:
                continue
            cells = []
            for g in GROUPS:
                surf = [r[g]["surface_mm"] for r in rows if r[g]["surface_mm"] is not None]
                rec = [r[g]["recall_2mm"] for r in rows if r[g]["recall_2mm"] is not None]
                cells.append(f"{np.mean(surf):5.2f}mm {np.mean(rec)*100:3.0f}%" if surf else f"{'-':>16s}")
            false = np.mean([r["false_surface"] for r in rows if r["false_surface"] is not None])
            lines.append(f"{variant:8s} {elevation:>4s} | " + " | ".join(f"{c:>16s}" for c in cells)
                         + f" | {false*100:4.1f}%   ({len(rows)} shoes)")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
