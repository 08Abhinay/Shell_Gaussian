"""The ring test with the free space the ring really observes.

``train``'s ring test told the code only that observed-empty samples have no
surface within 1 mm. The cross-sections showed that is far too weak: every
model kept a phantom closed upper over a sandal's dorsum at 1-3 mm, in space
the ring sees straight through between the straps.

A ring constrains more than that. Every ray that reaches a surface passes
through empty space first. Carving a grid with every camera's depth buffer
gives the observed-empty region, and a point in it is at least as far from any
surface as it is from the edge of that region: every surface, hidden ones
included, lies outside it. So each observed-empty sample gets a lower bound

    tau(x) = distance(x, not observed empty) - half a voxel diagonal,

computed from what the ring shows alone, with no ground truth. The bound is
checked against the true distance and the violation rate is reported.

The code is then fitted to visible surface samples (distance 0) and to
tau (hinge: the predicted distance may not fall below it). Everything else
is as in ``train``. No model is retrained; only the held-out shoes' codes are
re-fitted.

    python -m anatomical_coordinates.stage1.ringfit --splits sandals --seed 0
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy import ndimage

from . import common
from .base_envelope import region_of
from .dataset import SURFACE, VOLUME, _object_centric
from ..coordinate_mapping.address import ADDRESSED, INSIDE_ANATOMY
from .elevation import GROUPS, _groups, _metrics, _predict_all
from .geometry import sample_surface
from .ring import Ring
from .train import (CLAMP_MM, SPLITS, VARIANTS, Normalizer, RESIDUAL_SPARSITY,
                    _to_device, build_model, load, predict)

MM = common.MILLIMETRES
VOXEL_MM = 2.0


def carved_bound(name: str, elevation: float, device, dense: int = 0,
                 seed: int = 0, inside_box: bool = False
                 ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """tau for every sample of ``name`` (NaN where not observed empty), in mm.

    With ``dense`` > 0, also returns that many points drawn from the carved
    empty voxels themselves, each with its own tau. The stored samples cover
    open space only sparsely (they concentrate near the true surface), so a
    code fitted to them alone can keep surfaces the ring saw straight through
    - the cross-sections showed a phantom closed upper over a sandal. The
    carved region is exactly what a ring observes, so all of it is evidence.

    ``inside_box`` draws the dense points only inside the shoe's own bounding
    box. Drawn over all observed-empty space, most land far outside the shoe,
    where emptiness is trivial, and only a few hundred fall where the prior
    is actually wrong - above a sandal's footbed, between its straps.
    """

    mesh = common.shoe_mesh(name)
    data = np.load(common.STAGE1_OUTPUT / "dataset" / f"{name}.npz")
    vertices = mesh.vertices
    splats, _, _ = sample_surface(vertices, mesh.faces, 6_000_000, seed=1)
    centre = 0.5 * (vertices.min(0) + vertices.max(0))
    radius = 3.0 * float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    ring = Ring(splats, centre, radius, elevation, device=device)

    step = VOXEL_MM / MM
    lower = vertices.min(0) - 20 / MM
    upper = vertices.max(0) + 20 / MM
    axes = [np.arange(lower[i], upper[i] + step, step) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    free = np.zeros(len(grid), dtype=bool)
    for begin in range(0, len(grid), 2_000_000):
        free[begin:begin + 2_000_000] = ring.free(grid[begin:begin + 2_000_000])
    free = free.reshape([len(a) for a in axes])
    # Distance, in mm, from each observed-empty voxel to the nearest voxel
    # the ring could not see empty.
    edt = ndimage.distance_transform_edt(free, sampling=VOXEL_MM)

    points = data["points"].astype(np.float64)
    index = np.clip(np.round((points - lower) / step).astype(np.int64), 0,
                    np.array([len(a) - 1 for a in axes]))
    bound = edt[index[:, 0], index[:, 1], index[:, 2]] - 0.5 * np.sqrt(3) * VOXEL_MM
    observed = data[f"free{int(elevation)}"] & (data["kind"] != SURFACE)
    tau = np.where(observed, np.clip(bound, 1.0, CLAMP_MM), np.nan)
    truth = data["udf"] * MM
    ok = np.isfinite(tau)
    check = {
        "observed_samples": int(ok.sum()),
        "tau_median_mm": float(np.median(tau[ok])),
        "violations_over_0.5mm": float((tau[ok] > truth[ok] + 0.5).mean()),
    }
    extra = np.zeros((0, 3))
    extra_tau = np.zeros(0)
    if dense > 0:
        voxel_bound = edt.ravel() - 0.5 * np.sqrt(3) * VOXEL_MM
        usable = free.ravel() & (voxel_bound >= 1.0)
        if inside_box:
            usable &= np.all((grid >= vertices.min(0)) & (grid <= vertices.max(0)), axis=1)
        candidates = np.nonzero(usable)[0]
        pick = np.random.default_rng(seed).choice(candidates, size=min(dense, len(candidates)), replace=False)
        extra = grid[pick]
        extra_tau = np.clip(voxel_bound[pick], 1.0, CLAMP_MM)
        from .geometry import DistanceField
        true_extra = DistanceField(vertices, mesh.faces)(extra) * MM
        check["dense_points"] = int(len(pick))
        check["dense_violations_over_0.5mm"] = float((extra_tau > true_extra + 0.5).mean())
    return tau.astype(np.float32), check, extra, extra_tau.astype(np.float32)


def dense_inputs(name: str, points: np.ndarray, tau: np.ndarray, semantics, tracer, flows) -> dict:
    """Every variant's coordinates for the dense empty-space points.

    All variants get the same points: those that also have an anatomical
    address, so B is never shown evidence the others lack.
    """

    book = common.address_book(semantics, tracer, flows, name)
    address = book.query(points, exact=True, verify=False, step=0.002, max_steps=1500)
    valid = ((address.outcome == ADDRESSED) | (address.outcome == INSIDE_ANATOMY)) & (address.face >= 0)
    pts = points[valid]
    b_part = np.concatenate((address.surface_point(tracer)[valid],
                             address.arclength[valid][:, None]), axis=1)
    a_part = _object_centric(name, pts, common.shoe_mesh(name).vertices)
    return {
        "A": a_part, "Aplus": pts, "B": b_part, "Bshell": b_part,
        "AB": np.concatenate((a_part, b_part), axis=1),
        "r_mm": address.arclength[valid] * MM, "tau": tau[valid],
    }


def dense_tensors(inputs: dict, variant: str, norm, device) -> dict:
    return {
        "x": norm(torch.as_tensor(inputs[variant], dtype=torch.float32, device=device)),
        "r_mm": torch.as_tensor(inputs["r_mm"], dtype=torch.float32, device=device),
        "tau": torch.as_tensor(inputs["tau"], dtype=torch.float32, device=device),
    }


def fit_code(model, shoe: dict, tau: torch.Tensor, device, steps: int = 1500,
             batch: int = 16384, latent: int = 64, seed: int = 0,
             extra: dict | None = None) -> torch.Tensor:
    """``extra``: optional dense empty-space points {"x", "r_mm", "tau"}."""

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    train = ~shoe["held"]
    exact = torch.nonzero(train & (shoe["kind"] == SURFACE) & (shoe["region"] == 0)).squeeze(1)
    free = torch.nonzero(train & torch.isfinite(tau)).squeeze(1)
    pool_x, pool_r, pool_tau = shoe["x"][free], shoe["r_mm"][free], tau[free]
    if extra is not None and len(extra["x"]):
        pool_x = torch.cat((pool_x, extra["x"]))
        pool_r = torch.cat((pool_r, extra["r_mm"]))
        pool_tau = torch.cat((pool_tau, extra["tau"]))
    code = torch.zeros(1, latent, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([code], lr=5e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        pick = exact[torch.randint(len(exact), (batch,), device=device, generator=generator)]
        pred, residual = predict(model, shoe["x"][pick], code.expand(batch, -1), shoe["r_mm"][pick])
        loss = (pred.clamp(max=CLAMP_MM) - shoe["target"][pick]).abs().mean()
        if residual is not None:
            loss = loss + RESIDUAL_SPARSITY * residual.abs().mean()
        fp = torch.randint(len(pool_x), (batch,), device=device, generator=generator)
        fpred, _ = predict(model, pool_x[fp], code.expand(batch, -1), pool_r[fp])
        loss = loss + torch.relu(pool_tau[fp] - fpred).mean()
        loss = loss + 1e-4 * code.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return code.detach()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=[s for s in SPLITS if s != "all"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--dense", type=int, default=0,
                        help="also constrain the code with this many carved empty-space points per shoe")
    parser.add_argument("--inside-box", action="store_true",
                        help="draw the dense points only inside the shoe's bounding box")
    args = parser.parse_args()
    device = torch.device("cuda")
    semantics, tracer, _ = common.load_canonical(device=device)
    flows = common.load_flows(device=device) if args.dense else None
    everyone = json.loads((common.PIPELINE_OUTPUT / "coordinates" / "summary.json").read_text())["names"]
    reference = common.material(everyone[0])
    plantar_site = region_of(reference["site_point"], reference["site_label"],
                             [str(x) for x in reference["label_names"]]) == "plantar"

    results, checks = {}, {}
    for split in args.splits:
        for shoe_name in SPLITS[split]:
            tau_np, checks[shoe_name], extra_pts, extra_tau = carved_bound(
                shoe_name, args.elevation, device, dense=args.dense, inside_box=args.inside_box)
            raw_b = load(shoe_name, "B")
            site, _, _ = tracer.project_to_surface(raw_b["x"][:, :3].astype(np.float64))
            groups = _groups(raw_b, plantar_site[site])
            keep = np.load(common.STAGE1_OUTPUT / "dataset" / f"{shoe_name}.npz")["b_valid"]
            tau = torch.as_tensor(tau_np[keep], device=device)
            extra_inputs = None
            if args.dense:
                extra_inputs = dense_inputs(shoe_name, extra_pts, extra_tau, semantics, tracer, flows)
                checks[shoe_name]["dense_points_addressed"] = int(len(extra_inputs["tau"]))
            for variant in args.variants:
                for seed in args.seeds:
                    path = common.STAGE1_OUTPUT / "runs" / split / f"{variant}_seed{seed}.pt"
                    if not path.is_file():
                        continue
                    saved = torch.load(path, map_location=device, weights_only=False)
                    norm = Normalizer([load(n, variant) for n in saved["names"]])
                    raw = load(shoe_name, variant)
                    model = build_model(variant, raw["x"].shape[1]).to(device)
                    model.load_state_dict(saved["model"])
                    shoe = _to_device(raw, norm, device)
                    extra = (dense_tensors(extra_inputs, variant, norm, device)
                             if extra_inputs is not None else None)
                    code = fit_code(model, shoe, tau, device, seed=seed, extra=extra)
                    pred = _predict_all(model, shoe, code)
                    metrics = _metrics(pred, raw, groups)
                    empty = raw["held"] & (raw["kind"] == VOLUME) & (raw["target"] > 5.0)
                    metrics["empty_space_mm"] = float(np.abs(pred[empty] - raw["target"][empty]).mean())
                    results.setdefault(variant, {}).setdefault(f"{shoe_name}@seed{seed}", metrics)
            print(f"{split}/{shoe_name}: tau check {checks[shoe_name]}", flush=True)

    dense = f"_dense{args.dense}{'box' if args.inside_box else ''}" if args.dense else ""
    out = common.STAGE1_OUTPUT / "runs" / (
        f"ringfit_elev{int(args.elevation)}{dense}_{'_'.join(args.splits)}_{'_'.join(args.variants)}.json")
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite it")
    out.write_text(json.dumps({"results": results, "tau_checks": checks}, indent=2) + "\n")


def table(elevation: int = 0, dense: int = 0, box: bool = False) -> str:
    merged = {}
    tag = f"dense{dense}{'box' if box else ''}"
    pattern = f"ringfit_elev{elevation}_{tag}_*.json" if dense else f"ringfit_elev{elevation}_*.json"
    for path in sorted((common.STAGE1_OUTPUT / "runs").glob(pattern)):
        if not dense and "_dense" in path.name:
            continue
        for variant, rows in json.loads(path.read_text())["results"].items():
            merged.setdefault(variant, {}).update(rows)
    lines = [f"{'variant':8s} | " + " | ".join(f"{g:>16s}" for g in GROUPS) + " | false | empty-space err"]
    for variant in VARIANTS:
        rows = list(merged.get(variant, {}).values())
        if not rows:
            continue
        cells = []
        for g in GROUPS:
            surf = [r[g]["surface_mm"] for r in rows if r[g]["surface_mm"] is not None]
            rec = [r[g]["recall_2mm"] for r in rows if r[g]["recall_2mm"] is not None]
            cells.append(f"{np.mean(surf):5.2f}mm {np.mean(rec)*100:3.0f}%" if surf else f"{'-':>16s}")
        false = np.mean([r["false_surface"] for r in rows if r["false_surface"] is not None])
        empty = np.mean([r["empty_space_mm"] for r in rows])
        lines.append(f"{variant:8s} | " + " | ".join(f"{c:>16s}" for c in cells)
                     + f" | {false*100:4.1f}% | {empty:5.2f}mm   ({len(rows)} shoe-seeds)")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
