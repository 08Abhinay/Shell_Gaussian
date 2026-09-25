"""A picture of the blind-zone test: one slice through a held-out shoe.

For a shoe the models never saw, each variant's code is fitted to what a
0-degree ring observes, and the predicted distance is drawn on the shoe's
mid-sagittal plane next to the true distance. The interior, the footbed and
the sole all cross that plane, so it shows directly what each model puts where
the camera never looked.

    python -m anatomical_coordinates.stage1.slices --split sandals --shoe sandal_1
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import common
from .dataset import _object_centric
from .geometry import DistanceField
from .train import (CLAMP_MM, SPLITS, VARIANTS, Normalizer, _to_device, build_model,
                    fit_code, load, predict)
from ..coordinate_mapping.address import ADDRESSED, INSIDE_ANATOMY

MM = common.MILLIMETRES
LABELS = {"A": "A  object-centric", "Aplus": "A+  aligned", "B": "B  anatomical",
          "Bshell": "B + shell", "AB": "A and B together"}


def slice_grid(vertices: np.ndarray, spacing_mm: float = 0.75):
    lo, hi = vertices.min(0) - 10 / MM, vertices.max(0) + 10 / MM
    xs = np.arange(lo[0], hi[0], spacing_mm / MM)
    ys = np.arange(lo[1], hi[1], spacing_mm / MM)
    z = 0.5 * (vertices[:, 2].min() + vertices[:, 2].max())
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack((gx.ravel(), gy.ravel(), np.full(gx.size, z)), axis=1)
    return points, gx.shape, (xs[0], xs[-1], ys[0], ys[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=[s for s in SPLITS if s != "all"])
    parser.add_argument("--shoe", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weak-free-space", action="store_true",
                        help="fit with train's 1 mm free-space hinge instead of the carved bound")
    parser.add_argument("--dense", type=int, default=60000,
                        help="carved empty-space points added to the fit (0 = samples only)")
    parser.add_argument("--inside-box", action="store_true",
                        help="draw the dense points only inside the shoe's bounding box")
    args = parser.parse_args()
    device = torch.device("cuda")

    shoe_mesh = common.shoe_mesh(args.shoe)
    points, shape, extent = slice_grid(shoe_mesh.vertices)
    truth = np.minimum(DistanceField(shoe_mesh.vertices, shoe_mesh.faces)(points) * MM, CLAMP_MM)

    flows = common.load_flows(device=device)
    semantics, tracer, _ = common.load_canonical(device=device)
    book = common.address_book(semantics, tracer, flows, args.shoe)
    address = book.query(points, exact=True, verify=False, step=0.002, max_steps=1500)
    usable = ((address.outcome == ADDRESSED) | (address.outcome == INSIDE_ANATOMY)) & (address.face >= 0)
    grid_inputs = {
        "A": _object_centric(args.shoe, points, shoe_mesh.vertices),
        "Aplus": points,
        "B": np.concatenate((address.surface_point(tracer), address.arclength[:, None]), axis=1),
    }
    grid_inputs["Bshell"] = grid_inputs["B"]
    grid_inputs["AB"] = np.concatenate((grid_inputs["A"], grid_inputs["B"]), axis=1)
    r_mm = np.nan_to_num(address.arclength) * MM
    dense = None
    if not args.weak_free_space:
        from .ringfit import carved_bound, dense_inputs, dense_tensors, fit_code as fit_carved
        tau_all, _, extra_pts, extra_tau = carved_bound(args.shoe, 0.0, device, dense=args.dense,
                                                        inside_box=args.inside_box)
        keep = np.load(common.STAGE1_OUTPUT / "dataset" / f"{args.shoe}.npz")["b_valid"]
        tau = torch.as_tensor(tau_all[keep], device=device)
        if args.dense:
            dense = dense_inputs(args.shoe, extra_pts, extra_tau, semantics, tracer, flows)

    panels = [("ground truth", truth)]
    for variant in VARIANTS:
        path = common.STAGE1_OUTPUT / "runs" / args.split / f"{variant}_seed{args.seed}.pt"
        if not path.is_file():
            continue
        saved = torch.load(path, map_location=device, weights_only=False)
        norm = Normalizer([load(n, variant) for n in saved["names"]])
        raw = load(args.shoe, variant)
        model = build_model(variant, raw["x"].shape[1]).to(device)
        model.load_state_dict(saved["model"])
        shoe = _to_device(raw, norm, device)
        if args.weak_free_space:
            code = fit_code(model, shoe, "ring", device, seed=args.seed)
        else:
            extra = dense_tensors(dense, variant, norm, device) if dense is not None else None
            code = fit_carved(model, shoe, tau, device, seed=args.seed, extra=extra)
        x = norm(torch.as_tensor(np.nan_to_num(grid_inputs[variant]), dtype=torch.float32, device=device))
        r = torch.as_tensor(r_mm, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = torch.cat([predict(model, x[i:i + 65536], code.expand(len(x[i:i + 65536]), -1),
                                      r[i:i + 65536])[0] for i in range(0, len(x), 65536)])
        pred = pred.clamp(0, CLAMP_MM).cpu().numpy()
        if "B" in variant:
            pred = np.where(usable, pred, np.nan)
        panels.append((LABELS[variant], pred))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), constrained_layout=True)
    for ax, (title, values) in zip(axes, panels):
        image = values.reshape(shape)
        ax.imshow(image, extent=(extent[0], extent[1], extent[3], extent[2]), cmap="magma_r",
                  vmin=0, vmax=CLAMP_MM)
        ax.contour(np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0]),
                   image, levels=[1.0], colors="cyan", linewidths=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    evidence = ("1 mm free-space hinge" if args.weak_free_space
                else f"carved free-space bound, {args.dense} dense empty-space points" if args.dense
                else "carved free-space bound")
    fig.suptitle(f"{args.shoe} (held out of the '{args.split}' split): code fitted to a 0-degree ring only "
                 f"({evidence}). Colour = distance to surface (0-10 mm); cyan = 1 mm contour.", fontsize=10)
    out = common.STAGE1_OUTPUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tag = "weak" if args.weak_free_space else (
        f"dense{args.dense}{'box' if args.inside_box else ''}" if args.dense else "carved")
    path = out / f"slice_{args.split}_{args.shoe}_{tag}.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
