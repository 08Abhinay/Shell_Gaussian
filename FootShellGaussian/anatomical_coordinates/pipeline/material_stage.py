"""Measure what each shoe does at every place on the foot.

Turns each shoe from a mesh into three fields over the canonical foot - is
this part covered, where does material start, where does it end - which is what
the representation's base envelope is made of.

The fibers are traced once. They live in the canonical anatomy, which is shared,
so every shoe reuses the same curves and only the carrying-through and the
intersection are per shoe. Under the older per-instance volume this would have
been 27 separate traces through 27 separate meshes.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from foot_prior.mesh import load_triangle_mesh
from ..coordinate_mapping.batch import BatchedVelocityFields, integrate_batched
from ..coordinate_mapping.lookup import CanonicalSemantics
from ..coordinate_mapping import address as addressing
from ..coordinate_mapping import material as materials
from .stages import BY_KEY

MM = materials.MILLIMETRES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shoes", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trace-step", type=float, default=0.002)
    parser.add_argument("--fiber-samples", type=int, default=400)
    parser.add_argument("--neighbours", type=int, default=64)
    args = parser.parse_args()

    root = args.root
    joined = root / BY_KEY["join"].directory
    coords = root / BY_KEY["coordinates"].directory
    addresses = root / BY_KEY["address"].directory
    out = root / BY_KEY["material"].directory
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    semantics = CanonicalSemantics(
        root / "canonical" / "volume" / "canonical_volume.npz",
        root / "canonical" / "semantic_field" / "semantic_field.npz",
    )
    directions = np.load(
        root / "canonical" / "semantic_field" / "fiber_field.npz"
    )["directions"]
    tracer = addressing.FiberTracer(semantics, directions, device=device)
    sites = materials.anatomical_sites(semantics, tracer)
    print(f"{len(sites.face)} anatomical sites, one per inner-surface triangle")

    # Traced once, in canonical space, and reused by every shoe.
    cache = root / "canonical" / "fiber_paths.npz"
    if cache.is_file():
        stored = np.load(cache)
        curves = np.asarray(stored["curves"], dtype=np.float64)
        canonical_length = np.asarray(stored["arclength"], dtype=np.float64)
        print(f"fiber paths: read from {cache.name}")
    else:
        moment = time.time()
        curves, canonical_length, _ = tracer.trace_path(
            sites.face, sites.barycentric,
            samples=args.fiber_samples, step=args.trace_step,
        )
        np.savez_compressed(
            cache, curves=curves.astype(np.float32),
            arclength=canonical_length.astype(np.float32),
        )
        reach = np.nanmax(canonical_length, axis=1)
        print(f"fiber paths: {len(sites.face)} traced in {time.time()-moment:.0f}s, "
              f"median reach {np.nanmedian(reach)*MM:.1f} mm")
    reach = np.nanmax(canonical_length, axis=1)

    summary = json.loads((coords / "summary.json").read_text())
    names = summary["names"]
    reference = load_triangle_mesh(joined / "reference" / "neutral_foot_lower_leg.ply")
    instances = np.stack([
        load_triangle_mesh(joined / n / "foot_lower_leg.ply").vertices for n in names
    ])
    lower = np.stack([np.minimum(reference.vertices.min(0), instances[i].min(0)) - 0.25
                      for i in range(len(names))])
    upper = np.stack([np.maximum(reference.vertices.max(0), instances[i].max(0)) + 0.25
                      for i in range(len(names))])
    fields = BatchedVelocityFields(
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
    ).to(device)
    fields.load_state_dict(
        torch.load(coords / "coordinate_fields.pt", map_location=device, weights_only=True)
    )
    fields.eval()

    flat = curves.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    live = flat[finite]

    def carry(index: int) -> np.ndarray:
        """The canonical fiber curves, as they sit inside one shoe."""

        moved = np.empty_like(live)
        block = 262144
        with torch.no_grad():
            for begin in range(0, len(live), block):
                piece = live[begin : begin + block]
                tensor = torch.as_tensor(piece, dtype=torch.float32, device=device)
                tensor = tensor[None].expand(len(names), -1, -1).contiguous()
                moved[begin : begin + block] = integrate_batched(
                    fields, tensor, args.steps, 1.0
                )[index].cpu().numpy()
        out_curves = np.full_like(flat, np.nan)
        out_curves[finite] = moved
        return out_curves.reshape(curves.shape)

    started = time.time()
    print(f"\n{'shoe':44s} {'covered':>8s} {'delta_in':>9s} {'wall':>8s} "
          f"{'layers':>7s} {'pierced':>8s}")
    records = []
    wanted = set(args.shoes)
    for index, name in enumerate(names):
        if name not in wanted:
            continue
        shoe = load_triangle_mesh(
            root / "inputs" / "shoe_preparation" / name / "shoe_normalized.ply"
        )
        carried = carry(index)
        site, at_r, at_mm = materials.crossings(
            carried, canonical_length, shoe.vertices, shoe.faces,
            neighbours=args.neighbours,
        )

        count = len(sites.face)
        covered = np.zeros(count, dtype=bool)
        delta_in = np.full(count, np.nan)
        delta_out = np.full(count, np.nan)
        delta_in_mm = np.full(count, np.nan)
        delta_out_mm = np.full(count, np.nan)
        layers = np.zeros(count, dtype=np.int32)
        if site.size:
            first = np.ones(len(site), dtype=bool)
            first[1:] = site[1:] != site[:-1]
            last = np.ones(len(site), dtype=bool)
            last[:-1] = site[1:] != site[:-1]
            covered[site] = True
            delta_in[site[first]] = at_r[first]
            delta_in_mm[site[first]] = at_mm[first]
            delta_out[site[last]] = at_r[last]
            delta_out_mm[site[last]] = at_mm[last]
            np.add.at(layers, site, 1)

        penetration = materials.penetration_per_site(
            addresses / name / "addresses.npz", count,
            inside_code=addressing.INSIDE_ANATOMY,
        )
        result = materials.MaterialFields(
            sites, covered, delta_in, delta_out, delta_in_mm, delta_out_mm,
            layers, penetration, reach, site, at_r, at_mm,
        )

        shoe_dir = out / name
        shoe_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shoe_dir / "material.npz",
            site_face=sites.face, site_barycentric=sites.barycentric.astype(np.float32),
            site_label=sites.label, site_point=sites.point.astype(np.float32),
            covered=covered,
            delta_in=delta_in.astype(np.float32),
            delta_out=delta_out.astype(np.float32),
            delta_in_mm=delta_in_mm.astype(np.float32),
            delta_out_mm=delta_out_mm.astype(np.float32),
            signed_delta_in=result.signed_delta_in().astype(np.float32),
            layers=layers, penetration_mm=penetration.astype(np.float32),
            fiber_reach=reach.astype(np.float32),
            crossing_site=site, crossing_r=at_r.astype(np.float32),
            crossing_mm=at_mm.astype(np.float32),
            label_names=np.array(semantics.boundary_label_names),
        )

        wall = delta_out_mm - delta_in_mm
        skin = sites.label == semantics.boundary_label_names.index("foot_skin")
        record = {
            "shoe": name, "sites": int(count),
            "covered_fraction": float(covered.mean()),
            "covered_fraction_foot": float(covered[skin].mean()),
            "delta_in_median_mm": float(np.nanmedian(delta_in_mm[covered])) if covered.any() else None,
            "wall_median_mm": float(np.nanmedian(wall[covered])) if covered.any() else None,
            "layers_median": int(np.median(layers[covered])) if covered.any() else 0,
            "layers_max": int(layers.max()),
            "multi_layer_fraction": float((layers > 2).mean()),
            "pierced_fraction": float((penetration > 0).mean()),
            "penetration_median_mm": (
                float(np.median(penetration[penetration > 0]))
                if (penetration > 0).any() else None
            ),
            "penetration_max_mm": float(penetration.max()),
        }
        records.append(record)
        (shoe_dir / "material.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name[:44]:44s} {record['covered_fraction']*100:7.2f}% "
              f"{(record['delta_in_median_mm'] or float('nan')):8.2f} "
              f"{(record['wall_median_mm'] or float('nan')):7.2f} "
              f"{record['layers_median']:7d} {record['pierced_fraction']*100:7.2f}%",
              flush=True)

    merged = [
        json.loads(path.read_text())
        for path in sorted(out.glob("*/material.json"))
    ]
    (out / "summary.json").write_text(json.dumps({
        "shoes": merged, "sites": int(len(sites.face)),
        "seconds": time.time() - started,
        "trace_step": args.trace_step, "fiber_samples": args.fiber_samples,
    }, indent=2) + "\n")
    cov = np.array([r["covered_fraction"] for r in merged])
    print(f"\n{len(merged)} shoes in {time.time()-started:.1f}s   "
          f"covered min {cov.min()*100:.1f}% median {np.median(cov)*100:.1f}% "
          f"max {cov.max()*100:.1f}%")


if __name__ == "__main__":
    main()
