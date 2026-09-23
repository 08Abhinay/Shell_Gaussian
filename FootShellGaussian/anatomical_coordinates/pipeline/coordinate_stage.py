"""Build the shared volumetric coordinate map for every shoe, in one batch.

Replaces the tetrahedral cage deformation. The map is the flow of a velocity
field, so it is invertible by construction for any amount of deformation - there
is no mesh whose elements could invert, and therefore no need to constrain how
different a foot may be from the canonical one.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from foot_prior.mesh import load_triangle_mesh
from ..coordinate_mapping.batch import fit_batch, integrate_batched
from ..coordinate_mapping.sampling import shell_samples
from .stages import BY_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shoes", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--smooth", type=float, default=1e-3)
    parser.add_argument("--probes", type=int, default=4000)
    args = parser.parse_args()

    joined = args.root / BY_KEY["join"].directory
    out = args.root / BY_KEY["coordinates"].directory
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference = load_triangle_mesh(joined / "reference" / "neutral_foot_lower_leg.ply")
    names = [n for n in args.shoes if (joined / n / "foot_lower_leg.ply").is_file()]
    instances = np.stack([
        load_triangle_mesh(joined / n / "foot_lower_leg.ply").vertices for n in names
    ])
    fields, result = fit_batch(
        names, reference.vertices, instances, device,
        iterations=args.iterations, steps=args.steps, weight_smooth=args.smooth,
    )
    print(f"fitted {len(names)} coordinate maps in {result.seconds:.1f}s", flush=True)

    # Injectivity audit, concentrated in the shell where footwear material sits.
    probe = shell_samples(reference.vertices, reference.faces, args.probes, (-0.02, 0.25), seed=11)
    points = torch.as_tensor(probe, dtype=torch.float32, device=device)[None]
    points = points.expand(len(names), -1, -1).contiguous()
    h = 2e-4
    with torch.no_grad():
        forward = integrate_batched(fields, points, args.steps, 1.0)
        columns = []
        for axis in range(3):
            offset = torch.zeros(3, device=device); offset[axis] = h
            columns.append(
                (integrate_batched(fields, points + offset, args.steps, 1.0)
                 - integrate_batched(fields, points - offset, args.steps, 1.0)) / (2 * h)
            )
        jacobian = torch.stack(columns, dim=-1)
        determinant = torch.linalg.det(jacobian)
        singular = torch.linalg.svdvals(jacobian)
        round_trip = (
            integrate_batched(fields, forward, args.steps, -1.0) - points
        ).norm(dim=-1).max(dim=1).values * 262.5

    records = {}
    for index, name in enumerate(names):
        records[name] = {
            "boundary_rms_mm": result.vertex_rms_mm[index],
            "boundary_max_mm": result.vertex_max_mm[index],
            "jacobian_minimum": float(determinant[index].min()),
            "jacobian_median": float(determinant[index].median()),
            "inverted_samples": int((determinant[index] <= 0).sum()),
            "distortion_ratio": float(
                singular[index, :, 0].max() / singular[index, :, 2].min().clamp(min=1e-9)
            ),
            "round_trip_max_mm": float(round_trip[index]),
        }
    torch.save(fields.state_dict(), out / "coordinate_fields.pt")
    (out / "summary.json").write_text(json.dumps({
        "names": names, "shoes": records, "seconds": result.seconds,
        "integration_steps": args.steps, "iterations": args.iterations,
        "probes_per_shoe": args.probes,
        "inverted_total": int(sum(r["inverted_samples"] for r in records.values())),
    }, indent=2) + "\n")
    bad = sum(r["inverted_samples"] for r in records.values())
    print(f"injectivity: {bad} inverted of {len(names)*args.probes} samples; "
          f"worst determinant {min(r['jacobian_minimum'] for r in records.values()):.4f}")


if __name__ == "__main__":
    main()
