"""Check that one address means the same anatomical place in every shoe.

The address stage verifies each shoe on its own: an address, put back into the
shoe it came from, returns to its point. That is necessary but not sufficient.
The representation claims something stronger - that an address is *shared*, so
the same (u, v, r) names the same anatomical place in all 27 shoes at once.

This sends one set of addresses through every shoe and reads them back. It is
independent of the address stage: nothing here reuses that stage's results, and
the addresses are generated from the canonical anatomy rather than from any
shoe's samples.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from foot_prior.mesh import load_triangle_mesh
from ..coordinate_mapping.batch import BatchedVelocityFields, integrate_batched
from ..coordinate_mapping.lookup import CanonicalSemantics
from ..coordinate_mapping import address as addressing
from ..coordinate_mapping import cross_shoe
from .stages import BY_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--addresses", type=int, default=256)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trace-step", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    root = args.root
    joined = root / BY_KEY["join"].directory
    coords = root / BY_KEY["coordinates"].directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    semantics = CanonicalSemantics(
        root / "canonical" / "volume" / "canonical_volume.npz",
        root / "canonical" / "semantic_field" / "semantic_field.npz",
    )
    directions = np.load(
        root / "canonical" / "semantic_field" / "fiber_field.npz"
    )["directions"]
    tracer = addressing.FiberTracer(semantics, directions, device=device)
    table = addressing.CanonicalCorrespondence.load(root / "canonical" / "correspondence.npz")

    names = json.loads((coords / "summary.json").read_text())["names"]
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

    def maps(index: int):
        def flow(block: np.ndarray, direction: float) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.as_tensor(block, dtype=torch.float32, device=device)
                tensor = tensor[None].expand(len(names), -1, -1).contiguous()
                return integrate_batched(
                    fields, tensor, args.steps, direction
                )[index].cpu().numpy().astype(np.float64)
        return (lambda p: flow(p, -1.0)), (lambda p: flow(p, 1.0))

    books = {}
    for index, name in enumerate(names):
        back, forward = maps(index)
        books[name] = addressing.AddressBook(
            semantics, tracer, table, back, forward
        )

    # Addresses are drawn on the foot itself, by area, from the canonical
    # anatomy - never from a shoe - so no shoe is privileged.
    generator = np.random.default_rng(args.seed)
    skin = semantics.boundary_label_names.index("foot_skin")
    candidates = np.nonzero(semantics.inner_face_labels == skin)[0]
    corners = tracer.triangles[candidates]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    picked = generator.choice(
        candidates, size=args.addresses, p=areas / areas.sum()
    )
    u = generator.random(args.addresses)
    v = generator.random(args.addresses)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    weights = np.stack((1.0 - u - v, u, v), axis=1)

    started = time.time()
    trace = {"step": args.trace_step, "max_steps": int(3.0 / args.trace_step)}
    print(f"{len(names)} shoes, {args.addresses} addresses per level\n")
    print(f"{'level r':>8s} {'returned':>9s} {'spread med':>11s} {'spread max':>11s} "
          f"{'drift med':>10s} {'drift p99':>10s} {'drift max':>10s}")
    records = []
    for level in args.levels:
        target = np.full(args.addresses, float(level))
        report = cross_shoe.check(books, tracer, picked, weights, target, **trace)
        drift = report.drift_mm[report.returned]
        spread = report.spread_mm[np.isfinite(report.spread_mm)]
        record = {
            "level": float(level),
            "returned_fraction": float(report.returned.mean()),
            "spread_median_mm": float(np.median(spread)) if spread.size else None,
            "spread_max_mm": float(spread.max()) if spread.size else None,
            "drift_median_mm": float(np.median(drift)) if drift.size else None,
            "drift_p99_mm": float(np.percentile(drift, 99)) if drift.size else None,
            "drift_max_mm": float(drift.max()) if drift.size else None,
        }
        records.append(record)
        print(f"{level:8.2f} {report.returned.mean()*100:8.2f}% "
              f"{record['spread_median_mm']:11.2f} {record['spread_max_mm']:11.2f} "
              f"{record['drift_median_mm']:10.4f} {record['drift_p99_mm']:10.4f} "
              f"{record['drift_max_mm']:10.3f}", flush=True)

    out = root / BY_KEY["address"].directory / "cross_shoe.json"
    out.write_text(json.dumps({
        "shoes": names, "addresses": args.addresses,
        "levels": records, "seconds": time.time() - started,
        "trace_step": args.trace_step,
    }, indent=2) + "\n")
    print(f"\nspread is how far apart the same address lands between shoes - "
          f"large is correct, the shoes differ.")
    print(f"drift is how far the address moves on the foot after a round trip "
          f"through a shoe - small is correct.")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
