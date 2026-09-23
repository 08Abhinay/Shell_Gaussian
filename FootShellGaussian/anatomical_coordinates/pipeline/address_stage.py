"""Give every footwear surface point an anatomical address.

An address is a position on the canonical foot plus outward progress through the
surrounding volume, so the same address means the same anatomical place on every
shoe. Points that land inside the foot itself have no address by definition -
that region is not part of the footwear domain - and are reported separately
rather than being forced to a nearest value.

The address is written out per point, not just summarised. A coverage fraction
says how much of a shoe could be addressed; it does not say where anything is,
and the next stage of the work needs the addresses themselves - you cannot fit
a field over (u, v) without being able to hand it (u, v).

Every address is verified by sending it back through the inverse map and
measuring how far it lands from the point it came from, so the fraction that
is reported as addressed is the fraction that demonstrably round trips.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from foot_prior.mesh import load_triangle_mesh
from ..coordinate_mapping.batch import BatchedVelocityFields, integrate_batched
from ..coordinate_mapping.sampling import area_samples
from ..coordinate_mapping.lookup import CanonicalSemantics
from ..coordinate_mapping import address as addressing
from ..coordinate_mapping import fibers as fiber_tools
from ..coordinate_mapping import figures
from .stages import BY_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shoes", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fibers", type=int, default=1536)
    parser.add_argument("--fiber-samples", type=int, default=96)
    parser.add_argument("--trace-step", type=float, default=0.002)
    parser.add_argument("--tolerance-mm", type=float, default=0.1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    root = args.root
    joined = root / BY_KEY["join"].directory
    coords = root / BY_KEY["coordinates"].directory
    out = root / BY_KEY["address"].directory
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    semantics = CanonicalSemantics(
        root / "canonical" / "volume" / "canonical_volume.npz",
        root / "canonical" / "semantic_field" / "semantic_field.npz",
    )
    print(f"canonical index: {len(semantics.tetrahedra)} cells, built once for all shoes")

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
    fields.load_state_dict(torch.load(coords / "coordinate_fields.pt", map_location=device))
    fields.eval()

    # Fibers are traced once, in the canonical anatomy. Every shoe then gets
    # them by carrying the curves through its own map, which is an evaluation
    # rather than another integration.
    canonical_fibers = None
    directions_path = root / "canonical" / "semantic_field" / "fiber_field.npz"
    if directions_path.is_file():
        directions = fiber_tools.load_direction_field(
            directions_path, len(semantics.vertices)
        )
        canonical_fibers = fiber_tools.trace_canonical(
            semantics, directions, count=args.fibers, samples=args.fiber_samples
        )
        reached = int((canonical_fibers.stopped == 0).sum())
        print(f"canonical fibers: {reached}/{args.fibers} reached the envelope, "
              f"traced once for all shoes")
    else:
        raise SystemExit(
            f"no canonical direction field at {directions_path}. It defines the "
            "fibers, and without them a point has no position on the foot - "
            "this stage would have nothing to compute."
        )

    # The correspondence table: where every canonical vertex's fiber starts on
    # the skin. Built once and cached, because the canonical anatomy is the
    # same for all 27 shoes - this is the whole reason the flow formulation
    # makes addressing cheap.
    tracer = correspondence = None
    if directions_path.is_file():
        tracer = addressing.FiberTracer(semantics, directions, device=device)
        table = root / "canonical" / "correspondence.npz"
        if table.is_file():
            correspondence = addressing.CanonicalCorrespondence.load(table)
            print(f"correspondence table: read from {table.name}")
        else:
            moment = time.time()
            correspondence = addressing.CanonicalCorrespondence.build(
                tracer, step=args.trace_step,
                max_steps=int(3.0 / args.trace_step),
            )
            correspondence.save(table)
            landed = int((correspondence.status == addressing.REACHED).sum())
            print(f"correspondence table: {landed}/{len(correspondence.status)} "
                  f"vertices resolved in {time.time()-moment:.0f}s")

    def surface_determinants(slot: int) -> np.ndarray:
        """Local volume change of the map at the canonical surface.

        Central differences rather than autograd: six extra flow evaluations of
        7k points is cheaper than building a Jacobian graph, and the figure only
        needs the distribution.
        """

        h = 2.0e-4
        base = torch.as_tensor(
            reference.vertices, dtype=torch.float32, device=device
        )[None].expand(len(names), -1, -1).contiguous()
        columns = []
        with torch.no_grad():
            for axis in range(3):
                offset = torch.zeros(3, device=device)
                offset[axis] = h
                plus = integrate_batched(fields, base + offset, args.steps, 1.0)[slot]
                minus = integrate_batched(fields, base - offset, args.steps, 1.0)[slot]
                columns.append((plus - minus) / (2.0 * h))
            jacobian = torch.stack(columns, dim=-1)
            return torch.linalg.det(jacobian).cpu().numpy()

    started = time.time()
    print(f"\n{'shoe':44s} {'addressed':>10s} {'in foot':>8s} {'beyond':>8s} "
          f"{'resid med':>9s} {'max mm':>8s}")
    records = []
    # The fields are a fixed stack, so a shoe keeps its slot in ``names`` even
    # when only some are asked for; ``--shoes`` selects what to process, never
    # what to index.
    wanted = set(args.shoes)
    for index, name in enumerate(names):
        if name not in wanted:
            continue
        shoe = load_triangle_mesh(
            root / "inputs" / "shoe_preparation" / name / "shoe_normalized.ply"
        )
        points = area_samples(shoe.vertices, shoe.faces, args.samples, seed=17)
        with torch.no_grad():
            batch = torch.as_tensor(points, dtype=torch.float32, device=device)[None]
            batch = batch.expand(len(names), -1, -1).contiguous()
            canonical = integrate_batched(
                fields, batch, args.steps, -1.0
            )[index].cpu().numpy().astype(np.float64)
        def to_canonical(block: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.as_tensor(block, dtype=torch.float32, device=device)
                tensor = tensor[None].expand(len(names), -1, -1).contiguous()
                return integrate_batched(
                    fields, tensor, args.steps, -1.0
                )[index].cpu().numpy().astype(np.float64)

        def to_shoe(block: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.as_tensor(block, dtype=torch.float32, device=device)
                tensor = tensor[None].expand(len(names), -1, -1).contiguous()
                return integrate_batched(
                    fields, tensor, args.steps, 1.0
                )[index].cpu().numpy().astype(np.float64)

        book = addressing.AddressBook(
            semantics, tracer, correspondence, to_canonical, to_shoe
        )
        address = book.query(
            points, exact=True, verify=not args.no_verify,
            tolerance_mm=args.tolerance_mm,
            step=args.trace_step, max_steps=int(3.0 / args.trace_step),
        )
        addressed = address.addressed
        in_foot = address.outcome == addressing.INSIDE_ANATOMY
        depth = address.r
        progress = address.rho
        residual = address.residual[addressed]
        residual = residual[np.isfinite(residual)]

        shoe_dir = out / name
        shoe_dir.mkdir(parents=True, exist_ok=True)
        # The addresses themselves, per point. A coverage fraction cannot be
        # used to fit anything; these can.
        np.savez_compressed(
            shoe_dir / "addresses.npz",
            shoe_point=points.astype(np.float32),
            canonical_point=address.canonical.astype(np.float32),
            face=address.face,
            barycentric=address.barycentric.astype(np.float32),
            r=address.r.astype(np.float32),
            rho=address.rho.astype(np.float32),
            arclength=address.arclength.astype(np.float32),
            label=address.label,
            outcome=address.outcome,
            residual_mm=address.residual.astype(np.float32),
            label_names=np.array(semantics.boundary_label_names),
            outcome_names=np.array(
                [addressing.OUTCOME_NAMES[i] for i in range(len(addressing.OUTCOME_NAMES))]
            ),
        )
        counts = {
            addressing.OUTCOME_NAMES[int(code)]: int(n)
            for code, n in zip(*np.unique(address.outcome, return_counts=True))
        }
        record = {
            "shoe": name, "samples": args.samples,
            "addressed_fraction": float(addressed.mean()),
            "inside_foot_fraction": float(in_foot.mean()),
            "beyond_domain_fraction": float(
                (address.outcome == addressing.BEYOND_DOMAIN).mean()
            ),
            "not_anatomical_fraction": float(
                (address.outcome == addressing.NOT_ANATOMICAL).mean()
            ),
            "unresolved_fraction": float(
                (address.outcome == addressing.UNRESOLVED).mean()
            ),
            "unverified_fraction": float(
                (address.outcome == addressing.UNVERIFIED).mean()
            ),
            "outcome_counts": counts,
            "penetration_median_mm": (
                float(-np.nanmedian(address.arclength[in_foot]) * 262.5)
                if in_foot.any() else None
            ),
            "penetration_max_mm": (
                float(-np.nanmin(address.arclength[in_foot]) * 262.5)
                if in_foot.any() else None
            ),
            "depth_median": float(np.nanmedian(depth[addressed])) if addressed.any() else None,
            "progress_median": float(np.nanmedian(progress[addressed])) if addressed.any() else None,
            "residual_median_mm": float(np.median(residual)) if residual.size else None,
            "residual_p99_mm": float(np.percentile(residual, 99)) if residual.size else None,
            "residual_max_mm": float(residual.max()) if residual.size else None,
        }
        records.append(record)
        (shoe_dir / "address_coverage.json").write_text(json.dumps(record, indent=2) + "\n")

        if canonical_fibers is not None:
            def forward(block: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    tensor = torch.as_tensor(block, dtype=torch.float32, device=device)
                    tensor = tensor[None].expand(len(names), -1, -1).contiguous()
                    return integrate_batched(
                        fields, tensor, args.steps, 1.0
                    )[index].cpu().numpy().astype(np.float64)

            carried = fiber_tools.push_through(canonical_fibers, forward)
            fiber_tools.write_polydata(
                shoe_dir / "fibers.vtp", carried, canonical_fibers.progress
            )
            np.savez_compressed(
                shoe_dir / "fiber_samples.npz",
                curves=carried.astype(np.float32),
                progress=canonical_fibers.progress.astype(np.float32),
                origins=canonical_fibers.origins.astype(np.float32),
                inner_face_indices=canonical_fibers.face_indices,
                stopped=canonical_fibers.stopped,
            )

        if not args.no_figures:
            figures.write_coverage_overlay(
                shoe_dir / "coverage_overlay.ply", points, addressed, in_foot,
            )
            figures.coverage_figure(
                shoe_dir / "coverage.png", name, points, addressed,
                in_foot, depth,
                shoe_vertices=shoe.vertices, shoe_faces=shoe.faces,
            )
            with torch.no_grad():
                canonical_surface = torch.as_tensor(
                    reference.vertices, dtype=torch.float32, device=device
                )[None].expand(len(names), -1, -1).contiguous()
                landed = integrate_batched(
                    fields, canonical_surface, args.steps, 1.0
                )[index].cpu().numpy().astype(np.float64)
            # The shoe and the anatomy have to appear together: either alone
            # says nothing about whether the foot sits inside the shoe.
            figures.alignment_figure(
                shoe_dir / "alignment.png", name,
                shoe.vertices, shoe.faces,
                instances[index], reference.faces,
                carried if canonical_fibers is not None else None,
                reference.vertices, landed, instances[index],
            )
            figures.flow_figure(
                shoe_dir / "flow.png", name, reference.vertices, landed,
                reference.faces, determinant=surface_determinants(index),
            )
        print(f"{name[:44]:44s} {record['addressed_fraction']*100:9.2f}% "
              f"{record['inside_foot_fraction']*100:7.2f}% "
              f"{record['beyond_domain_fraction']*100:7.2f}% "
              f"{(record['residual_median_mm'] or float('nan')):9.5f} "
              f"{(record['residual_max_mm'] or float('nan')):8.3f}", flush=True)

    # Assembled from what is on disk rather than from this process's own
    # records, so the set can be split across GPUs and still summarise as one.
    # Shards finish at different times, so a record on disk may still be from
    # an earlier run with a different shape. Only fully formed ones count.
    records = [
        record for record in (
            json.loads(path.read_text())
            for path in sorted(out.glob("*/address_coverage.json"))
        )
        if "residual_median_mm" in record
    ]
    if not records:
        raise SystemExit("no addresses were written")
    fractions = [r["addressed_fraction"] for r in records]
    resid = [r["residual_median_mm"] for r in records if r["residual_median_mm"] is not None]
    worst = [r["residual_max_mm"] for r in records if r["residual_max_mm"] is not None]
    (out / "summary.json").write_text(json.dumps({
        "shoes": records, "seconds": time.time() - started,
        "addressed_min": min(fractions), "addressed_median": float(np.median(fractions)),
        "addressed_max": max(fractions),
        "inside_foot_mean": float(np.mean([r["inside_foot_fraction"] for r in records])),
        "unverified_mean": float(np.mean([r["unverified_fraction"] for r in records])),
        "residual_median_mm": float(np.median(resid)) if resid else None,
        "residual_worst_mm": max(worst) if worst else None,
        "verified": not args.no_verify,
        "tolerance_mm": args.tolerance_mm,
        "trace_step": args.trace_step,
    }, indent=2) + "\n")
    print(f"\n{len(records)} shoes in {time.time()-started:.1f}s   "
          f"addressed min {min(fractions)*100:.2f}% median {np.median(fractions)*100:.2f}% "
          f"max {max(fractions)*100:.2f}%   inside-foot mean "
          f"{np.mean([r['inside_foot_fraction'] for r in records])*100:.3f}%")
    if resid:
        print(f"round trip of every address: median {np.median(resid):.5f} mm, "
              f"worst {max(worst):.3f} mm   "
              f"unverified mean {np.mean([r['unverified_fraction'] for r in records])*100:.3f}%")


if __name__ == "__main__":
    main()
