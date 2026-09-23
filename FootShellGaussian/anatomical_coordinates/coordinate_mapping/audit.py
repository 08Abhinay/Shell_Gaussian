"""Audit a fitted flow map where it is hardest, not where it is easiest.

A uniform sample of the bounding box mostly lands in empty space far from the
anatomy, where any smooth field is near the identity and the determinant is
uninteresting. The map is under strain in a shell hugging the skin - which is
also exactly the region the footwear material occupies, so it is the region the
representation actually depends on.

This is still a sampled statement, not the per-element guarantee the cage deformation stage gives.
The honest claim it supports is "no inversion found in N samples concentrated
where inversion would occur", and N is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from foot_prior.mesh import load_triangle_mesh

from .build import domain_box, load_pair
from .sampling import shell_samples
from .field import VelocityField, integrate, jacobian_determinants, round_trip_error


def audit(
    root: Path, name: str, field_path: Path, steps: int = 16, count: int = 20000
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical, instance, faces = load_pair(root, name)
    lower, upper = domain_box(canonical, instance)
    field = VelocityField(
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
    ).to(device)
    field.load_state_dict(torch.load(field_path, map_location=device))
    field.eval()

    groups = {
        # outward into the footwear volume, where material lives
        "shell_outward": shell_samples(canonical, faces, count, (0.0, 0.30), seed=1),
        # a thin skin-hugging band, where the deformation gradient peaks
        "skin_band": shell_samples(canonical, faces, count, (-0.01, 0.03), seed=2),
        # slightly inward, still inside the mathematical domain
        "shell_inward": shell_samples(canonical, faces, count // 2, (-0.05, 0.0), seed=3),
    }
    report: dict = {"shoe": name, "steps": steps, "groups": {}}
    worst = float("inf")
    for tag, points in groups.items():
        tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
        determinants = jacobian_determinants(field, tensor, steps).detach()
        trip = round_trip_error(field, tensor, steps).detach()
        block = {
            "samples": int(len(points)),
            "jacobian_minimum": float(determinants.min()),
            "jacobian_p01": float(determinants.quantile(0.01)),
            "jacobian_median": float(determinants.median()),
            "jacobian_maximum": float(determinants.max()),
            "inverted_count": int((determinants <= 0.0).sum()),
            "round_trip_max_mm": float(trip.max()) * 262.5,
        }
        # A bi-Lipschitz statement needs the singular values, not the volume
        # ratio: a determinant near 1 can still hide a sheared element.
        singular = torch.linalg.svdvals(
            torch.stack([
                torch.func.jacrev(lambda p: integrate(field, p[None], steps, 1.0)[0])(row)
                for row in tensor[:512]
            ])
        )
        block["singular_min"] = float(singular.min())
        block["singular_max"] = float(singular.max())
        block["bi_lipschitz_ratio"] = float(singular.max() / singular.min().clamp(min=1e-9))
        report["groups"][tag] = block
        worst = min(worst, block["jacobian_minimum"])
    report["jacobian_minimum_overall"] = worst
    report["inverted_total"] = sum(
        g["inverted_count"] for g in report["groups"].values()
    )
    report["total_samples"] = sum(g["samples"] for g in report["groups"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shoes", nargs="+", required=True)
    parser.add_argument("--fields", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--count", type=int, default=20000)
    args = parser.parse_args()

    fields = args.fields or (args.root / "diffeomorphic_volume")
    for name in args.shoes:
        report = audit(
            args.root, name, fields / name / "velocity_field.pt",
            steps=args.steps, count=args.count,
        )
        (fields / name / "flow_audit.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"\n=== {name} ===")
        print(
            f"  {report['total_samples']} samples, "
            f"{report['inverted_total']} inverted, "
            f"min det {report['jacobian_minimum_overall']:.4f}"
        )
        for tag, block in report["groups"].items():
            print(
                f"  {tag:14s} n={block['samples']:6d} minJ={block['jacobian_minimum']:7.4f} "
                f"p01={block['jacobian_p01']:7.4f} med={block['jacobian_median']:6.4f} "
                f"biLip={block['bi_lipschitz_ratio']:6.2f} "
                f"rt={block['round_trip_max_mm']:.2e}mm"
            )


if __name__ == "__main__":
    main()
