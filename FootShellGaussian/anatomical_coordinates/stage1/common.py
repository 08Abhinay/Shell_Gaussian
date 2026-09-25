"""Loading what every Stage 1 step shares: the canonical anatomy and the flows.

The address and material stages each rebuild the batched flow fields inline.
This does it once, the same way, so Stage 1 code reads the pipeline's own
outputs without re-deriving any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from foot_prior.mesh import TriangleMesh, load_triangle_mesh

from ..coordinate_mapping.address import AddressBook, CanonicalCorrespondence, FiberTracer
from ..coordinate_mapping.batch import BatchedVelocityFields, integrate_batched
from ..coordinate_mapping.lookup import CanonicalSemantics
from ..pipeline.stages import BY_KEY

PIPELINE_OUTPUT = Path("/home/ab5298/Outputs/FootShellGaussian/anatomical_coordinates")
#: Stage 1 results. The pipeline outputs above are only read, never written.
STAGE1_OUTPUT = Path("/home/ab5298/Outputs/FootShellGaussian/stage1")
MILLIMETRES = 262.5
#: Integration steps used by every stage that built or read the flows.
FLOW_STEPS = 4


@dataclass
class Flows:
    """All 27 flows, and the order they are stored in."""

    fields: BatchedVelocityFields
    names: list[str]
    device: torch.device

    def _run(self, points: np.ndarray, index: int, direction: float,
             block: int = 262144) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        out = np.empty_like(points)
        with torch.no_grad():
            for begin in range(0, len(points), block):
                piece = torch.as_tensor(points[begin:begin + block],
                                        dtype=torch.float32, device=self.device)
                piece = piece[None].expand(len(self.names), -1, -1).contiguous()
                out[begin:begin + block] = integrate_batched(
                    self.fields, piece, FLOW_STEPS, direction
                )[index].cpu().numpy()
        return out

    def to_shoe(self, points: np.ndarray, name: str) -> np.ndarray:
        """Canonical space -> this shoe's normalized frame."""
        return self._run(points, self.names.index(name), 1.0)

    def to_canonical(self, points: np.ndarray, name: str) -> np.ndarray:
        """This shoe's normalized frame -> canonical space."""
        return self._run(points, self.names.index(name), -1.0)


def load_flows(root: Path = PIPELINE_OUTPUT, device=None) -> Flows:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    joined = root / BY_KEY["join"].directory
    coords = root / BY_KEY["coordinates"].directory
    names = json.loads((coords / "summary.json").read_text())["names"]
    reference = load_triangle_mesh(joined / "reference" / "neutral_foot_lower_leg.ply")
    instances = [load_triangle_mesh(joined / n / "foot_lower_leg.ply").vertices
                 for n in names]
    # The same box as the coordinate stage fitted in: without it the input
    # normalization, and so every flowed point, would differ.
    lower = np.stack([np.minimum(reference.vertices.min(0), v.min(0)) - 0.25
                      for v in instances])
    upper = np.stack([np.maximum(reference.vertices.max(0), v.max(0)) + 0.25
                      for v in instances])
    fields = BatchedVelocityFields(
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
    ).to(device)
    fields.load_state_dict(torch.load(coords / "coordinate_fields.pt",
                                      map_location=device, weights_only=True))
    fields.eval()
    return Flows(fields, names, device)


def load_canonical(root: Path = PIPELINE_OUTPUT, device=None):
    """The canonical volume, its fiber tracer and the cached fiber paths."""

    semantics = CanonicalSemantics(
        root / "canonical" / "volume" / "canonical_volume.npz",
        root / "canonical" / "semantic_field" / "semantic_field.npz",
    )
    directions = np.load(root / "canonical" / "semantic_field" / "fiber_field.npz")["directions"]
    tracer = FiberTracer(semantics, directions, device=device)
    paths = np.load(root / "canonical" / "fiber_paths.npz")
    return semantics, tracer, paths


def address_book(semantics, tracer, flows: Flows | None, name: str | None,
                 root: Path = PIPELINE_OUTPUT) -> AddressBook:
    """An AddressBook for one shoe, or for canonical space when ``flows`` is None."""

    table = CanonicalCorrespondence.load(root / "canonical" / "correspondence.npz")
    if flows is None:
        identity = lambda p: np.asarray(p, dtype=np.float64)
        return AddressBook(semantics, tracer, table, identity, identity)
    return AddressBook(
        semantics, tracer, table,
        lambda p: flows.to_canonical(p, name), lambda p: flows.to_shoe(p, name),
    )


def shoe_mesh(name: str, root: Path = PIPELINE_OUTPUT) -> TriangleMesh:
    return load_triangle_mesh(root / "inputs" / "shoe_preparation" / name / "shoe_normalized.ply")


def fitted_anatomy(name: str, root: Path = PIPELINE_OUTPUT) -> TriangleMesh:
    return load_triangle_mesh(root / BY_KEY["join"].directory / name / "foot_lower_leg.ply")


def material(name: str, root: Path = PIPELINE_OUTPUT):
    return np.load(root / BY_KEY["material"].directory / name / "material.npz")
