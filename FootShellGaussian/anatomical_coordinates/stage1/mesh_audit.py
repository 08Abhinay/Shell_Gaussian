"""Mesh sanity for the CAD set, as the report's Stage 1 asks, before any sign.

The winding number needs each surface to be present once and consistently
oriented. Two defects break it and both occur in this set:

    twins     a double-sided export: every face also present as a coincident
              copy facing the other way. The pair's solid angles cancel, so
              the winding number is 0 everywhere and nothing reads as inside.
              ``remove_twins`` keeps one face of each pair.
    flips     faces whose orientation disagrees with their neighbours'.

The audit reports both, plus open edges, and how decisive the winding number
is near the surface once twins are removed.

    python -m anatomical_coordinates.stage1.mesh_audit
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from . import common
from .winding import inside, orientation


def twin_faces(vertices: np.ndarray, faces: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    """Indices of faces that are the second of a coincident, opposite pair."""

    tri = vertices[faces]
    centre = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = normal / np.maximum(length, 1e-30)[:, None]
    distance, index = cKDTree(centre).query(centre, k=2)
    other = index[:, 1]
    twin = (distance[:, 1] <= tolerance) & ((normal * normal[other]).sum(1) < -0.99)
    # Of each pair keep the lower index, drop the higher.
    return np.nonzero(twin & (np.arange(len(faces)) > other))[0]


def remove_twins(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    drop = twin_faces(vertices, faces)
    keep = np.ones(len(faces), dtype=bool)
    keep[drop] = False
    return faces[keep]


#: A piece this small cannot be oriented from its own geometry; a mesh made
#: mostly of them is a triangle soup whose winding number is not trustworthy.
SOUP_PIECE_FACES = 10


def sign_ready(vertices: np.ndarray, faces: np.ndarray, device="cuda"):
    """The same surface, prepared so its winding number means inside.

    1. drop the second face of every coincident opposite pair;
    2. merge coincident vertices, so faces along a seam become neighbours;
    3. make orientation consistent within every connected piece;
    4. orient each piece outwards by its *own* winding number: a closed piece
       reads ~1 just behind its faces and ~0 just in front of them; a piece
       that reads the other way round is flipped.

    Returns the prepared vertices and faces, and how much of the mesh was
    made of pieces too small to orient (the soup fraction). The source mesh
    on disk is never modified.
    """

    import trimesh

    from .winding import winding_numbers

    mesh = trimesh.Trimesh(vertices, remove_twins(vertices, faces), process=False)
    mesh.merge_vertices()
    trimesh.repair.fix_winding(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    pieces = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1)
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    step = 0.2 / 262.5
    generator = np.random.default_rng(0)
    soup = 0
    flipped = 0
    for piece in pieces:
        piece = np.asarray(piece)
        if len(piece) < SOUP_PIECE_FACES:
            soup += len(piece)
            continue
        weight = area[piece]
        if weight.sum() <= 0:
            continue
        pick = generator.choice(piece, size=min(128, len(piece)), p=weight / weight.sum())
        centre = tri[pick].mean(axis=1)
        unit = normal[pick] / np.maximum(area[pick], 1e-30)[:, None]
        own = faces[piece]
        behind = winding_numbers(centre - step * unit, vertices, own, device)
        ahead = winding_numbers(centre + step * unit, vertices, own, device)
        if np.median(behind - ahead) < 0:
            faces[piece] = faces[piece][:, ::-1]
            flipped += 1
    return vertices, faces, {
        "pieces": int(len(pieces)),
        "pieces_flipped": int(flipped),
        "soup_face_fraction": float(soup / max(len(faces), 1)),
    }


def edge_report(faces: np.ndarray) -> dict:
    """Open edges, and edges whose two faces traverse them the same way."""

    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    # An edge shared by two faces is consistently oriented when the two faces
    # traverse it in opposite directions.
    forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
    net = np.bincount(inverse, weights=2 * forward - 1, minlength=len(counts))
    manifold = counts == 2
    return {
        "edges": int(len(counts)),
        "open_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "inconsistent_edges": int((manifold & (net != 0)).sum()),
    }


def _near_surface(vertices, faces, samples, offsets_mm, seed=0):
    """Points a small step to either side of the surface, and which side."""

    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    generator = np.random.default_rng(seed)
    pick = generator.choice(len(faces), size=samples, p=area / area.sum())
    u, v = generator.random(samples), generator.random(samples)
    over = u + v > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    base = tri[pick, 0] + u[:, None] * (tri[pick, 1] - tri[pick, 0]) + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    normal = np.cross(tri[pick, 1] - tri[pick, 0], tri[pick, 2] - tri[pick, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    offset = generator.uniform(*offsets_mm, samples) / common.MILLIMETRES
    side = np.where(generator.random(samples) < 0.5, -1.0, 1.0)
    return base + (side * offset)[:, None] * normal, side


def _score(w, side):
    decisive = (w < 0.25) | (w > 0.75)
    # A point just behind an outward face should read inside, one just in
    # front of it outside. Offsets are kept well under a wall's thickness so
    # a "behind" point has not already passed out through the far side.
    agrees = (w > 0.5) == (side < 0)
    return float(decisive.mean()), (float(agrees[decisive].mean()) if decisive.any() else None)


def audit(name: str, device="cuda", samples: int = 20000) -> dict:
    mesh = common.shoe_mesh(name)
    vertices, faces = mesh.vertices, mesh.faces
    twins = twin_faces(vertices, faces)
    cleaned = remove_twins(vertices, faces)
    points, side = _near_surface(vertices, cleaned, samples, (0.1, 0.4))
    before = _score(inside(points, vertices, cleaned, device), side)

    prepared_v, prepared_f, info = sign_ready(vertices, faces, device)
    # Fresh probes on the prepared mesh: a flipped piece has its "behind"
    # on the other side, so the original probes' sides would be wrong there.
    points, side = _near_surface(prepared_v, prepared_f, samples, (0.1, 0.4))
    after = _score(inside(points, prepared_v, prepared_f, device), side)
    return {
        "shoe": name,
        "faces": int(len(faces)),
        "twin_pairs_removed": int(len(twins)),
        **{f"after_{k}": v for k, v in edge_report(prepared_f).items()},
        **info,
        "decisive_before": before[0], "side_agreement_before": before[1],
        "decisive_after": after[0], "side_agreement_after": after[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shoes", nargs="*", default=None)
    args = parser.parse_args()
    out = common.STAGE1_OUTPUT / "mesh_audit"
    out.mkdir(parents=True, exist_ok=True)
    names = args.shoes or sorted(p.name for p in (common.PIPELINE_OUTPUT / "inputs" / "shoe_preparation").iterdir())
    names = [n for n in names if n != "sneaker_vibe"]
    rows = []
    for name in names:
        row = audit(name)
        rows.append(row)
        print(f"{name[:34]:34s} twins {row['twin_pairs_removed']:6d} pieces {row['pieces']:6d} "
              f"flipped {row['pieces_flipped']:4d} soup {row['soup_face_fraction']*100:5.1f}% | "
              f"decisive {row['decisive_before']*100:5.1f} -> {row['decisive_after']*100:5.1f}%  "
              f"side-agree {100*(row['side_agreement_before'] or 0):5.1f} -> "
              f"{100*(row['side_agreement_after'] or 0):5.1f}%", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
