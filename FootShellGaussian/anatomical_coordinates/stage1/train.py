"""Stage 1: identical auto-decoders that differ only in their input coordinates.

DeepSDF-style (Park et al., CVPR 2019): one latent code per shoe, optimized
jointly with a shared decoder; at test time the decoder is frozen and a new
code is fitted to whatever evidence the test shoe provides.

    variant  input                                   (see ``dataset``)
    A        CAD frame, unit sphere per shoe          3-D
    Aplus    normalized shoe frame, no anatomy        3-D
    B        anatomical address: skin point s, r      3 + 1
    Bshell   B, with representation.pdf 1.2's structure: a base shell plus a
             sparse residual (see ``ShellModel``)
    AB       A and B together                          3 + 3 + 1

AB was added after the first comparison. B completed the cavity best and A
the sole, where the ground plane that is flat in the shoe's own frame
becomes a curved surface in anatomical coordinates (r follows the arch).
Giving the decoder both asks whether their strengths add.

A, A+ and B share everything - architecture, width, latent size, Fourier
features, samples, targets, steps, seeds - so a difference among them comes
from the coordinates, which is the report's test (section 6.1). Bshell is
the fourth row argued for in the review of that plan: representation.pdf 1.2
says coordinates alone are not a prior, because a free decoder can treat
them as a mere change of variables; the shell is where the anatomy is meant
to become one.

The target is the unsigned distance in millimetres, clamped at 10 mm. Every
shoe keeps 10 % of its samples out of training; all numbers are measured on
those. Three evaluations:

    fit        codes trained with the decoder; held-out samples of training
               shoes (split "all")
    full       test shoe, code fitted to all its non-held-out samples
    ring       test shoe, code fitted only to what a 36-view ring at 0 degrees
               observes: visible surface samples (distance 0) and samples the
               ring shows to be empty (no surface within 1 mm). Nothing about
               the interior, sole or top reaches the code.

and the numbers reported per region (visible / interior / sole / top):

    surface_mm   mean predicted distance at true surface samples - how far the
                 model is from putting a surface where one exists
    recall_1mm   share of true surface samples predicted within 1 mm (and 2 mm)
    field_mm     mean |predicted - true| distance at near-surface samples
    false_surface  share of volume samples more than 5 mm from any surface
                 that the model reports within 1 mm of one

Surface error and recall alone reward a model that predicts "surface
everywhere"; field_mm and false_surface are what catch it, so the four are
read together.

Caveat for B and Bshell in the ring test: their coordinates come from a foot
fitted into the *complete* mesh, cavity included. A StockX shoe would need the
foot placed from the exterior alone, so this test is optimistic for them.

    python -m anatomical_coordinates.stage1.train --variant B --split sandals
"""

from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import common
from .dataset import NEAR, REGION_NAMES, SURFACE, VOLUME

MM = common.MILLIMETRES
CLAMP_MM = 10.0
#: Weight of the L1 penalty that keeps Bshell's residual small.
RESIDUAL_SPARSITY = 0.05

SPLITS = {
    "all": [],
    "sandals": ["birkenstock_arizona_sandal", "sandal_1", "sandals_0001",
                "duinn_shoes_womens_hiking_sandal_sport"],
    "crocs": ["crocs", "crocs_shoe", "crocs_by_speedyart_studio"],
    "boots": ["leather_boots", "ww_ii_german_jack_boots"],
    # Closed shoes with no near-duplicate in the set (sneaker_1 ~ sneaker_3,
    # 4 ~ 6 and 5 ~ 7 would leak across a split).
    "closed": ["adidas_substance", "canvas_shoe", "pb129_shoe_low"],
}
VARIANTS = ("A", "Aplus", "B", "Bshell", "AB")


def load(name: str, variant: str, holdout: float = 0.1, seed: int = 0) -> dict:
    data = np.load(common.STAGE1_OUTPUT / "dataset" / f"{name}.npz")
    keep = data["b_valid"]             # the same points for every variant
    if variant == "A":
        x = data["coords_a"]
    elif variant == "Aplus":
        x = data["points"]
    elif variant == "AB":
        x = np.concatenate((data["coords_a"], data["coords_s"], data["coords_r"][:, None]), axis=1)
    else:
        x = np.concatenate((data["coords_s"], data["coords_r"][:, None]), axis=1)
    # A fixed checksum, not hash(): Python randomizes str hashes per process,
    # and every variant must hold out exactly the same samples.
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    held = rng.random(len(keep)) < holdout
    return {
        "x": x[keep].astype(np.float32),
        "r_mm": (np.nan_to_num(data["coords_r"][keep]) * MM).astype(np.float32),
        "target": np.minimum(data["udf"][keep] * MM, CLAMP_MM).astype(np.float32),
        "kind": data["kind"][keep],
        "region": data["region"][keep],
        "free0": data["free0"][keep],
        "held": held[keep],
    }


def load_sites(name: str) -> dict:
    """Measured envelope per site: Bshell's shell is supervised by these."""

    mat = common.material(name)
    covered = mat["covered"]
    return {
        "s": mat["site_point"].astype(np.float32),
        "uncovered": (~covered).astype(np.float32),
        "covered": covered,
        # Material inside the foot is clipped to the skin, as the PDF's
        # 0 <= delta_in; the depth itself is fit error, not clearance.
        "din": np.nan_to_num(np.maximum(mat["signed_delta_in"], 0.0) * MM).astype(np.float32),
        "dout": np.nan_to_num(mat["delta_out"] * MM).astype(np.float32),
    }


class FourierFeatures(nn.Module):
    def __init__(self, dims: int, count: int = 64, scale: float = 2.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("basis", torch.randn((dims, count), generator=g) * scale)
        self.out = dims + 2 * count

    def forward(self, x):
        p = 2 * np.pi * x @ self.basis
        return torch.cat((x, p.sin(), p.cos()), dim=-1)


class Decoder(nn.Module):
    """MLP on (features, code), with the input re-injected halfway."""

    def __init__(self, dims: int, latent: int = 64, width: int = 256, depth: int = 6,
                 outputs: int = 1):
        super().__init__()
        self.encode = FourierFeatures(dims)
        first = self.encode.out + latent
        self.half = depth // 2
        self.layers = nn.ModuleList(
            nn.Linear(first if i == 0 else width + (first if i == self.half else 0), width)
            for i in range(depth))
        self.last = nn.Linear(width, outputs)
        self.outputs = outputs
        self.act = nn.Softplus(beta=100)

    def forward(self, x, z):
        inp = torch.cat((self.encode(x), z), dim=-1)
        h = inp
        for i, layer in enumerate(self.layers):
            if i == self.half:
                h = torch.cat((h, inp), dim=-1)
            h = self.act(layer(h))
        out = self.last(h)
        return out.squeeze(-1) if self.outputs == 1 else out


class ShellModel(nn.Module):
    """representation.pdf 1.2, for an unsigned target.

    An envelope network reads the skin point and the shoe's code and gives the
    base shell - coverage g, inner offset delta_in and thickness, in mm - and

        shell(s, r) = | max(g, delta_in - r, r - delta_out) |

    is the distance along the fiber to the nearest face of that shell. The
    residual network adds whatever the shell cannot say, and an L1 penalty on
    it makes the shell the default explanation. On training shoes the shell
    is also supervised directly with the measured per-site envelope.
    """

    def __init__(self, dims: int, latent: int = 64):
        super().__init__()
        self.residual = Decoder(dims, latent)
        self.envelope = Decoder(3, latent, depth=4, outputs=3)

    def shell_values(self, s, z):
        e = self.envelope(s, z)
        g, din = e[:, 0], e[:, 1]
        return g, din, din + functional.softplus(e[:, 2])

    def forward(self, x, z, r_mm):
        g, din, dout = self.shell_values(x[:, :3], z)
        shell = torch.maximum(torch.maximum(g, din - r_mm), r_mm - dout).abs()
        residual = self.residual(x, z)
        return shell + residual, residual


def build_model(variant: str, dims: int, latent: int = 64) -> nn.Module:
    return ShellModel(dims, latent) if variant == "Bshell" else Decoder(dims, latent)


def predict(model, x, z, r_mm):
    """(prediction, residual). The residual only exists for Bshell."""

    if isinstance(model, ShellModel):
        return model(x, z, r_mm)
    pred = model(x, z)
    return pred, None


class Normalizer:
    """Per-dimension affine map of the training inputs onto [-1, 1]."""

    def __init__(self, shoes: list[dict]):
        stacked = np.concatenate([s["x"] for s in shoes])
        lo, hi = np.nanmin(stacked, 0), np.nanmax(stacked, 0)
        self.centre = torch.as_tensor(0.5 * (lo + hi))
        self.half = torch.as_tensor(np.maximum(0.5 * (hi - lo), 1e-6))

    def __call__(self, x: torch.Tensor, dims: slice = slice(None)) -> torch.Tensor:
        return (x - self.centre[dims].to(x.device)) / self.half[dims].to(x.device)


def _to_device(shoe: dict, norm: Normalizer, device) -> dict:
    out = {k: torch.as_tensor(v, device=device) for k, v in shoe.items()}
    out["x"] = norm(out["x"].float())
    return out


def _envelope_loss(model: ShellModel, sites: dict, codes, count: int, generator):
    """The measured envelope, for every training shoe in one pass.

    The sites are the same canonical triangles for every shoe - only their
    measured values differ - so one batch of site points serves all shoes.
    """

    shoes = codes.shape[0]
    pick = torch.randint(len(sites["s"]), (shoes, count), device=codes.device, generator=generator)
    rows = torch.arange(shoes, device=codes.device)[:, None].expand(shoes, count)
    g, din, dout = model.shell_values(
        sites["s"][pick].reshape(-1, 3),
        codes[:, None, :].expand(shoes, count, -1).reshape(shoes * count, -1))
    take = lambda key: sites[key][rows, pick].reshape(-1)
    # g < 0 where covered, > 0 where not: a logistic margin, in mm.
    loss = functional.softplus(-(2.0 * take("uncovered") - 1.0) * g).mean()
    covered = take("covered")
    if covered.any():
        loss = loss + 0.1 * ((din - take("din")).abs()[covered].mean()
                             + (dout - take("dout")).abs()[covered].mean())
    return loss


def train(variant: str, names: list[str], device, steps: int = 12000,
          per_shoe: int = 2048, latent: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    shoes = [load(n, variant) for n in names]
    norm = Normalizer(shoes)
    gpu = [_to_device(s, norm, device) for s in shoes]
    train_index = [torch.nonzero(~s["held"]).squeeze(1) for s in gpu]
    model = build_model(variant, shoes[0]["x"].shape[1], latent).to(device)
    sites = None
    if variant == "Bshell":
        raw = [load_sites(n) for n in names]
        sites = {k: torch.as_tensor(np.stack([r[k] for r in raw]), device=device)
                 for k in ("uncovered", "covered", "din", "dout")}
        sites["s"] = norm(torch.as_tensor(raw[0]["s"], device=device), slice(0, 3))
    codes = nn.Parameter(torch.randn(len(names), latent, device=device) * 0.01)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 5e-4},
        {"params": [codes], "lr": 1e-3},
    ])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.time()
    for step in range(steps):
        xs, ts, rs, zs = [], [], [], []
        for i, s in enumerate(gpu):
            pick = train_index[i][torch.randint(len(train_index[i]), (per_shoe,),
                                                device=device, generator=generator)]
            xs.append(s["x"][pick]); ts.append(s["target"][pick]); rs.append(s["r_mm"][pick])
            zs.append(codes[i].expand(per_shoe, -1))
        pred, residual = predict(model, torch.cat(xs), torch.cat(zs), torch.cat(rs))
        loss = (pred.clamp(max=CLAMP_MM) - torch.cat(ts)).abs().mean() \
            + 1e-4 * codes.square().sum(1).mean()
        if residual is not None:
            loss = loss + RESIDUAL_SPARSITY * residual.abs().mean()
            loss = loss + _envelope_loss(model, sites, codes, 512, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    return model, codes.detach(), norm, gpu, time.time() - started


def fit_code(model, shoe: dict, mode: str, device, steps: int = 1500,
             batch: int = 16384, latent: int = 64, seed: int = 0) -> torch.Tensor:
    """A new code for a shoe the model never saw, from ``mode`` evidence."""

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    train = ~shoe["held"]
    if mode == "full":
        exact = torch.nonzero(train).squeeze(1)
        free = torch.zeros(0, dtype=torch.long, device=device)
    else:
        seen_surface = train & (shoe["kind"] == SURFACE) & (shoe["region"] == 0)
        exact = torch.nonzero(seen_surface).squeeze(1)
        free = torch.nonzero(train & shoe["free0"] & (shoe["kind"] != SURFACE)).squeeze(1)
    code = torch.zeros(1, latent, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([code], lr=5e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        pick = exact[torch.randint(len(exact), (batch,), device=device, generator=generator)]
        pred, residual = predict(model, shoe["x"][pick], code.expand(batch, -1), shoe["r_mm"][pick])
        loss = (pred.clamp(max=CLAMP_MM) - shoe["target"][pick]).abs().mean()
        if residual is not None:
            loss = loss + RESIDUAL_SPARSITY * residual.abs().mean()
        if len(free):
            fp = free[torch.randint(len(free), (batch,), device=device, generator=generator)]
            fpred, _ = predict(model, shoe["x"][fp], code.expand(batch, -1), shoe["r_mm"][fp])
            # Observed empty: no surface within a millimetre of this point.
            loss = loss + torch.relu(1.0 - fpred).mean()
        loss = loss + 1e-4 * code.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return code.detach()[0]


@torch.no_grad()
def evaluate(model, code: torch.Tensor, shoe: dict) -> dict:
    held = shoe["held"]
    chunks = []
    for i in range(0, len(shoe["x"]), 65536):
        x = shoe["x"][i:i + 65536]
        chunks.append(predict(model, x, code.expand(len(x), -1), shoe["r_mm"][i:i + 65536])[0])
    pred = torch.cat(chunks).clamp(0, CLAMP_MM)
    out = {}
    for k, name in enumerate(REGION_NAMES):
        surf = held & (shoe["kind"] == SURFACE) & (shoe["region"] == k)
        near = held & (shoe["kind"] == NEAR) & (shoe["region"] == k)
        # Read these together: predicting "surface everywhere" makes
        # surface_mm and recall perfect, and is caught by field_mm (near
        # samples really are 0.5-10 mm away) and by false_surface.
        out[name] = {
            "samples": int(surf.sum()),
            "surface_mm": float(pred[surf].mean()) if surf.any() else None,
            "recall_1mm": float((pred[surf] <= 1.0).float().mean()) if surf.any() else None,
            "recall_2mm": float((pred[surf] <= 2.0).float().mean()) if surf.any() else None,
            "field_mm": float((pred[near] - shoe["target"][near]).abs().mean()) if near.any() else None,
        }
    vol = held & (shoe["kind"] == VOLUME)
    far = vol & (shoe["target"] > 5.0)
    out["volume_field_mm"] = float((pred[vol] - shoe["target"][vol]).abs().mean())
    out["false_surface"] = float((pred[far] < 1.0).float().mean()) if far.any() else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")

    everyone = json.loads((common.PIPELINE_OUTPUT / "coordinates" / "summary.json").read_text())["names"]
    held_out = SPLITS[args.split]
    names = [n for n in everyone if n not in held_out]
    model, codes, norm, gpu, seconds = train(args.variant, names, device, args.steps, seed=args.seed)

    result = {"variant": args.variant, "split": args.split, "train_shoes": names,
              "train_seconds": seconds, "steps": args.steps, "seed": args.seed,
              "fit": {}, "full": {}, "ring": {}}
    for i, name in enumerate(names):
        result["fit"][name] = evaluate(model, codes[i], gpu[i])
    for name in held_out:
        shoe = _to_device(load(name, args.variant), norm, device)
        for mode in ("full", "ring"):
            code = fit_code(model, shoe, mode, device, seed=args.seed)
            result[mode][name] = evaluate(model, code, shoe)

    out = common.STAGE1_OUTPUT / "runs" / args.split
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_seed{args.seed}"
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "codes": codes.cpu(), "names": names},
               out / f"{tag}.pt")
    print(f"[{args.split}/{tag}] trained {len(names)} shoes in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
