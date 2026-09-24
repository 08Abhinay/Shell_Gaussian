# State of play — before starting §1.2 and §1.3

**Written for a fresh session implementing the material field and the
set-valued correspondence.** Everything below was verified against disk on
2026-09-23. Section numbers refer to `notes/representation.pdf`.

The two files in `docs/history/` describe the *foot fitter* and were accurate
on 2026-09-21. They predate the coordinate map, the addresses, the material
measurement and the lower-leg search, so read them only for background on how
the foot fitting came to be. `session_handoff.md` references a `CONTRACT.md`
that no longer exists; the file it means is `docs/geometry_contract.md`, which
is still current.

---

## Read in this order

1. `notes/representation.pdf` §1.1–§1.3 — the whole point. §1.2 is two pages.
2. This file.
3. `anatomical_coordinates/coordinate_mapping/material.py` — its module
   docstring states the measurement conventions and why each was chosen.
4. `anatomical_coordinates/coordinate_mapping/address.py` — the same for the
   coordinate query.
5. `docs/geometry_contract.md` — inherited frames and conventions. Still valid.

---

## §1.1 is built and measured

One canonical anatomy; a flow map Φᵢ carrying it into each of 27 shoes.

```
coordinate_mapping/field.py      one velocity field and its integrator
coordinate_mapping/batch.py      all 27 fields in one set of kernels
coordinate_mapping/lookup.py     the canonical tetrahedral volume, r, rho
coordinate_mapping/address.py    fiber tracing, the correspondence table,
                                 AddressBook.query / .place
coordinate_mapping/material.py   g, delta_in, delta_out per anatomical site
coordinate_mapping/cross_shoe.py one address sent through all 27 and back
```

Both directions of the address work:

```python
address = book.query(points)                 # points  -> (u, v, r)
points  = book.place(face, barycentric, r)   # (u, v, r) -> points
```

`(u, v)` is `face` (an inner-surface triangle) plus `barycentric`. `r` is
arclength along the fiber **in canonical space**, with millimetres recorded
alongside. See "Decisions already made" for why.

Measured, all 27 shoes:

| | |
|---|---|
| boundary fit of the map | 0.275 mm median, 4.61 mm worst |
| folded cells | **0 of 108,000** probes |
| address round trip | 0.00031 mm median |
| one address through all 27 shoes | 100% returned at r = 0.25 and 0.50, drift p99 0.0061 mm |
| …at r = 0 exactly | 95.4% (the skin is a knife edge; see the surface tolerance in `query`) |
| points addressed | 97.36% median |

---

## §1.2's measurement layer is built; the learning is not

`material_stage.py` writes, per shoe, a value at each of **16,444 sites** — one
per triangle of the canonical inner surface, identical across shoes, so field
values are comparable entry by entry.

`material/<shoe>/material.npz`:

| array | meaning |
|---|---|
| `covered` | bool — the sign of `g(u,v)` |
| `delta_in`, `delta_out` | canonical arclength; NaN where uncovered |
| `delta_in_mm`, `delta_out_mm` | the same in millimetres |
| `signed_delta_in` | negative where material sits inside the foot |
| `layers` | how many crossings — 2 is one wall |
| `penetration_mm` | how far material reaches inside the foot |
| `crossing_site`, `crossing_r`, `crossing_mm` | **the full crossing list** |

`material.base_field(g, delta_in, delta_out, r)` evaluates
`max{g, δin − r, r − δout}`. It is a plain function of four arrays so the
learned version, which substitutes networks for the first three, composes the
same way.

**`crossing_*` is already §1.3's `Rᵢ(u,v)`** — the set of radii where the fiber
meets the surface. Absence and multiplicity are recorded, not collapsed.

Current measurements:

| | |
|---|---|
| foot covered | 92.4% median, 64.1–99.9% |
| clearance `δ_in` | 6.21 mm median |
| wall thickness | 7.45 mm median, 1.87–17.58 mm |
| sites pierced | 1.39% median, 8.15% worst |
| deepest intrusion | 29.5 mm (`sneaker_7`) |

Sorting the 27 by coverage reproduces a footwear taxonomy nothing encoded:
boots (99.9% foot, 58–76% leg), closed shoes (89–99%), open footwear
(64–73%, 0% leg). That is `g` working.

**Not built:** everything learned. `μ_clr`, `Δⁱⁿ`, `δᵒᵘᵗ` as networks, the
residual field `R(q, zᵢ)`, the latents `zᵢ`, the eikonal term, the training
loop. §1.3's `Cᵢ(u,v)` as a queryable object.

---

## Decisions already made — do not relitigate without a reason

**`r` is canonical arclength, not millimetres.** `B` is evaluated at a point's
*anatomical* coordinate, so it takes that coordinate's units. §1.2's eikonal
term is what restores physical meaning: it constrains `‖∇ₓf‖ = 1` in Euclidean
space, which is why the flow map had to be differentiable and the older
tetrahedral cage could not have been used.

**Crossings, not containment.** The shoe meshes are not watertight — several
thousand open edges each, one with zero signed volume — so there is no inside
test to make. First and last crossing give the two extents.

**Material inside the foot is a measurement, not an error.** A rigid shoe on an
undeformable foot must overlap somewhere. It is recorded as negative
`signed_delta_in`, read from the address stage's buried points, because a fiber
starting inside material has no entry crossing to find.

**Oversized shoe triangles are subdivided before the crossing search.** Without
it a 90 mm-radius sole triangle escapes a 10 mm candidate search and a boot's
sole reads 88% covered instead of 100%. Verified converged: widening the search
from 96 to 320 neighbours changed no measurement at all.

**Nothing is judged by a training loss.** Every result is scored by the exact
evaluator in `foot_prior`. A loss going down is not evidence.

---

## Decide these before writing the training loop

**Does `μ_clr` condition on `Fᵢ`?** §1.2 writes it as
`δⁱⁿᵢ = μ_clr(u,v; Fᵢ) + Δⁱⁿᵢ`. All 27 shoes have *different* feet (foot ‖β‖
median 5.0), so each `Fᵢ` appears exactly once and "this foot is wide here" and
"this shoe is tight here" produce identical data. The conditioning is not
identifiable from this dataset. The `Δⁱⁿ → 0` regulariser still works as an
inductive bias without it, so a shared prior over `(u,v)` alone is the safer
reading.

**What does an odd crossing count mean?** 24.6% of covered sites median, 55.4%
worst — a fiber entering material and never cleanly exiting, through a genuinely
open mesh boundary such as a croc hole's rim. Confirmed *not* to be fibers
running out of room (`δ_out / reach` is 0.20, 0% near the end). At those sites
`δ_out` is a real crossing but not an exit. One shoe shows 0.0%, so this is mesh
quality, not method.

**Where is the line between compression and fit error?** ~1 mm of intrusion is
plausible soft tissue; 29.5 mm is not. A clearance prior trained on both learns
neither.

**27 shoes is small for an auto-decoder.** These methods normally want hundreds.
Expect `zᵢ` to memorise.

---

## Hard rules

- **`foot_prior/` is read-only.** Import from it freely; never modify it. It is
  the independent judge, and the comparison is worthless if it moves.
- **The exact evaluator decides**, never the differentiable surrogate. Measured
  against the judge, the collar surrogate disagrees exactly where the leg lives.
- **Re-running is expensive.** `join → coordinates → address → material` is
  ~35 min on three GPUs. Changing the foot or leg stage invalidates all of it.

---

## Operational notes

- Entry point: `./scripts/run_pipeline.sh --from <stage> --to <stage> --gpu-ids 1 2 3`
- Output tree: `/home/ab5298/Outputs/FootShellGaussian/anatomical_coordinates/`
- Python: `/home/ab5298/anaconda3/envs/Shell/bin/python`
- Tests: `python -m pytest anatomical_coordinates/tests/ -q` — 62 pass.
- `coordinates` is one batched process by design; it does not shard. The work
  is launch-bound, so 27 shoes share one set of kernels. `address` and
  `material` do shard.
- `canonical/correspondence.npz` and `canonical/fiber_paths.npz` are properties
  of the canonical anatomy, not of any shoe. They are cached and reused
  correctly across runs — do not delete them expecting a refresh.
- To find a running stage, match the interpreter, not the module name:
  `ps -eo comm,args --no-headers | awk '$1=="python" && /material_stage/'`.
  `ps -eo cmd | grep '[m]aterial_stage'` matches the wrapper shell that
  launched it and deadlocks any wait loop built on it.

---

## Known weaknesses, stated plainly

- **13 of 27 shoes still have lower-leg contact** with the shoe, total contact
  area 7.485%. The upgraded search cleared the worst case
  (`ww_ii_german_jack_boots`, 367 pairs → 0) but the remainder are limited by
  where the **foot** stage placed the foot, not by the leg search. That is the
  known next lever and it is deliberately parked.
- **Most feet sit exactly at the ‖β‖ = 5.0 cap**, meaning the fit wanted to go
  further than allowed. Worth understanding before trusting foot shape as an
  input to anything learned.
- The map's boundary fit degraded slightly (0.22 → 0.275 mm median) when the
  leg search began using shape coefficients. More varied anatomy, more for the
  flow to represent. Still well inside tolerance.
