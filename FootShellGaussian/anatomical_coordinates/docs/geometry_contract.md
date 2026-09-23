# Geometry contract inherited from the NumPy fitter

Everything here was read out of `foot_prior/{mesh,supr_foot,alignment,containment,cavity}.py`
on 2026-09-21. The PyTorch fitter in this folder must either reproduce these
conventions exactly or state explicitly where it departs from them. Nothing in
`foot_prior/` is modified; the old fitter stays as the independent judge.

## 1. Coordinate frames

Three frames, in order.

| Frame | X | Y | Z |
|---|---|---|---|
| raw SUPR foot | width | anatomical height, **up** | length, heel -> toe positive |
| normalized shoe | functional heel -> toe, heel at 0 | vertical, **positive DOWN toward the sole** | width |
| original shoe | whatever the capture gave; reached via `normalized_to_shoe` | | |

`make_supr_to_shoe_axis_remap()` (alignment.py:442) is the fixed bridge:
`(x, y, z)_supr -> (z, -y, x)_shoe`. Note the Y flip. Because shoe +Y is down,
"above the ankle" is a *smaller* Y, and plantar faces are the ones whose remapped
normal points toward **+Y**.

`shoe_to_normalized` / `normalized_to_shoe` are a validated mutual-inverse 4x4
pair read from `shoe_preparation.json:normalization`. One unit of normalized
functional length == `SHOE_FUNCTIONAL_LENGTH_MM` == 262.5 mm, always, regardless
of how long the fitted foot turns out. Every mm figure below converts with that
constant.

## 2. The anchored scale (the one thing easiest to break)

`neutral_length_scale()` (alignment.py:478) computes

    scale = ANCHOR_FOOT_LENGTH_RATIO / ptp(remapped neutral template X)
    ANCHOR_FOOT_LENGTH_RATIO = 250.0 / 262.5 = 0.952381

from the **neutral, zero-beta template alone**. It is then applied to every
candidate unchanged. The consequence is the central design decision of the old
fitter:

> Foot length is an *output* of the betas, never a search variable, and the foot
> is never independently rescaled or renormalized to fit.

A torch fitter that adds a free scale parameter, or that lets `translation`
absorb a scale, silently discards this. Keep `scale` a buffer, not a parameter.

Toe allowance is the reported consequence:
`allowance_mm = 262.5 * (1 - heel_offset_x - foot_length_ratio)`.
Old acceptance band: 18-22 mm target, 20 mm centre (containment.py:43-45).
Hard reject below `MIN_TOE_ALLOWANCE_MM = 10`; running long is rejected, running
short only reports.

## 3. Plantar region and contact regions

Computed **once** on the neutral template, giving stable vertex/face indices
(`identify_supr_contact_regions`, alignment.py:720):

- plantar faces: remapped face normal `y >= cos(45deg) = 0.70711`
- plantar vertices: unique vertices of those faces
- longitudinal partition on `u = (x - x_min) / ptp(x)` of the **neutral** mesh:
  heel `<= 0.18`, arch `(0.18, 0.55)`, forefoot `[0.55, 0.80]`, toes `> 0.80`

These indices are precomputable constants for the torch path. Do not recompute
them per candidate — the old code deliberately does not.

## 4. Placement, in the order the old code applies it

1. scale (anchored, fixed) and axis remap
2. `translation_x` so the minimum plantar X lands on `heel_offset_x`
3. `translation_z` = area-weighted mean residual of plantar face centroids
   against the normalized footbed centerline, plus a free `lateral_offset_z`
4. `translation_y`:
   - zero compression -> **first contact**: `min` over all supported plantar
     vertex and centroid shifts (the foot rests on the highest point)
   - 1 mm compression -> `solve_seating()`, region-weighted, bounded by the
     same first-contact constraint

`COMPRESSION_ALLOWANCE_MM = 1.0` normalized is `1.0 / 262.5`.

## 5. Cavity: why a naive watertight SDF is wrong here

The shoe is split into **footbed** faces (contact allowed, indices from
`shoe_preparation.json:footbed_selection.original_face_indices`) and
**obstacle** faces (everything else; contact *and* proximity forbidden).

Signed clearance (`CavityEvaluator.signed_clearances`, cavity.py:1287) is not a
closed-volume SDF. It is two independent ray casts per sample:

- **upper**: origin at the footbed height directly below the sample, ray toward
  `-Y` (upward in the shoe); first obstacle hit gives the ceiling.
  `clearance = sample.y - boundary.y`. Positive == below the ceiling == inside.
- **side**: origin on the centerline at the same (x, y), ray along `+/-Z` toward
  whichever side the sample is on; `clearance = hit_distance - |offset|`.
- combined = min over the finite ones. **NaN where no boundary was found.**

That NaN is load-bearing. It encodes *open space* — the ankle collar is a real
hole, and the old code refuses to invent a boundary there. A face counts as
outside only if the conservative min over its centroid and three corners is
`< -tolerance`.

> Implication for the planned `grid_sample` SDF: the rasterized volume needs an
> explicit validity mask alongside the value, and "no boundary" must not be
> baked in as 0 or as a large positive number. Kaolin `check_sign` and any
> parity-based sign are unsafe on this geometry for the same reason.

**Correction, found by measurement after the fitter was running.** The two ray
families cover `+/-Y` and `+/-Z` only — *nothing casts along X*. The toe box and
the heel counter are therefore invisible to this clearance: on a real fit, 42%
of the faces the exact SAT test reported as colliding had **positive** signed
clearance. The old fitter is not affected because it ranks candidates on exact
collisions and only uses the signed value as a secondary term, but anything that
optimizes the signed clearance directly will happily slide the foot out through
the front of the shoe. `anatomical_coordinates` carries a second field channel (a true
signed distance to the obstacle surface) for this reason; see README section 6.1.

Separately, `_ankle_signed_exemptions` (containment.py:465) drops samples above
the ankle joint and behind the midfoot from **signed scoring only** — exact
collision stays active there.

## 6. Exact evaluation (the judge; keep it unchanged)

- triangle/triangle SAT, zero thickness, 13 axes incl. the coplanar in-plane
  edge normals; `tolerance = np.spacing(float32(max(1, max|coord|)))`
- `CavityAnalysis.status == "clear"` requires **all** of: no collision pairs, no
  excess footbed penetration, no obstacle *touching* (unsigned clearance
  `<= tol`), and no outside faces
- `SuprMeshSubdivision` (supr_foot.py:186) adds deterministic midpoint
  subdivision for denser collision sampling. The represented surface is
  unchanged; SUPR still evaluates at 266 vertices / 515 faces.

## 7. Metrics to report for any A/B against the old fitter

outside vertex/face count and area fraction, area tier, max and p95 protrusion
depth, protrusion energy, collision pair count and area fraction, per-region
plantar rms gap / median gap / coverage, seating acceptance
(`MAX_FINAL_REGION_RMS_GAP = 0.030`, `MAX_FINAL_MEDIAN_GAP = 0.015`), toe
allowance mm, `||beta||_2`, `|dtheta|`, wall time, batch throughput, GPU memory.

## 8. The gradient blocker

`SuprFootModel.evaluate` (supr_foot.py:35) wraps the call in `torch.no_grad()`
and returns `.detach().cpu().numpy()`. The official `SUPR` is already an
`nn.Module` whose forward is pure torch. The returned object carries vertices
*and* a `.J_transformed` attribute for joints. Requires CUDA — the official
implementation builds CUDA buffers unconditionally.

Active pose indices: `SUPR_ANKLE_PITCH_INDEX = 3`, `SUPR_MIDFOOT_PITCH_INDEX = 6`
of 39; 13 joints; 10 betas by default; `|beta| <= 3`.
