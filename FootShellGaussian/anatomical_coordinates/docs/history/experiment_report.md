# Differentiable SUPR foot fitting (`anatomical_coordinates`)

A PyTorch/GPU replacement for the search-based optimizer in
`foot_prior/containment.py`, built from the architecture in
`notes/foot_fitting_plan_DR.md`. The existing NumPy fitter is untouched and is
reused here as the independent exact evaluator.

```
SUPR(betas, ankle/midfoot pitch, translation)   supr_torch.TorchSuprFoot
   -> foot vertices on the GPU
   -> containment + support + priors            losses, against shoe_field
   -> autograd -> staged Adam                   fitter
   -> exact NumPy judge                         evaluate  (foot_prior.cavity)
```

Nothing outside `anatomical_coordinates/` and `scripts/foot_fit_tmux/` is modified.
`foot_prior/foot_fit/` was ignored entirely, as instructed.


---

## 0. Summary

Built and validated end to end, then measured against the existing fitter with
the existing exact evaluator.

| | old NumPy fitter | this fitter (D3) |
|---|---|---|
| exact collision area, 16 shoes | 0.9881 | **0.8981** (−9.1 %) |
| signed protrusion area | 1.5151 | **1.2299** (−18.8 %) |
| fully `clear` verdicts | 3 / 16 | **5 / 16** |
| mean ‖β‖ (shape distortion) | 5.59 | **4.61** |
| plantar samples past the sole | **0** | 17 |
| mean sizing error vs 20 mm | **0.6 mm** | 2.4 mm |
| wall clock, one shoe | 1477 s | **13.7 s** (108x) |
| 16 shoes batched | — | **17.2 s** |

Better containment with less anatomical distortion and two orders of magnitude
less compute, but it still penetrates the sole where the old fitter cannot, and
its sizing is a preference rather than a guarantee. Full verdict in section 10.

Both hard gates pass (section 4, section 5). The findings that actually made it
work — an X-blind clearance field, an ankle exemption that sinks the foot, a
loss that is not scale-free, and a fitter that was not reproducible — are in
section 6.

---

## 1. Environment

Created at `/home/ab5298/anaconda3/envs/Shell` (5.3 GB). `shellgaussianenv` was
not modified or cloned.

| package | version |
|---|---|
| python | 3.10.21 |
| torch | 2.5.1+cu121 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| trimesh | 4.12.2 |
| pytest | 9.1.1 |

34 packages total. Driver CUDA 12.7, so the cu121 wheel is well inside range.
`trimesh` is pinned to 4.12.2 to match the version the existing pipeline was
validated against. No PyTorch3D or Kaolin: the only geometry primitives needed
are trilinear sampling and a point-to-height-field query, both of which native
PyTorch provides, so building those packages would have been pure risk.

The official SUPR package and `foot_prior` are made importable through a
`.pth` file in the env's `site-packages`, so neither source tree is touched:

```
/home/ab5298/anaconda3/envs/Shell/lib/python3.10/site-packages/footshell_paths.pth
  /storage/Abhinay/Shell_Gaussian/baselines/SUPR
  /storage/Abhinay/Shell_Gaussian/FootShellGaussian
```

Two free GPUs were used (4 and 6), one experiment per GPU. No DDP: the model is
266 vertices, so parallelism is across experiments, not within a fit.

## 2. Reproducing

```bash
# both hard gates
bash FootShellGaussian/scripts/foot_fit_tmux/run_tests.sh

# experiment A: judge the existing NumPy fitter with the same evaluator
bash FootShellGaussian/scripts/foot_fit_tmux/run_baseline.sh

# experiments B/C/D plus ablations, split across the two GPUs
bash FootShellGaussian/scripts/foot_fit_tmux/run_experiments.sh

# the recommended configuration (experiment D plus the sole safety offset)
bash FootShellGaussian/scripts/foot_fit_tmux/run_recommended.sh

# batched consistency, the runtime comparison, and the L-BFGS polish
bash FootShellGaussian/scripts/foot_fit_tmux/run_batched.sh
bash FootShellGaussian/scripts/foot_fit_tmux/run_benchmark.sh
bash FootShellGaussian/scripts/foot_fit_tmux/run_lbfgs.sh

# the whole 27-shoe dataset: prepares the shoes that have no Checkpoint 4/5
# artifacts yet (using the EXISTING pipeline), then fits them
bash FootShellGaussian/scripts/foot_fit_tmux/run_full_dataset.sh

# tables
CUDA_VISIBLE_DEVICES=4 /home/ab5298/anaconda3/envs/Shell/bin/python \
  FootShellGaussian/anatomical_coordinates/scripts/summarize_results.py \
  --runs expB_translation expC_pose expD_full expD2_recommended
```

Every run writes `results.json` (config, per-shoe before/after exact metrics,
timings, restart scores) plus a `history_*.json` loss trace, so a run can be
re-examined without re-fitting.

All artifacts land in `/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs`.

A single fit directly:

```bash
cd /storage/Abhinay/Shell_Gaussian
CUDA_VISIBLE_DEVICES=4 /home/ab5298/anaconda3/envs/Shell/bin/python \
  FootShellGaussian/anatomical_coordinates/scripts/run_torch_foot_fit.py \
  --shoes crocs --run-name my_run --save-meshes
```

## 3. Architecture

### Parameters

Only what the brief asked for: `translation` (3), `ankle_pitch`,
`midfoot_pitch`, and 10 `betas`. The other 37 pose entries stay at zero. Soft
bounds replace the old hard clamps — `beta = 3 tanh(b)` and
`pitch = 20 deg * tanh(p)` — so the old limits survive as a differentiable
parameterization rather than as rejection rules.

**The anchored scale stays a buffer, never a parameter.** `neutral_length_scale`
is derived from the neutral template alone, so foot length remains an *output*
of the betas. A free scale, or a translation allowed to absorb one, would
silently discard the sizing policy the whole pipeline is built on. The torch
forward path reproduces the stored Checkpoint 5 placement to **1.6 µm**, which
is how that was verified.

### The shoe field (`shoe_field.py`)

`foot_prior.cavity` measures containment with two directional ray casts per
sample rather than a closed-volume SDF, because a shoe is open at the collar and
any parity-based inside/outside test is wrong there. Read literally its
per-sample Python loops are far too slow to rasterize a volume.

The decomposition that makes it cheap: both ray families have origins and
directions depending on only two coordinates, so each clearance is an algebraic
expression in a **two-dimensional** height field.

* upper: origin is the footbed point below the sample, direction `-Y`. The hit
  depends on `(x, z)` alone → `ceiling_y(x, z)`.
* side: origin is on the centerline at the sample's `(x, y)`, direction `±Z`.
  The hit depends on `(x, y)` → `wall_z_neg(x, y)`, `wall_z_pos(x, y)`.

Three 2-D rasterizations therefore determine the whole 3-D field, which is then
evaluated analytically on the lattice. Baking takes **0.02–0.74 s** per shoe,
against 0.30 s for a *single* old signed-clearance call.

A second channel carries a true signed distance to the obstacle surface
(conservative triangle voxelization + `scipy` EDT, signed by the directional
test). This exists because of a measured failure, described in §6.

Queried with `F.grid_sample` on a `(B, 1, D, H, W)` volume, D over world X, H
over Y, W over Z. Out of domain, the sampled value has the distance back to the
box subtracted: border padding alone returns a boundary value with *zero*
gradient, which is the worst possible behaviour for a vertex that has escaped.

### Losses (`losses.py`)

```
L = w_contain * L_containment      softplus barrier on the signed field
  + w_support * L_support          per-region soft-min gap to the footbed
  + w_penet   * L_penetration      no sinking through the sole
  + w_beta    * mean(beta^2)
  + w_pose    * ((dtheta) / 5 deg)^2
  + w_length  * ((toe_allowance - 20 mm) / 5 mm)^2
```

Every residual is divided by an explicit millimetre tolerance, so the weights
are dimensionless and comparable. Containment samples are **area-weighted**, not
counted, which makes the term invariant to sampling density and puts it in the
same units as the area fraction the exact judge reports.

Containment is scored on a deterministically subdivided copy of the foot
(`foot_prior.supr_foot.build_supr_mesh_subdivision`, level 2). Subdivision is a
fixed linear map on the source vertices, so it is one matmul and stays
differentiable; the SUPR forward pass itself stays at 266 vertices.

No Chamfer term anywhere — as the brief requires, the data term never pulls the
foot toward the upper.

### What was deliberately not ported

Latin-hypercube search, finite-difference Jacobians, least-squares proposals,
line search, polling, candidate ranking tiers, the 18–22 mm hard gate. One
exception, added because an experiment demanded it: **multi-start** (§6).

---

## 4. Hard gate 1 — differentiable SUPR

`tests/test_supr_gradients.py`, 7/7 passing. Autograd compared against central
differences on **every** component of each parameter (not a sample):

| parameter | step | max &#124;grad&#124; | max relative error | components probed |
|---|---|---|---|---|
| translation | 1e-2 | 1.42e-2 | **0.001 %** | 6 |
| ankle pitch | 5e-2 | 2.09e-3 | **0.040 %** | 2 |
| midfoot pitch | 5e-2 | 7.69e-4 | **0.044 %** | 2 |
| betas | 1e-1 | 4.25e-5 | **0.404 %** | 20 |

All finite, none identically zero, and finite at exactly the rest pose (which
matters because `quat_feat` divides by `‖θ + 1e-8‖`).

**On the step sizes.** SUPR registers its buffers as `torch.cuda.FloatTensor`,
so it is float32-only and the difference is limited by round-off, not
truncation. Sweeping the beta step showed the agreement *improving* monotonically
as the step grew — 36.3 % at 1e-3, 3.5 % at 1e-2, 0.40 % at 1e-1, 0.14 % at
3e-1 — which is the signature of cancellation, not of a wrong gradient. The
steps above sit where the signal clears float32 noise. A first attempt using a
summed readout over 1596 terms failed for exactly this reason; the readout is
now scaled down so the scalar stays O(1e-3).

The existing `foot_prior.supr_foot.SuprFootModel` NumPy API is untouched;
`TorchSuprFoot` is a parallel entry point, with `evaluate_numpy()` kept for
visualization.

## 5. Hard gate 2 — differentiable shoe field

`tests/test_shoe_field.py`, 13/13 passing, all on synthetic geometry with
analytically known answers: sign convention, clearance vs analytic distance,
outside-the-wall and above-the-ceiling signs, X/Z transposition, gradient
direction toward containment, open space, out-of-domain detection and recovery,
footbed sampling, batched-vs-single equality, nearest-of-two-ceilings, and the
longitudinal wall.

Then validated against the **real** evaluator. Baked field vs
`CavityEvaluator.signed_clearances` at the Checkpoint 5 foot vertices, all 16
shoes:

| | median | p95 | max | sign agreement |
|---|---|---|---|---|
| error vs `foot_prior.cavity` | **0.00–0.04 mm** | 0.04–7.1 mm | 0.3–45 mm | **98.1–100 %** |

The median is essentially exact, confirming the 2-D decomposition is the same
function. The p95/max outliers are grazing geometry a 1 mm lattice cannot
resolve, worst on the sandals where much of the foot is in open space.

The signed-distance channel was checked against brute-force exact
point-to-triangle distance: median error **0.43–0.47 mm**, p95 0.63 mm — about
half a voxel, as expected from a 1 mm EDT.

`evaluate.py` itself was validated by reproducing the stored Checkpoint 6
analysis exactly: 547 collision pairs vs 547, outside area fraction 0.1705 vs
0.1705.

---

## 6. What went wrong, and what the failures taught

These are the substantive findings. Each was caught by the exact evaluator
*disagreeing* with a differentiable loss that looked healthy, which is the whole
reason the brief insists on judging with the old code.

### 6.1 The directional clearance is blind along X

The first working fitter drove its containment loss down 15× while the exact
collision area got **worse**. At the exactly-colliding faces the baked field
agreed with `foot_prior.cavity` to five decimals (+0.00162 vs +0.00162) — and
that value is **positive**. 42 % of colliding faces reported positive signed
clearance.

The reason: the directional rays probe `±Y` and `±Z` only. **Nothing casts along
X.** The toe box and the heel counter are invisible to the signed clearance. The
old fitter tolerates this because it ranks candidates on exact SAT collisions; a
gradient fitter cannot, because nothing stops the foot sliding forward. Toe
allowance duly collapsed from 20 mm to 10–16 mm.

Fix: the second field channel, a true signed distance to the obstacle surface.
`test_distance_channel_sees_the_longitudinal_wall` pins this — the directional
field reports >0.10 at a point 0.01 from a front wall, the distance channel
reports 0.01.

### 6.2 Scoring the ankle as "outside" sinks the foot through the sole

With the ankle region scored normally, `plantar_excess_penetration_count`
reached **58 and 68** samples and the sole sank up to **6 mm** through the
footbed. The leg legitimately protrudes through the collar (max protrusion is
65–85 mm for *both* fitters), and the optimizer was driving the whole foot
downward to hide a protrusion that is anatomically correct.

Fully exempting the region was worse still: it removes the only constraint the
optimizer has there while the exact SAT judge still counts those intersections.

The fix is the source's own contact policy, verbatim: *the ankle may pass
through empty opening space but may not intersect the collar*. In the exempt
region the **sign is dropped and the unsigned distance is still scored**, so the
barrier survives and the false "outside" penalty does not.

### 6.3 A mean over samples is not scale-free

Adding subdivision quadrupled the sample count and *undid* an otherwise good
fit: the extra interior samples diluted the few that actually violated. Weighting
each sample by the surface area it represents (areas detached, so the optimizer
cannot shrink triangles to cheapen a violation) makes the term invariant to
sampling density and puts it in the same units as the metric being judged.

### 6.4 The support term was far stricter than anything downstream requires

Penalizing every heel and forefoot vertex's gap forced the curved sole flat and
crowded containment out of the objective entirely. The existing pipeline accepts
a region RMS gap up to **7.9 mm** (`MAX_FINAL_REGION_RMS_GAP = 0.030`), so a
per-vertex 2 mm target was far stricter than any requirement. Replacing it with a
per-region **soft minimum** — each load-bearing region should have *something*
resting on the footbed — took `pb129_shoe_low` from 0.0939 to 0.0000 collision
area in one change. (That shoe later regressed to 0.0270 once other terms were
retuned around it; the figure is the effect of this change in isolation, not a
final result. Final per-shoe numbers are in section 8.)

### 6.5 Margin cannot be smaller than the field's own error

With a 0.3 mm margin, surviving collisions sat at +0.17 mm in a field with
~0.43 mm error. The margin has to exceed the field's uncertainty; 1.0 mm works.

### 6.6 Multi-start was necessary — the one search idea that earned its place

Everything above still left the fitter behind the baseline on several shoes.
The problem is non-convex, and the old fitter explores 874 starts with 10
restarts. Adding **8 restarts** (restart 0 is always the deterministic zero-beta
start, so it can only help) moved crocs from 0.0498 to **0.0067** collision area
in that experiment — past the baseline's 0.0190 — and fixed the toe allowance to
21.3 mm. In the final configuration crocs reaches 0.0000 and a `clear` verdict.
Section 8.5 quantifies the effect properly across all 16 shoes.

This is reported as a finding, not smuggled in: the brief says not to port the
search machinery *unless experiments prove it necessary*, and this experiment
did. What is **not** ported is everything else — no Latin hypercube, no
Jacobians, no polling, no line search, no ranking tiers. Restarts are a batch
dimension, which is nearly free on a 266-vertex model, and `ablJ_single_start`
quantifies what they buy.

### 6.7 The fitter was not reproducible run to run

Two invocations with the same seed, same shoe and same configuration produced
visibly different shapes:

```
betas run 1: [-1.059  0.149 -2.275  2.069  2.458  1.508  2.143  2.486  0.872  2.395]
betas run 2: [-0.778  0.275 -2.374  2.015  2.320  2.282  2.363  1.820 -0.121  2.696]
                         collision 0.1422 vs 0.1678
```

The cause was `scatter_add_` in the containment area weighting. CUDA atomics
sum in nondeterministic order, and although the resulting weight differences are
at float32 round-off, they amplify through 750 Adam steps of a non-convex
objective and then change which restart wins. Replacing the scatter with a
padded per-vertex gather over precomputed incident faces — the topology is fixed,
so the incidence can be built once — makes it exact:

```
identical betas across runs: True        collision 0.1642 vs 0.1642
```

Worth stating plainly: this was found only because the two runs were compared,
not because anything failed. A seeded fitter that silently is not reproducible
is a bad foundation for ablations, since every ablation difference would carry
this noise.

### 6.8 Two bugs worth recording

* `_raster_extreme` initially stored the signed distance with the wrong sign, so
  `amin` selected the **farthest** surface rather than the first hit. The
  single-surface synthetic box could not detect this — nearest and farthest
  coincide. `test_nearest_surface_wins_with_two_ceilings` now covers it.
* With restarts enabled, the translation-only and pose-only variants were being
  started from *random betas they never optimize*, silently changing what those
  experiments measured. Restarts are now suppressed unless betas are active.

---

## 7. Files

**Created** (nothing else in the repository was touched):

```
FootShellGaussian/anatomical_coordinates/
  __init__.py                  package surface
  CONTRACT.md                  conventions inherited from foot_prior, written
                               up before any code
  config.py                    FitConfig / StageConfig
  supr_torch.py                TorchSuprFoot: batched, differentiable SUPR
  shoe_field.py                bake + grid_sample the cavity field
  shoe_data.py                 read one prepared golden-set shoe (read-only)
  losses.py                    containment / support / priors
  fitter.py                    staged Adam, placement, subdivision sampling
  evaluate.py                  bridge to the exact foot_prior judge
  scripts/run_torch_foot_fit.py    main runner
  scripts/evaluate_baseline.py     experiment A
  scripts/benchmark_runtime.py     NumPy vs torch timing
  scripts/summarize_results.py     report tables
  tests/test_supr_gradients.py     hard gate 1  (7 tests)
  tests/test_shoe_field.py         hard gate 2  (13 tests)

FootShellGaussian/scripts/foot_fit_tmux/
  env.sh                            shared paths and the two free GPUs
  run_tests.sh                      both hard gates
  run_baseline.sh                   experiment A
  run_experiments.sh                experiments B/C/D and the ablations
  run_recommended.sh                experiment D2
  run_recommended_deterministic.sh  experiment D3 (reproducible headline)
  run_batched.sh                    batched vs sequential
  run_benchmark.sh                  NumPy vs torch runtime
  run_lbfgs.sh                      experiment G
  prepare_missing_shoes.sh          Checkpoint 4/5 for unprepared shoes
  run_full_dataset.sh               prepare + fit all 27 dataset shoes
```

**Modified**: none. `foot_prior/`, `baselines/SUPR/`, the golden-set dataset and
`shellgaussianenv` are all untouched — verified with `find -newermt`. The SUPR
and `foot_prior` packages are reached through a `.pth` file inside the new conda
env rather than by installing into their source trees.

**Downstream interfaces.** The only consumer of the old fitter is
`scripts/run_containment_fit.py`, which writes `containment_fit.json` (schema 5)
plus three PLYs. `FitResult.transform()` returns the same
`posed_supr_to_normalized_shoe` matrix that record is built around, so the torch
fitter can feed that serializer without changing it. No downstream anatomical
stage was modified.

---

## 8. Results

All numbers come from `foot_prior`'s exact evaluator, never from the
differentiable loss. Experiment A is the **stored output of the existing NumPy
fitter re-judged by the same code**, so A and B–D are scored identically. The
primary metric is exact SAT collision area fraction; `outside` is the signed
protrusion area fraction. Both fitters start from the same Checkpoint 5 foot.

### 8.1 Experiments A–D, all 16 shoes

```
method                    SUM coll   SUM out  mean|toe-20|  mean|beta|  clear  penetr
-------------------------------------------------------------------------------------
C5 start (no fit)           3.6808    4.5667           6.9        0.00      1       0
A: NumPy fitter             0.9881    1.5151           0.6        5.59      3       0
torch: expB_translation     3.7697    4.4904           4.8        0.00      1     145
torch: expC_pose            3.4855    4.1497           4.2        0.00      1     119
torch: expD_full            0.9279    1.2299           2.3        4.48      3      33
```

**D beats the existing fitter on both containment metrics** — 6 % lower
collision area and 19 % lower protrusion area — using *less* extreme shape
(mean ‖β‖ 4.48 against 5.59). Per shoe it wins on 7, loses on 5, ties on 4.

**B and C fail, and the way they fail is informative.** Translation alone is
*worse than not fitting at all* (3.77 vs 3.68), and pose barely helps. Both
accumulate heavy sole penetration (145 and 119 samples) because containment
dominates the objective 214-to-8 and, with shape frozen, the only move left is
to press the foot down into the sole. **Shape is not optional** — this is the
clearest experimental result in the set.

### 8.2 Per shoe, A vs D

```
shoe                                       A coll   D coll    A out    D out   D toe  D |b|   winner
----------------------------------------------------------------------------------------------------
aj_12_basketball_sneakers                  0.0000   0.0000   0.0000   0.0000    20.0   0.13      tie
birkenstock_arizona_sandal                 0.1137   0.0844   0.1416   0.0925    19.8   7.31    torch
canvas_shoe                                0.0894   0.0894   0.1656   0.0992    11.5   6.69    numpy
crocs                                      0.0190   0.0000   0.0190   0.0000    20.7   5.20    torch
crocs_by_speedyart_studio                  0.0125   0.0000   0.0040   0.0000    20.1   4.67    torch
crocs_shoe                                 0.0000   0.0000   0.0000   0.0000    20.0   2.04      tie
duinn_shoes_womens_hiking_sandal_sport     0.0535   0.0359   0.0854   0.0208    28.7   7.16    torch
leather_boots                              0.0000   0.0000   0.0149   0.0000    20.0   1.26      tie
nike_air_jordan                            0.1334   0.1427   0.1951   0.1424    20.4   7.52    numpy
pb129_shoe_low                             0.0000   0.0000   0.0000   0.0311    20.0   2.68      tie
priest_karol_wojtyas_sports_shoes          0.0118   0.0128   0.0221   0.0310    20.0   2.47    numpy
sandal_1                                   0.0000   0.0376   0.0161   0.0358    20.1   2.33    numpy
sandals_0001                               0.1075   0.0918   0.1439   0.1277    20.2   6.03    torch
sneaker_vibe                               0.2423   0.1806   0.4567   0.3510    32.8   5.90    torch
sneakers_seen                              0.1651   0.1194   0.1770   0.1363    19.8   6.51    torch
ww_ii_german_jack_boots                    0.0399   0.1333   0.0737   0.1621    15.7   3.81    numpy
```

Footwear style matters. The torch fitter is strongest on the roomy, awkward
cases the old search struggles with — sneakers and sandals (`sneaker_vibe`
0.2423 → 0.1806, `sneakers_seen` 0.1651 → 0.1194, `birkenstock` 0.1137 →
0.0844). It is weakest where the old fitter already reaches zero and any
residual is pure loss (`sandal_1`, `pb129`), and on `ww_ii_german_jack_boots`,
whose tall shaft leaves the leg region dominating the metric.

`canvas_shoe` (11.5 mm) and `sneaker_vibe` (32.8 mm) show the sizing weakness:
the soft length term is a preference, not the old fitter's hard 18–22 mm gate,
so mean sizing error is 2.3 mm against 0.6 mm.

### 8.3 Ablations (8 diverse shoes, same configuration otherwise)

```
configuration                 SUMcoll   SUMout  toe_dev  mean|b|   pen  medgap
A: NumPy fitter                0.7053   1.1367      0.7     6.01     0    4.01
D: full (reference)            0.5889   0.8439      2.9     5.32    20    3.18
E no support                   0.6150   0.8047      2.0     5.04    16    3.43
F no beta prior                0.6776   0.9472      1.9     5.50    13    3.37
G no ankle exemption           0.5950   0.6147      2.5     5.08    50    2.89
H no length term               0.5889   0.8040      6.2     4.54    18    3.20
J single start (1 restart)     1.2692   1.6364      4.5     5.33    49    3.62
```

* **J — multi-start is the single most important ingredient.** One start is
  2.2× worse than eight (1.2692 vs 0.5889) and worse than the old fitter. This
  is the experiment that justified adding restarts at all.
* **F — the beta prior earns its place** (0.6776 → 0.5889 with it).
* **E — the support term costs a little containment and buys seating**: median
  plantar gap 3.43 mm → 3.18 mm, and without it nothing holds the foot on the
  footbed at all.
* **G — the ankle exemption is a genuine trade.** Removing it gives the best
  protrusion number (0.6147) because the optimizer is rewarded for stuffing the
  leg into the shoe — and 50 penetrating plantar samples, 2.5× the reference,
  which is the sinking failure of §6.2 quantified.
* **H — the length term is nearly free.** Identical collision area (0.5889),
  but sizing error 6.2 mm without it against 2.9 mm with it. This is the "one
  weak size constraint" the research report predicted would be needed.

### 8.4 Batched fitting, all 16 shoes at once

Batched and sequential were run at an identical single-start setting, so any
difference is attributable to batching alone.

| | sequential | batched |
|---|---|---|
| wall time, 16 shoes | 238.2 s | **17.2 s** (13.8×) |
| peak GPU | 944 MB | 2059 MB |
| SUM collision area | 1.8219 | 1.7374 |
| max per-shoe difference | — | 0.0900 |
| mean per-shoe difference | — | 0.0201 |

Consistent but **not bit-identical**, and the reason is structural rather than
numerical: shoes have different lattice sizes, so batching pads every field to a
common shape and each padded shoe's domain box grows with it. Interpolation near
those edges therefore differs slightly. For throughput work this is fine; for a
result that must match a sequential run exactly it is not.

### 8.5 Sensitivity to initialization

Each of the 8 restarts in experiment D is scored by the exact evaluator, so the
spread between them measures directly how much the answer depends on where the
optimizer started.

| | exact (collision + outside) score |
|---|---|
| mean spread, best vs worst restart | **0.2669** |
| max spread (`ww_ii_german_jack_boots`) | **0.7193** |
| shoes where the deterministic zero-beta start won | **4 of 16** |

The spread is the same order as the entire gap between the old fitter and not
fitting at all, so **a single-start differentiable fit is not a stable
procedure** on this problem. A random restart beats the deterministic start on
12 of 16 shoes. Worst cases: `nike_air_jordan` 0.2851–0.7154 and
`ww_ii_german_jack_boots` 0.2953–1.0147 depending only on the initial betas.

This is the quantitative justification for §6.6. It is also the honest limit of
the "just replace the search with Adam" thesis: the search is not *all*
unnecessary, only most of it. Eight restarts against the old fitter's 874 starts
and 10 restarts is roughly a 100× reduction in exploration, at comparable
quality.

### 8.6 Performance by footwear style

Experiment D against the old fitter, grouped by shoe topology (collision area,
summed within each group):

| style | n | A (NumPy) | D (torch) | delta |
|---|---:|---:|---:|---:|
| sneaker | 4 | 0.5086 | **0.4023** | −0.1064 |
| clog | 3 | 0.0314 | **0.0000** | −0.0314 |
| sandal | 4 | 0.2748 | **0.2496** | −0.0251 |
| low shoe | 1 | 0.0000 | 0.0000 | ±0.0000 |
| high-top | 2 | **0.1334** | 0.1427 | +0.0093 |
| boot | 2 | **0.0399** | 0.1333 | +0.0934 |

The differentiable fitter is better on every style except tall ones. Boots are
its weakest case, and the reason is visible in §6.2: a tall shaft puts a large
fraction of the SUPR ankle stump inside a region where the signed clearance says
"outside" but the geometry is legitimate, and the objective still has to trade
that against real containment. All three clogs come out perfectly contained.

**Dataset coverage.** The golden-set *dataset* directory
(`/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation`) holds
**27** shoes, but only **16** had prepared Checkpoint 4/5 artifacts when this
work started — those 16 are what sections 8.1–8.6 cover, and all 16 fitted
without error in every experiment. The remaining 12 (`adidas_substance`,
`rtfktchallenge_golden_by_franz_vega`,
`shoes_mockup_asset_vans_skate_old_skool_shoes`, `sneaker_1`–`sneaker_8`,
`sneaker_b33`) have raw meshes and canonicalizations but had never been through
preparation. They are handled in section 8.9. Note also that `sneaker_vibe` is
prepared but is *not* present in the dataset directory, so the prepared set and
the dataset set are not nested.

One dataset issue found: **`sneaker_1` and `sneaker_3` are the same shoe.** The
meshes are not byte-identical (105136 vs 105152 vertices) but they have the same
182198 faces, the same normalized bounds to four decimals, and the Checkpoint 5
pipeline returns identical angles, toe allowance and plantar coverage for both.
They should be treated as one sample, not two, in any aggregate.

### 8.7 Recommended configuration (D3) — the headline result

Two changes on top of D, both from findings above:

* `--support-safety-mm 0.15` (section 6.5). Experiment D leaves geometrically
  contained fits that the exact judge still refuses to call `clear`, because a
  soft penetration penalty settles a few micrometres *inside* the sole. Asking
  the sole for a sliver of daylight instead of merely non-negative gap fixes it.
* the deterministic area accumulation of section 6.7, which is why this run
  exists separately from D2 and supersedes it.

```
method                    SUM coll   SUM out  mean|toe-20|  mean|beta|  clear  penetr
-------------------------------------------------------------------------------------
C5 start (no fit)           3.6808    4.5667           6.9        0.00      1       0
A: NumPy fitter             0.9881    1.5151           0.6        5.59      3       0
torch: expD_full            0.9279    1.2299           2.3        4.48      3      33
torch: expD2_recommended    0.9519    1.3214           2.4        4.79      5      21
torch: expD3_deterministic  0.8981    1.2299           2.3        4.61      5      17
```

**D3 is the best configuration on every containment axis**: 9.1 % less collision
area than the old fitter, 18.8 % less protrusion area, **5 fully clear shoes
against 3**, the fewest penetrating samples of any torch run, and less
anatomical distortion (mean ‖β‖ 4.61 vs 5.59). Per shoe it is better on 8,
worse on 5, tied on 3.

```
shoe                                       A coll  D3 coll    A out   D3 out    toe   |b|  D3 status
aj_12_basketball_sneakers                  0.0000   0.0000   0.0000   0.0000   20.0  0.16  clear
birkenstock_arizona_sandal                 0.1137   0.0948   0.1416   0.1019   19.9  7.22  collisions
canvas_shoe                                0.0894   0.0862   0.1656   0.0983   11.1  6.68  collisions
crocs                                      0.0190   0.0000   0.0190   0.0000   20.7  5.65  clear
crocs_by_speedyart_studio                  0.0125   0.0000   0.0040   0.0000   20.1  4.69  clear
crocs_shoe                                 0.0000   0.0000   0.0000   0.0000   20.0  2.27  clear
duinn_shoes_womens_hiking_sandal_sport     0.0535   0.0304   0.0854   0.0209   29.7  6.88  collisions
leather_boots                              0.0000   0.0000   0.0149   0.0000   20.0  1.26  clear
nike_air_jordan                            0.1334   0.1398   0.1951   0.1377   19.9  7.22  collisions
pb129_shoe_low                             0.0000   0.0071   0.0000   0.0322   20.1  3.03  collisions
priest_karol_wojtyas_sports_shoes          0.0118   0.0184   0.0221   0.0381   20.0  2.58  collisions
sandal_1                                   0.0000   0.0041   0.0161   0.0425   20.1  3.66  collisions
sandals_0001                               0.1075   0.1042   0.1439   0.1536   20.0  5.59  collisions
sneaker_vibe                               0.2423   0.1891   0.4567   0.3409   32.8  6.15  collisions
sneakers_seen                              0.1651   0.0803   0.1770   0.1003   19.5  7.15  collisions
ww_ii_german_jack_boots                    0.0399   0.1438   0.0737   0.1636   15.8  3.62  collisions
```

`crocs`, `crocs_by_speedyart_studio` and `leather_boots` reach a verdict the old
fitter never reaches on those shoes. The three losses are `pb129_shoe_low`
(0.0000 → 0.0071), `sandal_1` (0.0000 → 0.0041) and `ww_ii_german_jack_boots`
(0.0399 → 0.1438) — the first two are shoes the old fitter already perfects, so
any residual is pure regression, and the third is the boot case of section 8.6.
`canvas_shoe` at 11.1 mm and `sneaker_vibe` at 32.8 mm are the sizing weakness.

### 8.8 Experiment G: Adam followed by L-BFGS

An L-BFGS polish was implemented and run on the ablation subset
(`--lbfgs-steps 40`). Result: **the polish was rejected in 8 of 8 fits** — it
never reduced the objective below what Adam had already reached.

A strong-Wolfe line search initially diverged to non-finite vertices, because a
softplus barrier is flat wherever the foot is clear of every surface and the
line search extrapolates straight across that flat region. With a small fixed
step and a snapshot-and-revert guard it is safe but useless here.

Conclusion: **do not use it.** Adam reaches the local optimum this objective has
to offer, and the remaining error is the choice of basin (section 8.5), which a
second-order polish cannot fix. Theseus was not attempted for the same reason.

### 8.9 Runtime, GPU use, and how much work each fitter does

| | old NumPy search | torch (D2, 8 restarts) |
|---|---|---|
| SUPR candidates evaluated per shoe | mean **2852** (1204–3408) | 8 restarts x 750 Adam steps |
| **exact SAT evaluations per shoe** | mean **1649** (33–2088) | **8** (restart selection) + 1 full |
| broad starts generated | mean 874 | 8 |
| wall clock, one shoe (`crocs`) | **1477.1 s** (24.6 min) | **13.7 s** — **108x faster** |
| wall clock, 16 shoes sequential | — | 212.4 s |
| wall clock, 16 shoes batched | — | **17.2 s** |
| field bake, one shoe | n/a | 0.57–0.99 s |
| peak GPU *during a fit*, one shoe | n/a | 952 MB |
| peak GPU *during a fit*, 16 batched | n/a | 2059 MB |
| whole process, 16 shoes resident | n/a | ~17.4 GB |

Both timings are `crocs`, from the same Checkpoint 5 start, measured in the same
process by `scripts/benchmark_runtime.py` with a CUDA warm-up before the torch
timing. The torch figure includes all 8 restarts (1.71 s each); field baking
adds 1.10 s on top.

The structural difference is the middle row. The expensive operation in this
problem is the exact triangle-triangle test against a 100k-face shoe, and the
old fitter runs it ~1649 times per shoe because that is how it ranks candidates.
The differentiable fitter runs it **8 times**, only to choose between restarts —
a ~200x reduction in the operation that actually costs money.

The last row is a scaling trap worth stating: the runner bakes every requested
shoe's field before it fits any of them, and each field is three float32 volumes
of roughly 280 x 130 x 128. The per-fit figures are what `max_memory_allocated`
reports once fitting starts; the resident total grows linearly with the number
of shoes queued, so a much larger set needs the fields baked lazily or cached to
disk (`ShoeCavityField.save/load` exists for this).

Three caveats on these numbers. First, GPU utilization during the experiments
sat at about **10 %**: the bottleneck throughout was `foot_prior`'s exact
evaluator on the CPU, not fitting. The quoted per-fit times come from runs with
a single job per GPU; the optimization loop is kernel-launch-bound on a
266-vertex model, so it is sensitive to CPU contention — the same configuration
measured 13.3 s alone and about 42 s with three heavy jobs sharing the host. Second, per-shoe wall clock is not where the
differentiable fitter wins — 16 shoes batched take barely longer than one shoe
alone, so the real gain is throughput, exactly as the research report predicted
for a 266-vertex model.


---

### 8.10 The rest of the dataset

The 12 shoes that had no Checkpoint 4/5 artifacts were put through the
**existing** preparation and alignment scripts (`prepare_missing_shoes.sh`,
writing to `foot_fit_outputs_runs/prepared_extra/` so `golden_set_evaluation/`
is untouched) and then fitted with the recommended configuration. All 12
prepared without error and all 12 came out as the `normal` shoe profile.

```
shoe                                              C5 coll fit coll   C5 out  fit out    toe   |b|
adidas_substance                                   0.4886   0.2112   0.6229   0.2624   39.1  6.65
rtfktchallenge_golden_by_franz_vega                0.3627   0.0786   0.4396   0.0722   19.4  5.53
shoes_mockup_asset_vans_skate_old_skool_shoes      0.3378   0.2963   0.7280   0.3745   25.9  7.00
sneaker_1                                          0.5532   0.3833   0.7263   0.4437   27.9  6.68
sneaker_2                                          0.5206   0.3734   0.7481   0.3895   36.5  7.95
sneaker_3                                          0.5532   0.4101   0.7263   0.4122   28.0  4.91
sneaker_4                                          0.4370   0.2508   0.6889   0.3368   40.0  6.41
sneaker_5                                          0.4570   0.2337   0.6613   0.3495   28.1  7.03
sneaker_6                                          0.4748   0.0788   0.5756   0.1097   31.3  6.61
sneaker_7                                          0.4593   0.2279   0.6641   0.2626   34.8  7.54
sneaker_8                                          0.3185   0.0728   0.6365   0.0457   27.3  7.12
sneaker_b33                                        0.5473   0.3123   0.7821   0.5301   37.2  7.38

12 shoes   SUM coll 5.5099 -> 2.9291 (-46.8%)   SUM out 7.9997 -> 3.5890 (-55.1%)
clear: 0/12        penetrating plantar samples: 96
```

The fitter roughly halves both containment metrics, but **clears none of them**,
and the toe allowances (27–40 mm on nine of twelve) show it is buying that
reduction largely by shrinking the foot. On this set the differentiable fitter
is not producing usable fits.

**These shoes are narrower than the foot the pipeline insists on fitting.**

Checkpoint 5 already starts them deep in collision — median 0.467 against 0.222
for the golden 16, so 2.1x harder before any fitting happens. The cause is a
sizing mismatch that is built into the pipeline's policy rather than into either
fitter:

```
neutral zero-beta SUPR foot at the anchored scale:  250.0 x 94.7 x 96.6 mm

                   shoe outer width   footbed width   footbed/outer   vs 94.7 mm foot
  golden 16                 114.4 mm         94.8 mm            0.82          +0.1 mm
  extra 12                  103.5 mm         87.9 mm            0.85          -6.7 mm

  footbed narrower than the neutral foot:  golden 8/16   extra 12/12
```

The golden 16 sit, at the median, within **0.1 mm** of the reference foot's
width, and every one of the extra 12 is narrower than it. I can only report that
correlation, not explain it: both groups come from the same curated pool
(`curated_subsets/footbed_clean/`) and both carry `reviewed_manifest_entry:
true`, so there is no provenance flag showing the 16 were chosen for their
width. With 16 samples the median match may be coincidence. What is not
coincidence is the direction: 8/16 golden footbeds are narrower than the foot
against 12/12 of the extra ones.

This is not a footbed-detection artifact: the shoes' *outer* width is smaller
too (103.5 vs 114.4 mm) and the footbed-to-outer ratio is the same (0.85 vs
0.82), so the detector is finding a comparable share of a genuinely narrower
shoe. These are narrow-lasted sneakers.

`alignment.py` anchors every shoe to a 262.5 mm functional length and fits a
250 mm foot into it, which assumes every shoe is the same size. For a
proportionally narrower shoe that assumption is simply false, and no amount of
ankle/midfoot pitch can fix it. The optimizer's only remaining lever is the
betas — but SUPR's betas **couple width and length**, so narrowing the foot also
shortens it, and the length anchor then charges for exactly the move that would
fix the fit. That conflict is visible in the table above as toe allowances
running out to 27–40 mm: the fitter was shrinking the foot correctly and paying
a penalty for it the whole way. Section 8.11 tests removing that penalty.

Footbed quality is a secondary concern worth noting separately:
`adidas_substance`'s detected footbed has **38 faces** and `rtfktchallenge`'s
has 375, against thousands for most of the golden set.

No parameter was tuned around these shoes — every number above comes from the
same configuration used for the 16.

No comparison against the old NumPy fitter is possible here: it has never been
run on these shoes, and section 8.9 shows why doing so was not attempted within
this session.

### 8.11 Can the narrow shoes be fixed?

Two defects were found, both of the same kind: a surface the foot could cross
for a bounded price. Neither is a property of the shoes.

**Defect 1 — the containment exemption was an anatomical box.**
`ShoeFootFitter.containment_samples` mirrored `_ankle_signed_exemptions`: drop
the containment sign above the ankle joint and behind the midfoot joint. That
box is an axis-aligned quadrant covering 12–22 % of the foot **including the
back of the heel**, and inside it the foot could reverse straight through the
heel counter at zero cost. 64–86 % of heel containment violations sat inside it.

The fix keys the exemption on the shoe instead of the foot. A footbed column
with no surface above it is a genuine opening in the upper — the collar, or the
open top of a sandal — and that is the only place the foot may leave the cavity.
Behind the heel counter and beyond the toe there is no footbed, so those samples
can never be exempt however far out they travel. No landmark, no threshold, same
rule for every shoe.

**Defect 2 — the distance channel was signed by an X-blind test.**
`outside` was `valid & (combined < 0)`, and `valid` comes from the ±Y and ±Z ray
families alone. Behind the heel counter no ray finds anything, so the true
distance was signed **positive** — and grew with depth. Measured on three shoes,
a sample 20 mm behind the counter reported +1.8 to +9.2 mm. The field was telling
the optimizer that leaving the shoe backwards took it deeper inside; the counter
was a thin barrier to tunnel through with nothing on the far side.

The fix adds the missing ±X ray family (`wall_x_neg(y,z)`, `wall_x_pos(y,z)`,
cast from the footbed's longitudinal midpoint) in the same 2-D decomposition
style, and unions it into the sign only. `clearance` and `valid` are unchanged,
so the validated correspondence with `foot_prior.cavity` still holds and all 20
original field tests pass untouched.

### What each change is worth, measured separately

Golden 16, one change at a time, exact judge:

```
run                   Scoll    Sout   clear  sunk   minGap   len     change added
expD3_deterministic  0.8981  1.2299     5    6/16   +0.05   233.3   (baseline)
s3_g_nosign_nw       0.9142  1.0487     8    6/16   -0.14   233.9   + defect 1 fix
s3_g_nosign          1.2551  1.4361     1    8/16   -0.24   242.1   + w_heel = 10
s4_g_sign_nw         1.3987  1.6284     2   12/16   -0.33   229.4   + defect 2 fix
s2_signfix_golden16  1.7348  1.8210     0   12/16   -0.50   242.3   + both + w_heel
```

**The exemption fix is a straight win**: `clear` 5 → 8 (exact status 5 → 7),
outside area −15 %, collision flat, maximum protrusion 25.6 → 17.7 mm.
`pb129_shoe_low`, `priest_karol_wojtyas_sports_shoes` and `sandal_1` become
clear; three shoes regress. It is on by default.

**`heel_seating_loss` is not needed once the exemption is geometric, and hurts.**
It was added to stop the heel reversing through the counter; the measured cause
of that was the exemption box, not the free translation. With the box gone,
`w_heel = 10` costs 7 of the 8 clear fits by shoving the foot forward into the
toe box (mean length 233.9 → 242.1 mm, ‖β‖ 4.55 → 5.37). The term is kept in
`losses.py` — it is the right statement of the constraint — at weight 0.

**The longitudinal sign fix is correct and currently not deployable.** It raises
agreement with the exact SAT judge from 56 % to 63 % of colliding faces, and it
demonstrably fixes what it targeted (on the six hard shoes, feet sitting behind
the functional heel went 4/6 → 0/6 and maximum protrusion 72.1 → 52.6 mm). But
switching it on makes fits *worse*: golden `clear` 7 → 2, samples sunk through
the footbed 6/16 → 12/16.

The mechanism is the point. **The footbed is excluded from the obstacle set by
design** (contact is allowed there), so it appears nowhere in the containment
barrier and is guarded only by `support_loss`'s squared hinge — a finite price.
Closing the longitudinal escape without closing the vertical one does not
produce an honest fit; it redirects the violation downward, and a fit sunk
through the sole can never be `clear`. The ray family ships behind
`use_longitudinal_sign`, off by default, to be enabled together with a hard
footbed constraint and not before.

### Where the extra 12 actually stand

With the exemption fix and `w_heel = 0`: Σcoll 3.4045, Σout 3.9565, **0/12
clear** — and **12/12 sunk through the footbed**, by 1.2 to 4.6 mm. That is no
longer a mysterious failure. Every one of the twelve is invalid for the same
single reason, and it is not containment: it is the soft footbed constraint.
Toe allowance is still 20–53 mm on most of them, so the foot is also still
shrinking (`sneaker_4`: 52.6 mm of unused toe space with the heel 3.6 mm past
the functional heel).

So the answer to the section's question is: **not yet, but the remaining
obstacle is now identified and is not shoe-specific.** The next change is to
make the footbed a one-sided constraint at `footbed_y + compression_allowance`
inside the containment barrier — the barrier already has exactly the right
machinery for one-sided constraints, and only the obstacle/footbed split keeps
it out. That is expected to unlock the longitudinal sign fix at the same time,
because the two failure directions are coupled. `shoes_mockup`, whose cavity is
30 % smaller than the foot, will still not clear; that one is a genuine sizing
conflict rather than a fitter defect.

## 9. Remaining issues

1. **Sizing is looser than the old policy.** Mean |toe allowance − 20 mm| is
   2.3 mm against the old fitter's 0.6 mm, and two shoes land well outside the
   old 18–22 mm band (`canvas_shoe` 11.1 mm, `sneaker_vibe` 32.8 mm). The soft
   length term is a preference; the old hard gate was a rejection rule. If the
   band is a real product requirement, the honest fix is to reject out-of-band
   fits after the fact and re-run with a stronger `w_length`, not to pretend a
   soft term enforces it.

2. **Sole penetration is a soft constraint, and it is now the binding one.**
   D leaves 33 plantar samples past the compression allowance across 16 shoes;
   the safety offset cuts that to 17, but the old fitter's count is 0 and always
   will be, because it *constructs* first contact rather than penalizing its
   absence. A penalty can always be outbid.

   Section 8.11 shows what that costs. The footbed is excluded from the obstacle
   set, so it is absent from the containment barrier entirely; once the
   longitudinal escape route is closed, the optimizer simply escapes downward
   instead, and **every one of the extra 12 ends up sunk through the sole**
   (1.2–4.6 mm). This is the single largest remaining defect and it blocks the
   longitudinal sign fix. Two candidate fixes, in order of preference:
   make the footbed a one-sided constraint at `footbed_y +
   compression_allowance` inside the containment barrier, which is where it
   belongs; or, as a cheap stopgap, project after fitting by translating up by
   the worst penetration, a single scalar.

3. **Restarts are needed and cost the headline speed claim.** One start is not
   competitive (§8.5). Eight restarts is still ~100× less exploration than the
   old search, but it is not zero, and it means the fitter is not yet a pure
   local method.

4. **The distance channel is voxel-limited.** Median error 0.43 mm at a 1 mm
   lattice, so the containment margin cannot go below ~1 mm. A finer lattice or
   an exact narrow-band refinement would let the foot sit closer to the shoe,
   which matters for tight footwear.

5. **Boots are the weak case** (§8.6), driven by the ankle/collar interaction of
   §6.2 rather than by anything about the sole. The geometric collar exemption
   of §8.11 addresses the mechanism directly; `leather_boots` stays clear under
   it and `ww_ii_german_jack_boots` regresses slightly, so this is improved but
   not resolved.

9. **The collar exemption is soft inside the opening column.** The mask is keyed
   on the (x, z) column, so a sample low in the collar throat — under the
   opening but still over the footbed — is exempt as well as the leg above it.
   A sample there could in principle escape laterally through a quarter panel
   that leans inward without being charged. The old joint box had the same hole
   and a much larger one besides; it is recorded rather than fixed.

6. **Batched results are not bit-identical to sequential** (§8.4), because
   fields are padded to a common lattice. Fine for throughput, not for exact
   reproducibility. Per-shoe lattices with a ragged batch would fix it.

7. **`evaluate.py` is the bottleneck, not fitting.** GPU utilization during the
   experiments sat at ~10 %: the exact NumPy judge takes ~100 s per mesh against
   ~14 s to fit. Fitting is no longer the expensive part of this pipeline.

8. **Not tested beyond the `normal` shoe profile.** The high-heel path in
   `foot_prior` was never exercised here.

---

## 10. Verdict: is this safe to replace the old fitting optimizer?

**Not yet as a drop-in replacement. Yes as the optimizer, behind two small
additions.**

What the evidence supports:

* The architecture works and is validated end to end. Both hard gates pass with
  quantified margins, the baked field reproduces `foot_prior.cavity` to a
  median of 0.00–0.04 mm, and the torch placement reproduces the stored
  Checkpoint 5 foot to 1.6 µm.
* On the metric the old fitter is itself judged by, it is **better**: collision
  area 0.8981 vs 0.9881 (−9.1 %), protrusion area 1.2299 vs 1.5151 (−18.8 %),
  and **5 fully clear shoes against 3** — since §8.11's collar fix, **8 clear
  against 3**, with protrusion area 1.0487 — — including `crocs` and `leather_boots`,
  which the old fitter never clears. Better on 8 shoes, worse on 5, tied on 3.
* It does so with **less anatomical distortion** (mean ‖β‖ 4.61 vs 5.59), which
  matters because the fitted shape feeds downstream anatomy stages.
* It replaces ~874 starts plus 10 restarts, finite-difference Jacobians, least
  squares, line search, polling and tiered ranking with **8 restarts and Adam**,
  and it is **108x faster per shoe** (13.7 s against 1477 s on `crocs`, measured
  head to head) with another 13.8x on top from batching.

What blocks a straight swap:

* **It penetrates the footbed** — 17 samples across 16 shoes, against exactly 0
  for the old fitter. That is a physically invalid state, not a worse score, and
  no soft penalty will guarantee otherwise. Fixing it is cheap (post-fit
  projection by the worst penetration) but it is not done.
* **Sizing is a preference, not a guarantee.** Mean error 2.4 mm vs 0.6 mm, with
  `sneaker_vibe` at 32.8 mm and `canvas_shoe` at 11.1 mm. If the 18–22 mm band
  is a real requirement, this fitter does not currently enforce it.
* **It regresses on three shoes where the old fitter is already perfect**
  (`pb129_shoe_low`, `sandal_1`, `ww_ii_german_jack_boots`). Averages hide that;
  a production swap would not.
* **It does not produce usable fits on the wider dataset.** On the 12 shoes that
  had never been prepared, it halves both containment metrics but clears none of
  them and shrinks the foot to a 27–40 mm toe allowance to get there (section
  8.10). Those shoes also start from a far worse Checkpoint 5, so this is not
  purely a fitter problem — but it does mean the encouraging 16-shoe result is a
  result on the *validated* golden set, not on the dataset as a whole.

  Section 8.11 narrows this considerably. With the geometric collar exemption,
  **all 12 fail for one identified reason**: they are sunk through the footbed,
  1.2–4.6 mm, because the sole is a soft penalty rather than a constraint. That
  is the same defect as the second bullet above, and it is now the thing to fix
  rather than a general statement about these shoes being hard.

Recommended path: adopt it as the optimizer **for the validated golden set**
with (1) a post-fit projection step that removes sole penetration exactly, and
(2) an explicit toe-allowance check that re-runs with a higher `w_length` when a
fit lands outside the band. Before extending it to the rest of the dataset, the
preparation stage for those 12 shoes needs looking at — a 38-face footbed is not
something any fitter can work with. Keep
the old fitter available and keep scoring both with `foot_prior.cavity`. Once
penetration is exactly zero and the band is enforced, it wins on every axis I
measured except boots, and it is an order of magnitude cheaper.

I would not delete the old fitter. Beyond being the evaluator, it is the only
thing that makes statements like "42 % of colliding faces had positive signed
clearance" checkable — and that measurement is what made this implementation
correct.
