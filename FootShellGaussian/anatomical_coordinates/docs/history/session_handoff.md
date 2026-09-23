# Context Transfer — `anatomical_coordinates`

**Written for a fresh Claude Code session picking this up cold.**
Everything below was verified against disk on 2026-09-21 unless marked
*(recalled)*. Read this file first, then `README.md` (the deliverable report)
and `CONTRACT.md` (inherited coordinate/semantic conventions).

---

## 0. One-paragraph orientation

We replaced a slow search-based NumPy foot fitter (`foot_prior/containment.py`)
with a differentiable GPU pipeline: **SUPR → batched vertices → baked shoe
cavity field → soft containment/contact losses → autograd → staged Adam**. It
works and beats the old fitter on the 16 "golden" shoes. It does **not** produce
clean fits on 12 harder shoes, and the user has asked for a good, robust fit on
**all** of them. One concrete defect in the loss has been identified but **not
yet fixed** — that is the immediate next task (§7).

---

## 1. Hard constraints (user-imposed, still in force)

| Rule | Detail |
|---|---|
| Write scope | **Only** inside `/storage/Abhinay/Shell_Gaussian/FootShellGaussian`. Nothing outside it. |
| Extra writable dir | `/home/ab5298/anaconda3/envs/` (the `Shell` env lives there, already built). |
| Work dir | All new code in `FootShellGaussian/anatomical_coordinates/`. |
| Artifacts | All outputs to `/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs`. Never write run artifacts into the repo. |
| Ignore | `FootShellGaussian/foot_prior/foot_fit/` — do not read, do not touch. |
| GPUs | At most **2**, and they are **GPU 4 and GPU 6** (`FF_GPU_A`/`FF_GPU_B` in `env.sh`). Check they are free before using. |
| No DDP | Single-GPU by design; the model is tiny. Parallelism is *across* experiments, not within one. |
| Don't touch | `shellgaussianenv` conda env — never modify or clone it. |
| Don't duplicate | Datasets. Reference them in place. |
| Final runs | Must go through **tmux**, with the scripts kept in `FootShellGaussian/scripts/foot_fit_tmux/`. |
| Destructive ops | Shared machine. **Never delete or kill anything without showing the user first.** |

### Explicitly NOT to be ported from the old fitter
Latin-hypercube search, finite-difference Jacobians, polling, least-squares
machinery, candidate ranking tiers, ten-restart logic, hand-written line search,
large numbers of hard thresholds, arbitrary toe-allowance scoring rules —
**unless an experiment proves one is necessary.** (Exactly one earned its place:
multi-start, see §5.)

### Standard of proof
> "Do not declare success merely because the differentiable training loss became
> small."

Every claim must be checked with the **independent exact evaluator** built from
the original NumPy cavity/collision code (`evaluate.py`), which reproduces the
stored Checkpoint 6 metrics exactly.

---

## 2. Core architecture constraints (decisions we committed to)

These are load-bearing. Violating one silently breaks correctness.

1. **Anchored scale.** `neutral_length_scale()` fixes the SUPR→shoe scale from
   the *neutral template*. Foot length is an **output** of the betas, never a
   search variable. 1 normalized unit = **262.5 mm**.
2. **Normalized shoe frame.** X = heel→toe (heel at 0, toe at 1);
   **Y is positive DOWN**; Z = width. Axis remap from SUPR is
   `(x, y, z) → (z, −y, x)`.
3. **Cavity semantics.** Two directional ray casts only: *upper* along −Y from
   the footbed, *side* along ±Z from the centreline. `NaN` = open space.
   **There is no ray along X** — this is a genuine blind spot at the toe and
   heel, and it is why the signed-distance channel exists.
4. **2-D decomposition.** `ceiling_y(x,z)`, `wall_z_neg/pos(x,y)` and
   `footbed_y(x,z)` fully determine the 3-D field.
5. **Signed-distance channel.** Conservative triangle voxelisation +
   `scipy.ndimage.distance_transform_edt`, signed by the directional test. This
   covers the X blind spot.
6. **Sampling.** `F.grid_sample` on `(B,1,D,H,W)`; D↔world X, H↔Y, W↔Z; the grid's
   last dim must be reversed with `.flip(-1)`.
7. **`extend_outside`.** Border padding alone returns the boundary value with
   **zero gradient**, so escaped vertices get no restoring force. We subtract the
   distance back to the box so they do.
8. **Area-weighted barriers, not means.** A mean over samples is not
   subdivision-invariant. Weights are computed under `no_grad` so the optimizer
   can never shrink a triangle to cheapen a violation.
9. **Per-region soft-min support** (heel and forefoot separately). A mean
   over-constrains the arch — a real arch does not touch.
10. **Soft bounds, not clamps.** `beta = 3·tanh(raw)`,
    `pitch = 20°·tanh(raw)`.
11. **Determinism is mandatory.** No `scatter_add_` on CUDA (atomics reorder →
    different betas across identical seeded runs). Use the padded-gather pattern
    over precomputed incident faces.
12. **Multi-start = batch dimension.** 8 restarts fit simultaneously; the winner
    is chosen by the **exact** evaluator, not the training loss.

---

## 3. File paths & components

### The package — `FootShellGaussian/anatomical_coordinates/`

| File | Role |
|---|---|
| `supr_torch.py` | `TorchSuprFoot` — differentiable batched SUPR that **keeps the graph alive**. The existing `foot_prior/supr_foot.py` uses `torch.no_grad()` + `.cpu().numpy()` and cannot be used. Key helper: `pose_from_pitches()` (scatter into pose indices **3 = ankle pitch**, **6 = midfoot pitch**). |
| `shoe_field.py` | Bakes the cavity field. `_raster_extreme`, `_sample_height_field`, `build_cavity_field`, `sample_field` (with `extend_outside`), `sample_footbed_height`, `_voxelize_triangles`, `build_obstacle_distance`. `ShoeCavityField` carries `clearance, valid, distance, lower, upper, spacing, footbed_x/z/y, metadata`. |
| `losses.py` | `containment_loss`, `support_loss`, `beta_prior`, `pose_prior`, `heel_seating_loss`, `length_residual`, `millimetres`. |
| `fitter.py` | `ShoeFootFitter`. `place()` = `remapped * scale + translation`. **`containment_samples()` at line 188 is where the current bug lives.** `fit()` at 226, `evaluate()` at 419. |
| `config.py` | `FitConfig` + `StageConfig`. All weights and the 4-stage schedule. |
| `evaluate.py` | `build_evaluator`, `exact_metrics`, `selection_metrics`. **This is the independent judge.** |
| `shoe_data.py` | `GOLDEN_SET_ROOT`, `available_shoes()`, `load_shoe_case()`. |
| `README.md` | The 875-line deliverable report, sections 0–10. **§8.11 still contains a `PLACEHOLDER_811` marker that must be filled in.** |
| `CONTRACT.md` | Inherited conventions (incl. the X-blindness correction). |
| `tests/` | `test_shoe_field.py`, `test_supr_gradients.py` — **20 tests, all passing** (verified this session, 5.31 s). |
| `scripts/` | `run_torch_foot_fit.py` (main runner), `evaluate_baseline.py`, `benchmark_runtime.py`, `summarize_results.py`. |

### Reference code (read-only — the old pipeline)
- `FootShellGaussian/foot_prior/alignment.py` — Checkpoint 5. Its longitudinal
  seating rule matters: `translation_x = heel_offset − min(plantar_x)` with
  `heel_offset ≥ 0`, so the heel **can never sit behind the functional heel**.
  Our free 3-DOF translation threw that constraint away.
- `FootShellGaussian/foot_prior/containment.py` — the old fitter being replaced.
- `FootShellGaussian/foot_prior/cavity.py`, `mesh.py`, `supr_foot.py`.
- `notes/foot_fitting_plan_DR.md` — the design plan. Two instructions from it are
  **not yet acted on**; see §7.

### tmux scripts — `FootShellGaussian/scripts/foot_fit_tmux/`
`env.sh` (sourced by all others), `run_tests.sh`, `run_baseline.sh`,
`run_experiments.sh`, `run_recommended.sh`, `run_recommended_deterministic.sh`,
`run_batched.sh`, `run_benchmark.sh`, `run_lbfgs.sh`, `prepare_missing_shoes.sh`,
`run_full_dataset.sh`, `run_extra_nolength.sh`, `sweep_extra.sh`.

`sweep_extra.sh` is the current tool for hard-shoe experiments — it takes a GPU
id then any number of `name|flags` specs and writes to `x_<name>`.

### Data & outputs
- Dataset root: `/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation` (27 shoes).
- Prepared golden artifacts: `/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation` (the 16).
- Prepared extras: `/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs/prepared_extra` (the 12, prepared by the **existing** CP4/CP5 scripts).
- Runs: `/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs/` — **69 entries**, indexed in `RUNS.md`.

### Environment
```bash
FF_PY=/home/ab5298/anaconda3/envs/Shell/bin/python   # python3.10
# NOTE: bare `python` does NOT exist on this host. Always use the full path.
```

---

## 4. The two shoe groups

**Golden 16** (had CP4/CP5 artifacts already): `aj_12_basketball_sneakers`,
`birkenstock_arizona_sandal`, `canvas_shoe`, `crocs`, `crocs_by_speedyart_studio`,
`crocs_shoe`, `duinn_shoes_womens_hiking_sandal_sport`, `leather_boots`,
`nike_air_jordan`, `pb129_shoe_low`, `priest_karol_wojtyas_sports_shoes`,
`sandal_1`, `sandals_0001`, `sneaker_vibe`, `sneakers_seen`,
`ww_ii_german_jack_boots`.

**Extra 12** (prepared by us, using the existing pipeline): `adidas_substance`,
`rtfktchallenge_golden_by_franz_vega`,
`shoes_mockup_asset_vans_skate_old_skool_shoes`, `sneaker_1` … `sneaker_8`,
`sneaker_b33`.

---

## 5. Current state — what works

### Both hard gates pass
1. Differentiable SUPR with FD-verified gradients.
2. Differentiable shoe field via `grid_sample`.

Field validated against the real evaluator: median error **0.00–0.04 mm**,
**98–100%** sign agreement. `evaluate.py` reproduces stored Checkpoint 6 exactly.

### Headline result — `expD3_deterministic` vs the old NumPy fitter (16 shoes)

| metric | old fitter | ours (D3) | change |
|---|---|---|---|
| Σ collision area | 0.9881 | **0.8981** | −9.1% |
| Σ outside area | 1.5151 | **1.2299** | −18.8% |
| clear fits | 3/16 | **5/16** | +2 |
| mean ‖β‖ | 5.59 | **4.61** | lower = more anatomical |
| seconds/shoe | 1477.1 | **13.7** | **108× faster** |
| 16 shoes batched | — | 17.2 s | 13.8× more from batching |

Clear fits: `aj_12_basketball_sneakers` (‖β‖=0.16 — essentially the neutral
foot), `crocs`, `crocs_by_speedyart_studio`, `crocs_shoe`, `leather_boots`.

### Findings that earned their keep
- **Multi-start is necessary** — the one search idea we kept.
- **L-BFGS polish is not** — rejected in 8/8; strong-Wolfe extrapolates across
  the flat barrier into non-finite vertices.

---

## 6. Edge cases & known issues

### 6a. Bugs already fixed (do not reintroduce)
| Bug | Fix |
|---|---|
| `torch.flatnonzero` does not exist | `torch.nonzero(..., as_tuple=False).squeeze(1)` |
| **`_raster_extreme` sign bug** — stored `sign*(reference − value)`, so `amin` selected the *farthest* surface | store `sign*(value − reference)`, recover `reference + sign*result`. A single-surface synthetic box cannot catch this; `test_nearest_surface_wins_with_two_ceilings` can. |
| Out-of-domain zero gradient | `extend_outside` subtracts distance back to the box |
| Batched broadcasting | `lower` is `(B,3)` → needs `(B,1,3)` |
| Batched padding domain bug | padding grew the tensor but left `upper` stale → `_extend_axis` + recompute `uppers` |
| Restart/variant bug | restarts perturbed betas even in translation-only variants that never optimize them → suppress restarts unless betas are active |
| **Non-determinism** | `scatter_add_` CUDA atomics → different betas on identical seeds. Replaced with padded gather → bit-identical. |
| L-BFGS divergence | `lr=0.05`, no line search, snapshot-and-revert guard |
| Slice-to-EOF patch | an `s[s.index(...):]` replacement silently deleted `_voxelize_triangles` and `build_obstacle_distance`. **Never patch by slicing to EOF.** |

### 6b. Measurement traps (I fell into these; don't repeat them)
- **Measure heel clearance against the actual obstacle surface, not the shoe's
  outer rear surface.** Doing the latter suggested "11–24 mm of room"; the true
  nearest-obstacle distance at the heel is **0.02–0.26 mm** (touching/through).
- **FD probe conditioning**: a summed readout over 1596 terms destroys the beta
  signal. Also, FD error here *decreases* as the step grows — that is float32
  round-off dominating truncation, not a gradient bug.

### 6c. Provenance discrepancies found while preparing this handoff — **unresolved**
1. **Subdivision level mismatch.** `FitConfig.containment_subdivision_levels`
   defaults to **2**, but `run_torch_foot_fit.py`'s `--subdivision-levels`
   defaults to **1**. Every stored run therefore used **level 1**. Anywhere the
   README or config implies level 2 was used is wrong. Decide which is intended
   and make them agree.
2. **`x_noex_only` is misnamed.** Its stored config is byte-identical to
   `x_noex_heel10` (both `w_heel: 10.0`), and the two produce identical metrics.
   There is therefore **no clean exemption-off / heel-off control** in the run
   set. Re-run one if that comparison matters.
3. `expD3_deterministic`'s serialized config predates the `w_heel`,
   `heel_*_scale_mm` and `length_scale_mm` fields, so they are absent there.
   Expected, not a bug.

### 6d. Claims that were tested and **refuted** (do not resurrect)
- *"The length anchor blocks narrow-shoe fits."* `w_length=0` (`expM`) made it
  **worse**: Σcoll 2.9291 → 2.9723, still 0/12 clear.
- *"The |β| ≤ 3 bound is binding."* Only **1 saturated coefficient across all 12
  extras, 0 across the 16 golden.** The shape space is not exhausted.
- *"The golden set was pre-selected for the reference foot width."* Provenance
  shows both groups come from the same curated pool with
  `reviewed_manifest_entry: true`. Only a measured correlation survives.

---

## 7. The open problem — the extra 12

**0/12 clear under every configuration tried.** Σcoll 2.9291, Σout 3.5890.

### The user's own observation, confirmed quantitatively
> "there is a lot of space in front of the toes, then the heel back side of the
> foot goes through the back side of the shoes"

Toe allowance, target 20.0 mm:
- **Golden 16**: most land on exactly 20.0.
- **Extra 12**: 19.4 – **40.0** mm, mostly 26–40.

So on the extras the foot is sitting **too far back and too short**, leaving a
large unused toe gap while the heel is 0.02–0.26 mm from (or through) the heel
counter. Exactly as described.

### The identified defect — **not yet fixed**
`fitter.py:217-223`:
```python
ankle_y   = joints[:, 1, 1][:, None]
midfoot_x = joints[:, 2, 0][:, None]
exempt = (dense[:, :, 1] < ankle_y) & (dense[:, :, 0] <= midfoot_x)
```
This is an **axis-aligned box covering the entire rear-upper quadrant**, 12–22%
of the foot, including the back of the heel. Inside it the containment **sign is
dropped**, so the foot can reverse through the heel counter **at zero cost**.

**64–86% of heel containment violations sit inside this box.** This is the
mechanism behind the user's observation. It is our bug, not a property of the shoe.

The exemption should cover only where the leg genuinely exits the **collar
opening** — i.e. keyed on the field's own validity / absence of a ceiling —
not a joint-based bounding box. Open space already receives a correct positive
sign from the distance channel.

### But the extras are also genuinely tight
| | cavity volume ÷ foot volume | fill fraction |
|---|---|---|
| golden (clear fits) | 1.26 – 2.52 | 0.37 – 0.53 |
| extras | **0.70 – 1.51** | median 0.51, worst **0.63 – 0.71** |

Foot volume ≈ 759 cm³. `shoes_mockup`'s cavity is **30% smaller than the foot**.
Collar collisions are the largest single component: 36–52% of collision area on
the extras, 47% on non-clear golden shoes — and **all 5 clear golden shoes have
collar collision exactly 0.000**. `pb129_shoe_low` and `priest_karol` are 100%
collar.

### The trap that has stalled progress
**Every physically-sensible fix so far makes the exact collision metric worse**
(all on the same 6 hard shoes, baseline `x_heel10_len1` Σcoll 1.7399):

| run | Σcoll | Σout | note |
|---|---|---|---|
| `x_heel10_len10_ls2` | 1.7318 | 2.2937 | best Σcoll |
| `x_heel0_len10` | 1.7698 | 2.1604 | |
| `x_noex_only` (=noex+heel10) | 1.8432 | **2.1347** | best Σout |
| `x_heel10_len10` | 1.8481 | 2.3789 | |
| `x_noex_heel10_len10` | 1.9394 | 2.2161 | |
| `x_heel10_len10_m03` | 1.9523 | 2.4176 | worst |

**Interpretation to carry forward:** the metric rewards the degenerate
short/floating fit that exploits the exemption loophole. An *honest* fit of a
too-large foot in a too-small shoe collides more than a dishonest one. So
"Σcoll went up" is **not** sufficient evidence that a fix is wrong. Report both
the honest and the exploiting number and let the user judge the trade — do not
silently pick the flattering configuration.

### Plan-document guidance not yet acted on
1. *"Start with a zero or very small interior containment margin."* We use
   **1.0 mm**.
2. *"If fitting needs sub-millimetre boundary precision, first measure whether
   trilinear interpolation from that resolution is actually the limiting error
   before increasing the grid."* Measured: **0.43 mm EDT error at 1 mm spacing**
   — so the margin cannot usefully go below ~0.5 mm without finer spacing.

---

## 8. RESOLVED on 2026-09-22 — read this before acting on §7

Sections 6 and 7 above are left as written so the reasoning is traceable, but
§7's plan has been carried out and **two of its conclusions were wrong**. What
follows supersedes them. Full detail and numbers: `README.md` §8.11.

### What was done

1. **The joint-box exemption is gone** (§7's step 1). `containment_samples` now
   keys the exemption on `open_above(x, z) = footbed exists & no ceiling found`,
   a column mask read off the shoe itself and carried on `ShoeCavityField`.
   Behind the heel counter there is no footbed, so a heel sample there can never
   be exempt. `sample_open_above` in `shoe_field.py`; still behind
   `use_ankle_exemption`.

2. **A second, unlisted defect was found and fixed**: `outside = valid &
   (combined < 0)` signed the distance channel from the X-blind ray families, so
   a sample 20 mm *behind* the heel counter read **+1.8 to +9.2 mm** and the
   value grew with depth. A ±X ray family (`wall_x_neg/pos`, cast from the
   footbed's longitudinal midpoint) now contributes to the sign. `clearance` and
   `valid` are untouched, so the source correspondence and all 20 original tests
   are unaffected.

3. Tests: 29 pass (20 original + 9 new in `tests/test_longitudinal_sign.py`,
   covering the sign behind the heel and beyond the toe, interior signs,
   restoring gradient, and that the collar mask never exempts a heel sample
   behind the counter).

4. `heel_x_mm` is now reported by `evaluate.py`. Note when comparing to older
   runs: it is the minimum *plantar* X, whereas `262.5 - toe - length` is the
   minimum over *all* vertices. They differ by the Achilles overhang — do not
   mix them, as was done once during this session.

### What was wrong in §7

- **"Keep `heel_seating_loss`" (§7 step 2) is refuted.** It was compensating for
  the exemption box. With the box replaced, `w_heel = 10` costs the golden set
  **7 of its 8 clear fits**, by shoving the foot forward into the toe box (mean
  foot length 233.9 -> 242.1 mm). `w_heel` now defaults to **0**; the term stays
  in `losses.py`.
- **"Open space already receives a correct positive sign from the distance
  channel" (§7) is false** behind the heel and beyond the toe. That sentence is
  what hid defect 2.
- §6c.1 resolved: `FitConfig.containment_subdivision_levels` is now **1**,
  matching what every stored run actually used.

### Current state, exact judge

| | Sigma coll | Sigma out | clear | sunk through footbed |
|---|---|---|---|---|
| golden 16, `expD3_deterministic` | 0.8981 | 1.2299 | 5 | 6/16 |
| golden 16, **collar fix, w_heel 0** | 0.9142 | **1.0487** | **8** | 6/16 |
| extra 12, **collar fix, w_heel 0** | 3.4045 | 3.9565 | 0 | **12/12** |

Runs: `s3_g_nosign_nw` (golden), `s4_e_nosign_nw` (extra). The defaults in
`FitConfig` now reproduce these; only `--support-safety-mm 0.15` still has to be
passed.

### The one open problem, and it is specific

**The footbed is excluded from the obstacle set** (contact is allowed there), so
it is absent from the containment barrier and guarded only by `support_loss`'s
squared hinge, which is a finite price. Closing the longitudinal escape route
therefore does not produce an honest fit — it redirects the violation downward.
Turning the ±X sign on raises agreement with the exact SAT judge from 56 % to
63 % and fixes heel reversal (4/6 -> 0/6 on the hard shoes), but drives golden
`clear` 7 -> 2 and sunk 6/16 -> 12/16. It therefore ships **off** behind
`use_longitudinal_sign` / `--longitudinal-sign`.

**All 12 extra shoes now fail for this single reason** — sunk 1.2-4.6 mm — not
for twelve different ones.

**Next change:** make the footbed a one-sided constraint at `footbed_y +
compression_allowance` *inside* the containment barrier, which already has the
right machinery; only the obstacle/footbed split keeps it out. Expect this to
unlock the ±X sign fix at the same time, since the two escape directions are
coupled. Then re-run both groups with `--longitudinal-sign` and compare against
the table above. `shoes_mockup` will still not clear: its cavity is 30 % smaller
than the foot, which is a genuine sizing conflict, not a fitter defect.

Still true from §7: do not lower `containment_margin_mm` below ~1 mm without
first raising field resolution (README §6.5 measured 0.3 mm failing at 0.43 mm
field error).

### Uncommitted state
`anatomical_coordinates/` and `scripts/foot_fit_tmux/` are still **entirely untracked**
in git (branch `dev`). Nothing is mid-edit or broken; 29/29 tests pass.

### Useful commands
```bash
FF_PY=/home/ab5298/anaconda3/envs/Shell/bin/python

cd /storage/Abhinay/Shell_Gaussian && $FF_PY -m pytest FootShellGaussian/anatomical_coordinates/tests -q

# side-by-side exact comparison, including heel seating and toe allowance
$FF_PY FootShellGaussian/anatomical_coordinates/scripts/compare_runs.py \
  expD3_deterministic s3_g_nosign_nw s4_g_sign_nw

# the recommended configuration on both groups
cd FootShellGaussian/scripts/foot_fit_tmux && ./run_signfix_noheel.sh
```
