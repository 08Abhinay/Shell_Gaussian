# Foot Alignment Reproduction Notes

This file records the exact pipeline used for the current shoe/foot alignment
work. The current accepted path is the turntable-phase canonical dataset:

```text
/data/abelde/datasets/processed/gshell_shoes_turntable_canonical
```

The older visual-hull canonical dataset:

```text
/data/abelde/datasets/processed/gshell_shoes_canonical
```

was useful for exploration, but it is not the dataset used by the current
`turntable-512-768` meshes and alignment outputs.

## 0. Common Paths

Run commands from:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

export PROJECT_ROOT=/data/abelde/projects/active/Shell_Gaussian
export FOOTSHELL_ROOT=${PROJECT_ROOT}/FootShellGaussian
export GSHELL_ROOT=${PROJECT_ROOT}/baselines/GShell
export PY=${GSHELL_ROOT}/GShell_env/bin/python
```

The GShell environment used here is:

```text
/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env
```

## 1. Canonicalize The Processed Dataset

Problem found: the images look like a consistent turntable dataset, but COLMAP
can give each scene a different global yaw. So the mesh from one shoe may face
the opposite X direction from another shoe even when the images look consistent.

Current fix: keep every image and mask unchanged, but rotate each scene's
camera poses so `img01.jpg` lands at the same turntable angle for every shoe.

Command:

```bash
${PY} ${FOOTSHELL_ROOT}/dataset/canonicalize_gshell_turntable_phase.py \
  --input-root /data/abelde/datasets/processed/gshell_shoes \
  --output-root /data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
  --reference-frame img01.jpg \
  --target-angle-deg 90 \
  --overwrite
```

Expected outputs:

```text
/data/abelde/datasets/processed/gshell_shoes_turntable_canonical/<shoe>/transforms.json
/data/abelde/datasets/processed/gshell_shoes_turntable_canonical/<shoe>/turntable_canonicalization.json
/data/abelde/datasets/processed/gshell_shoes_turntable_canonical/summary.csv
/data/abelde/datasets/processed/gshell_shoes_turntable_canonical/summary.json
```

Important: `image/` and `mask/` are symlinks to the original processed dataset.
The original dataset is not modified.

Quick validation:

```bash
${PY} - <<'PY'
import json
from pathlib import Path

root = Path("/data/abelde/datasets/processed/gshell_shoes_turntable_canonical")
summary = json.loads((root / "summary.json").read_text())
print("scene_count:", summary["scene_count"])
print("status_counts:", summary["status_counts"])
print("max_target_error_deg:", summary["max_target_error_deg"])
print("all_rotations_passed:", summary["all_rotations_passed"])
print("all_frame_counts_36:", summary["all_frame_counts_36"])
print("original_dataset_modified:", summary["original_dataset_modified"])
PY
```

Optional normalization report:

```bash
${PY} ${FOOTSHELL_ROOT}/scripts/check_shoe_normalization.py \
  --dataset-root /data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
  --sphere-init-norm 0.17 \
  --output-json ${FOOTSHELL_ROOT}/output/turntable_canonical_normalization_report.json
```

## 2. Train GShell On The Canonicalized Dataset

Use the lower-resolution config to match the existing faster run:

```text
${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json
```

This config has:

```text
train_res = [512, 768]
sphere_init_norm = 0.17
```

Train a selected set of shoes into:

```text
${GSHELL_ROOT}/output/turntable-512-768
```

Command for the 20 shoes used in the current debugging pass:

```bash
cd ${GSHELL_ROOT}

SHOES=(
  Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant
  Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Infants
  Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Kids
  Air-Jordan-1-Mid-Wear-Away-Chicago-Gs
  Air-Jordan-1-Retro-High-Hyper-Royal-Smoke-Grey-Gs
  Air-Jordan-1-Retro-High-Og-Washed-Black-Gs
  Air-Jordan-1-Retro-High-Og-White-Cement-Gs
  Air-Jordan-12-Retro-Arctic-Punch-Gs
  Air-Jordan-13-Retro-Houndstooth-Gs
  Air-Jordan-5-Retro-Plaid-Gs
  Air-Jordan-6-Retro-Washed-Denim-2022-Gs
  Birkenstock-Boston-Suede-Stone-Coin
  Crocs-Classic-Clog-Cinnamon-Toast-Crunch
  Crocs-Classic-Clog-Cinnamon-Toast-Crunch-Gs
  Crocs-Classic-Clog-Cocoa-Puffs-Kids
  Crocs-Classic-Clog-Staple-Sidewalk-Luxe
  Nike-Calm-Slide-Cinnamon-Monarch
  Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail
  Ugg-Bailey-Bow-Ii-Boot-Ribbon-Red-Kids
  Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler
)

MIN_FREE_MB=51200 \
MAX_PARALLEL_JOBS=5 \
SKIP_EXISTING=1 \
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
GSHELL_CONFIG=${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=${GSHELL_ROOT}/output/turntable-512-768 \
bash ${GSHELL_ROOT}/scripts/train_all_shoes_tmux.sh gshell_turntable_20 "${SHOES[@]}"
```

Attach to the tmux session:

```bash
tmux attach -t gshell_turntable_20
```

Check completed outputs:

```bash
find ${GSHELL_ROOT}/output/turntable-512-768 \
  -mindepth 2 -maxdepth 2 -path '*/mesh/mesh.obj' | wc -l

find ${GSHELL_ROOT}/output/turntable-512-768 \
  -mindepth 2 -maxdepth 2 -path '*/mesh_watertight/mesh.obj' | wc -l
```

To train all scenes later, omit the `SHOES` list:

```bash
cd ${GSHELL_ROOT}

MIN_FREE_MB=51200 \
MAX_PARALLEL_JOBS=5 \
SKIP_EXISTING=1 \
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
GSHELL_CONFIG=${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=${GSHELL_ROOT}/output/turntable-512-768 \
bash ${GSHELL_ROOT}/scripts/train_all_shoes_tmux.sh gshell_turntable_all
```

`SKIP_EXISTING=1` means completed shoes are skipped when both
`mesh/mesh.obj` and `validate/metrics.txt` exist.

## 3. Re-run One Failed Shoe

One shoe previously failed due to training instability:

```text
Air-Jordan-6-Retro-Washed-Denim-2022-Gs
```

If a shoe fails and you want a clean retry, remove only that shoe's output:

```bash
rm -rf ${GSHELL_ROOT}/output/turntable-512-768/Air-Jordan-6-Retro-Washed-Denim-2022-Gs_turntable
```

Then retrain just that shoe:

```bash
cd ${GSHELL_ROOT}

MIN_FREE_MB=51200 \
MAX_PARALLEL_JOBS=1 \
SKIP_EXISTING=0 \
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
GSHELL_CONFIG=${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=${GSHELL_ROOT}/output/turntable-512-768 \
bash ${GSHELL_ROOT}/scripts/train_all_shoes_tmux.sh gshell_turntable_retry \
  Air-Jordan-6-Retro-Washed-Denim-2022-Gs
```

Keep `sphere_init_norm = 0.17` in the config.

## 4. Generate Foot Alignment Debug Outputs

This step uses the meshes already exported by training:

```text
${GSHELL_ROOT}/output/turntable-512-768/<shoe>_turntable/mesh/mesh.obj
${GSHELL_ROOT}/output/turntable-512-768/<shoe>_turntable/mesh_watertight/mesh.obj
```

It does not regenerate meshes from `model.pt` unless the exported meshes are
missing or `--force-reexport-from-checkpoint` is passed.

Command for the same 20 shoes:

```bash
cd ${PROJECT_ROOT}

SHOES=(
  Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant
  Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Infants
  Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Kids
  Air-Jordan-1-Mid-Wear-Away-Chicago-Gs
  Air-Jordan-1-Retro-High-Hyper-Royal-Smoke-Grey-Gs
  Air-Jordan-1-Retro-High-Og-Washed-Black-Gs
  Air-Jordan-1-Retro-High-Og-White-Cement-Gs
  Air-Jordan-12-Retro-Arctic-Punch-Gs
  Air-Jordan-13-Retro-Houndstooth-Gs
  Air-Jordan-5-Retro-Plaid-Gs
  Air-Jordan-6-Retro-Washed-Denim-2022-Gs
  Birkenstock-Boston-Suede-Stone-Coin
  Crocs-Classic-Clog-Cinnamon-Toast-Crunch
  Crocs-Classic-Clog-Cinnamon-Toast-Crunch-Gs
  Crocs-Classic-Clog-Cocoa-Puffs-Kids
  Crocs-Classic-Clog-Staple-Sidewalk-Luxe
  Nike-Calm-Slide-Cinnamon-Monarch
  Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail
  Ugg-Bailey-Bow-Ii-Boot-Ribbon-Red-Kids
  Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler
)

SHOE_ARGS=()
for shoe in "${SHOES[@]}"; do
  SHOE_ARGS+=(--shoe-name "$shoe")
done

CUDA_VISIBLE_DEVICES=0 ${PY} ${FOOTSHELL_ROOT}/scripts/prepare_dataset_foot_alignment_debug.py \
  --dataset-root /data/abelde/datasets/processed/gshell_shoes_turntable_canonical \
  --baseline-output-root ${GSHELL_ROOT}/output \
  --baseline-subdir turntable-512-768 \
  --baseline-suffix _turntable \
  --gshell-config ${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json \
  --out-root ${GSHELL_ROOT}/output/foot_alignment_turntable-512-768 \
  --device cuda \
  "${SHOE_ARGS[@]}" \
  --overwrite
```

Expected outputs:

```text
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/<shoe>/overview.png
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/<shoe>/axis_alignment_views.png
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/<shoe>/foot_inside_shoe_overlay.ply
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/<shoe>/alignment.json
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/summary.csv
${GSHELL_ROOT}/output/foot_alignment_turntable-512-768/all_shoes_contact_sheet.png
```

Notebook for visual inspection:

```text
${FOOTSHELL_ROOT}/notebooks/foot_dataset_alignment_playground.ipynb
```

## 5. Extract Support Footprint And Pseudo-Footbed V4

This stage prepares the geometry signals used before foot fitting. It does not
fit the foot yet.

Goal: build a reusable support footprint and a smooth pseudo-footbed estimate
from the trained shoe meshes.

The V4 path does this:

1. Uses the watertight mesh as the main shoe surface.
2. Uses the open/mSDF mesh lower boundary to find the outer bottom outline.
3. Fills holes in that outline, so Crocs holes do not break the footprint.
4. Filters fake watertight bottom bulges using that outline.
5. Builds the support footprint, centerline, width profile, heel/ball/toe
   regions, and floor diagnostic plots.
6. Builds a smooth pseudo-footbed from the open lower boundary plus a small
   inward offset toward the shoe opening.

The key V4 offset is:

```text
offset = clamp(0.055 * support_length, 0.008, 0.022)
```

In the current shoe coordinates, moving from the outer bottom toward the shoe
opening decreases `Y`, so this offset is subtracted from the open lower boundary
height.

Command used for the completed 20-shoe V4 run:

```bash
cd ${PROJECT_ROOT}

${PY} ${FOOTSHELL_ROOT}/scripts/run_support_footbed_analysis.py \
  --mesh-root ${GSHELL_ROOT}/output/turntable-512-768 \
  --output-root ${GSHELL_ROOT}/output/support_footbed_analysis_v4 \
  --grid-resolution 192 \
  --footbed-offset 0.015 \
  --heightmap-min-samples-per-cell 2 \
  --heightmap-smooth-sigma 1.25 \
  --heightmap-profile-clip 0.025 \
  --footbed-inner-margin-cells 7 \
  --smooth-footbed-window-fraction 0.18 \
  --footbed-height-fraction 0.22 \
  --open-boundary-footbed-offset-ratio 0.055 \
  --open-boundary-footbed-offset-min 0.008 \
  --open-boundary-footbed-offset-max 0.022 \
  --overwrite
```

Expected outputs:

```text
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/support_footprint.json
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/footprint_centerline.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/width_profile.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/footbed_profile.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/floor_profile_v2.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/floor_samples_overlay.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/floor_surface_v2.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/floor_diagnostic_v2.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_heightmap.npz
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/outer_floor_surface.obj
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_surface.obj
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_smooth_surface.obj
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_heightmap.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_cross_sections.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_surface_preview.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/pseudo_footbed_smooth_surface_preview.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/<shoe>_turntable/support_faces_overlay.png
${GSHELL_ROOT}/output/support_footbed_analysis_v4/support_footbed_summary.json
${GSHELL_ROOT}/output/support_footbed_analysis_v4/support_footbed_summary.csv
```

Quick validation:

```bash
${PY} - <<'PY'
import json
from pathlib import Path

root = Path("/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/support_footbed_analysis_v4")
rows = json.loads((root / "support_footbed_summary.json").read_text())
ok_rows = [row for row in rows if "error" not in row]
offsets = [float(row["smooth_footbed_offset"]) for row in ok_rows]
print("rows:", len(rows))
print("errors:", [row for row in rows if "error" in row])
print("npz_count:", len(list(root.glob("*/pseudo_footbed_heightmap.npz"))))
print("smooth_obj_count:", len(list(root.glob("*/pseudo_footbed_smooth_surface.obj"))))
print("smooth_preview_count:", len(list(root.glob("*/pseudo_footbed_smooth_surface_preview.png"))))
print("footprint_sources:", sorted({row.get("footprint_source") for row in ok_rows}))
print("smooth_sources:", sorted({row.get("smooth_footbed_source") for row in ok_rows}))
print("offset_min_median_max:", min(offsets), sorted(offsets)[len(offsets) // 2], max(offsets))
PY
```

For the current 20-shoe run, the expected validation result is:

```text
rows: 20
errors: []
npz_count: 20
smooth_obj_count: 20
smooth_preview_count: 20
footprint_sources: ['open_bottom_silhouette']
smooth_sources: ['open_boundary_offset']
offset_min_median_max: 0.011506403759121896 0.01690333917737007 0.02157927393913269
```

Notebook for visual inspection:

```text
${FOOTSHELL_ROOT}/notebooks/foot_support_footbed_playground.ipynb
```

At the end of that notebook, run the executable section:

```text
Run Full V4 Support-Footbed Analysis
```

That cell builds the same command with `sys.argv` and runs
`run_support_footbed_analysis.py` from inside Jupyter.

## 6. Run The V4 Hybrid Footbed-Aware Optimizer

This stage implements the practical Section 1.4 optimizer. The shoe mesh stays
fixed. The SUPR foot is scaled, yawed, pitched, rolled, and translated into the
shoe using the V4 smooth pseudo-footbed and footprint.

The hybrid version uses the previous translation-friendly optimizer as a warm
start when available. This keeps the placements that already looked good for
normal shoes, while using the V4 footbed/centerline as soft guidance. Tall boots
are detected by height ratio and routed to a stronger boot-specific fitting
mode.

Command:

```bash
cd ${PROJECT_ROOT}

CUDA_VISIBLE_DEVICES=5 ${PY} ${FOOTSHELL_ROOT}/scripts/run_foot_fit_optimization.py \
  --mesh-root ${GSHELL_ROOT}/output/turntable-512-768 \
  --support-root ${GSHELL_ROOT}/output/support_footbed_analysis_v4 \
  --baseline-alignment-root ${GSHELL_ROOT}/output/foot_alignment_turntable-512-768 \
  --warm-start-alignment-root ${GSHELL_ROOT}/output/foot_alignment_optimized_turntable-512-768 \
  --output-root ${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768 \
  --foot-obj ${PROJECT_ROOT}/baselines/SUPR/output/debug_playground/supr_male_right_foot_neutral.obj \
  --device cuda \
  --style-mode auto \
  --adam-steps 120 \
  --lbfgs-steps 12 \
  --overwrite
```

To run only the four smoke-test shoes, add:

```bash
  --scene Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant_turntable \
  --scene Crocs-Classic-Clog-Cinnamon-Toast-Crunch-Gs_turntable \
  --scene Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler_turntable \
  --scene Air-Jordan-1-Retro-High-Og-White-Cement-Gs_turntable
```

Expected outputs:

```text
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/alignment_optimized.json
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/fit_metrics.json
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/foot_aligned_optimized.obj
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/foot_inside_shoe_optimized.ply
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/pseudo_cavity.npz
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/fit_before_after.png
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/cavity_slices.png
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/plantar_clearance.png
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/<shoe>/fit_contact_sheet.png
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/summary.json
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/summary.csv
${GSHELL_ROOT}/output/foot_alignment_optimized_v4_hybrid_turntable-512-768/all_shoes_optimized_contact_sheet.png
```

Quick validation:

```bash
${PY} - <<'PY'
import json
from pathlib import Path

root = Path("/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/foot_alignment_optimized_v4_hybrid_turntable-512-768")
rows = json.loads((root / "summary.json").read_text())
ok = [row for row in rows if row.get("status") == "ok"]
print("rows:", len(rows))
print("ok:", len(ok))
print("errors:", [row for row in rows if row.get("status") != "ok"])
print("loss_improved:", sum(float(row["loss_after"]) < float(row["loss_before"]) for row in ok), "/", len(ok))
print("fit_metrics:", len(list(root.glob("*/fit_metrics.json"))))
print("contact_sheets:", len(list(root.glob("*/fit_contact_sheet.png"))))
print("warm_start_used:", sum(bool(row.get("warm_start_used")) for row in ok), "/", len(ok))
print("styles:", sorted({row.get("style_mode") for row in ok}))
PY
```

Notebook entry point:

```text
${FOOTSHELL_ROOT}/notebooks/foot_dataset_alignment_playground.ipynb
Section 15: Run Section 1.4 V4 Hybrid Optimizer
```

## 7. What We Have Achieved So Far

We now have:

1. A processed dataset whose turntable phase is consistent across shoes.
2. GShell meshes trained from that adjusted dataset.
3. Foot alignment debug outputs using those trained meshes.
4. A support-footprint extractor that gives the sole outline, centerline, width
   profile, and heel/ball/toe regions.
5. A cleaner way to ignore fake watertight bottom bulges by using the open mesh
   bottom outline as a guide.
6. A V2 floor estimate that uses many watertight samples inside the black
   footprint instead of trusting sparse red support faces directly.
7. A V4 smooth pseudo-footbed surface made from the open lower boundary plus a
   controlled inward offset, exported as
   `pseudo_footbed_smooth_surface.obj`.
8. A V4 hybrid optimizer that starts from the previous translation-friendly
   placement, samples `B(x,z)` from `pseudo_footbed_heightmap.npz`, routes tall
   boots separately, and writes optimized foot placement diagnostics.

## 8. Current Limitations

The baseline debug alignment is still basic. The V4 optimizer improves the
placement after that baseline, but the baseline code still lives in:

```text
${FOOTSHELL_ROOT}/foot_prior/foot_alignment.py
```

The main known problems are:

1. The optimizer uses a simplified pseudo-cavity, not a true 3D cavity SDF.
2. SUPR toe articulation and shape coefficients are not optimized yet.
3. Shoe resizing is not implemented; the shoe mesh stays fixed.
4. The mSDF cutting logic has not yet been updated to use the final footbed
   placement.

## 9. Next Steps

Recommended next implementation order:

1. Visually inspect the V4 optimized contact sheets for the 20-shoe batch.
2. Tune optimizer weights only if the V4 overlays show systematic issues.
3. Use confidence checks from `support_footprint.json` to flag bad shoes instead
   of silently trusting every result.
4. After foot placement is reliable, update:

```text
${FOOTSHELL_ROOT}/geometry/gshell_tets.py
```

so the mSDF logic preserves the sole correctly and cuts only above the usable
foot base.

## 10. Key Files

```text
${FOOTSHELL_ROOT}/dataset/canonicalize_gshell_turntable_phase.py
${FOOTSHELL_ROOT}/scripts/prepare_dataset_foot_alignment_debug.py
${FOOTSHELL_ROOT}/scripts/run_support_footbed_analysis.py
${FOOTSHELL_ROOT}/scripts/run_foot_fit_optimization.py
${FOOTSHELL_ROOT}/foot_prior/foot_alignment.py
${FOOTSHELL_ROOT}/foot_prior/foot_fit_optimizer.py
${FOOTSHELL_ROOT}/foot_prior/support_footprint.py
${FOOTSHELL_ROOT}/geometry/gshell_tets.py
${FOOTSHELL_ROOT}/notebooks/foot_dataset_alignment_playground.ipynb
${FOOTSHELL_ROOT}/notebooks/foot_support_footbed_playground.ipynb
```
