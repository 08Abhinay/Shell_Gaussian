#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/abelde/projects/active/Shell_Gaussian"
PROCESSED_ROOT="/data/abelde/datasets/processed"
EVALUATION_ROOT="${PROCESSED_ROOT}/evaluation"
SOURCE_ROOT="/data/abelde/datasets/external/source"
TURN_CANON_ROOT="${PROCESSED_ROOT}/gshell_shoes_turntable_canonical"
ALIGNMENT_MESH_ROOT="${PROJECT_ROOT}/FootShellGaussian/output/foot_aware_alignment"
MANIFEST="${PROJECT_ROOT}/FootShellGaussian/configs/external_shoes_render_manifest.json"
BLENDER="${BLENDER:-${PROJECT_ROOT}/baselines/GShell/GShell_env/opt/blender-4.2.21-linux-x64/blender}"
LOG="${LOG:-${PROJECT_ROOT}/FootShellGaussian/output/pipeline_logs/evaluation_generation_$(date -u +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "[$(timestamp)] Starting evaluation dataset generation"
echo "[$(timestamp)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(timestamp)] BLENDER=${BLENDER}"
echo "[$(timestamp)] EVALUATION_ROOT=${EVALUATION_ROOT}"
echo "[$(timestamp)] TURN_CANON_ROOT=${TURN_CANON_ROOT}"
echo "[$(timestamp)] LOG=${LOG}"

if [[ ! -x "${BLENDER}" ]]; then
  echo "[$(timestamp)] ERROR: Blender executable is missing or not executable: ${BLENDER}" >&2
  exit 1
fi

echo "[$(timestamp)] Removing old evaluation and preview folders"
rm -rf \
  "${EVALUATION_ROOT}" \
  "${PROCESSED_ROOT}/external_shoes_canonical_preview" \
  "${PROCESSED_ROOT}/external_shoes_canonical_preview_axis_min"
mkdir -p "${EVALUATION_ROOT}"

echo "[$(timestamp)] Rendering external shoes as multi_elevation_360 with masks and invdepth"
"${BLENDER}" --background --python "${PROJECT_ROOT}/FootShellGaussian/scripts/render_obj_top_bottom_evaluation.py" -- \
  --manifest "${MANIFEST}" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${EVALUATION_ROOT}" \
  --mode multi_elevation_360 \
  --render-invdepth \
  --overwrite

echo "[$(timestamp)] Rendering invdepth for existing GShell turntable-canonical shoes"
"${BLENDER}" --background --python "${PROJECT_ROOT}/FootShellGaussian/scripts/render_existing_turntable_invdepth.py" -- \
  --dataset-root "${TURN_CANON_ROOT}" \
  --mesh-root "${ALIGNMENT_MESH_ROOT}" \
  --overwrite

echo "[$(timestamp)] Validating generated counts"
python - <<'PY'
import json
from pathlib import Path

evaluation_root = Path("/data/abelde/datasets/processed/evaluation")
turn_root = Path("/data/abelde/datasets/processed/gshell_shoes_turntable_canonical")
summary_path = evaluation_root / "summary.json"

errors = []
summary = {"evaluation_root": str(evaluation_root), "turntable_root": str(turn_root), "external": {}, "turntable_invdepth": {}}

if not summary_path.is_file():
    errors.append(f"Missing renderer summary: {summary_path}")
else:
    payload = json.loads(summary_path.read_text())
    for row in payload.get("rows", []):
        if row.get("status") != "ok":
            errors.append(f"External render failed for {row.get('shoe')}: {row.get('error')}")

for shoe_dir in sorted(path for path in evaluation_root.iterdir() if path.is_dir()):
    all_dir = shoe_dir / "multi_elevation_360" / "all"
    train_dir = shoe_dir / "multi_elevation_360" / "train"
    val_dir = shoe_dir / "multi_elevation_360" / "val"
    counts = {
        "all_images": len(list((all_dir / "image").glob("*.jpg"))),
        "all_masks": len(list((all_dir / "mask").glob("*.png"))),
        "all_invdepth": len(list((all_dir / "invdepth").glob("*.npy"))),
        "train_images": len(list((train_dir / "image").glob("*.jpg"))),
        "train_masks": len(list((train_dir / "mask").glob("*.png"))),
        "train_invdepth": len(list((train_dir / "invdepth").glob("*.npy"))),
        "val_images": len(list((val_dir / "image").glob("*.jpg"))),
        "val_masks": len(list((val_dir / "mask").glob("*.png"))),
        "val_invdepth": len(list((val_dir / "invdepth").glob("*.npy"))),
    }
    summary["external"][shoe_dir.name] = counts
    expected = {
        "all_images": 180,
        "all_masks": 180,
        "all_invdepth": 180,
        "train_images": 150,
        "train_masks": 150,
        "train_invdepth": 150,
        "val_images": 30,
        "val_masks": 30,
        "val_invdepth": 30,
    }
    for key, value in expected.items():
        if counts[key] != value:
            errors.append(f"{shoe_dir.name} {key}: expected {value}, got {counts[key]}")

for shoe_dir in sorted(path for path in turn_root.iterdir() if (path / "transforms.json").is_file()):
    transforms = json.loads((shoe_dir / "transforms.json").read_text())
    expected = len(transforms.get("frames", []))
    actual = len(list((shoe_dir / "invdepth").glob("*.npy")))
    summary["turntable_invdepth"][shoe_dir.name] = {"expected": expected, "actual": actual}
    if actual != expected:
        errors.append(f"{shoe_dir.name} invdepth: expected {expected}, got {actual}")

summary["status"] = "failed" if errors else "ok"
summary["errors"] = errors
(evaluation_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if errors:
    print("Validation failed:")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

print(f"Validation ok: {len(summary['external'])} external shoes, {len(summary['turntable_invdepth'])} turntable-canonical shoes")
PY

echo "[$(timestamp)] Done"
