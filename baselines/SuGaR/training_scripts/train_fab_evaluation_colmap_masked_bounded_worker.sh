#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="/storage/Abhinay/Shell_Gaussian/baselines/SuGaR"
RGB_DATASET_ROOT="/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_colmap"
MASKED_DATASET_ROOT="/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_sugar"
RGB_OUTPUT_ROOT="$REPO_ROOT/output/fab_evaluation_colmap_rgb_only"
SUGAR_OUTPUT_ROOT="$REPO_ROOT/output/fab_evaluation_colmap_masked_bounded"
ENV_PREFIX="/home/ab5298/anaconda3/envs/SuGaR_env/SuGaR_env"
CONDA_ROOT="/home/ab5298/anaconda3"
MASK_VALIDATOR="/storage/Abhinay/Shell_Gaussian/dataset_tools/golden_set_evaluation/validate_sugar.py"
COARSE_SCRIPT="$REPO_ROOT/training_scripts/train_fab_evaluation_colmap_masked_bounded_scene.sh"
REFINE_SCRIPT="$REPO_ROOT/training_scripts/refine_fab_evaluation_colmap_masked_bounded_scene.sh"
RUNS_ROOT="$SUGAR_OUTPUT_ROOT/batch_runs"

usage() {
    cat <<'EOF'
Usage:
  train_fab_evaluation_colmap_masked_bounded_worker.sh \
    --worker-index N --worker-count N --gpu N --run-id ID

The sorted 20-scene evaluation set is partitioned by index modulo worker-count.
For each assigned scene, the worker runs the reproducible Zyon-tested protocol:
  1. validate the 180-view masked COLMAP dataset;
  2. train vanilla 3DGS for 7,000 iterations on RGB-only COLMAP images;
  3. run masked, bounded, scale-capped coarse SuGaR and mesh extraction;
  4. run 15,000-step refinement and final textured-mesh export.

Completed stages are validated and skipped. Partial outputs are never silently
overwritten.
EOF
}

worker_index=""
worker_count=""
gpu=""
run_id=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker-index) worker_index=$2; shift 2 ;;
        --worker-count) worker_count=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --run-id) run_id=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in worker_index worker_count gpu run_id; do
    if [[ -z "${!value_name}" ]]; then
        echo "Missing required value: $value_name" >&2
        exit 2
    fi
done
if ! [[ "$worker_index" =~ ^[0-9]+$ && "$worker_count" =~ ^[1-9][0-9]*$ && "$gpu" =~ ^[0-9]+$ ]]; then
    echo "worker-index, worker-count, and gpu must be non-negative integers" >&2
    exit 2
fi
if (( worker_index >= worker_count )); then
    echo "worker-index must be smaller than worker-count" >&2
    exit 2
fi

run_dir="$RUNS_ROOT/$run_id"
summary="$run_dir/worker_${worker_index}_summary.tsv"
mkdir -p "$run_dir"
printf 'timestamp_utc\tworker\tgpu\tscene\tstage\tstatus\tdetail\n' > "$summary"

record() {
    local scene=$1 stage=$2 status=$3 detail=$4
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$worker_index" "$gpu" \
        "$scene" "$stage" "$status" "$detail" | tee -a "$summary"
}

valid_coarse_result() {
    local result=$1
    [[ -s "$result" ]] || return 1
    local status coarse_model coarse_mesh
    status=$(jq -r '.status // empty' "$result")
    coarse_model=$(jq -r '.coarse_model_path // empty' "$result")
    coarse_mesh=$(jq -r '.coarse_mesh_path // empty' "$result")
    [[ "$status" == "complete" && -s "$coarse_model" && -s "$coarse_mesh" ]]
}

valid_refinement_result() {
    local result=$1
    [[ -s "$result" ]] || return 1
    local status refined_model refined_mesh
    status=$(jq -r '.status // empty' "$result")
    refined_model=$(jq -r '.refined_model_path // empty' "$result")
    refined_mesh=$(jq -r '.refined_mesh_path // empty' "$result")
    [[ "$status" == "complete" && -s "$refined_model" && -s "$refined_mesh" ]]
}

set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
cd "$REPO_ROOT"

mapfile -t scenes < <(find "$MASKED_DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#scenes[@]} -ne 20 ]]; then
    echo "Expected exactly 20 masked evaluation scenes; found ${#scenes[@]}" >&2
    exit 2
fi

network_port=$((6500 + worker_index))
echo "Run ID:       $run_id"
echo "Worker:       $worker_index/$worker_count"
echo "Physical GPU: $gpu"
echo "Network port: $network_port"
echo "Protocol:     RGB 3DGS 7k -> masked/bounded coarse -> refinement 15k"
echo "Summary:      $summary"

assigned=0
completed=0
failed=0
for scene_index in "${!scenes[@]}"; do
    (( scene_index % worker_count == worker_index )) || continue
    scene=${scenes[$scene_index]}
    assigned=$((assigned + 1))
    echo
    echo "================================================================"
    echo "Worker $worker_index scene $scene (sorted index $scene_index)"
    echo "================================================================"

    if ! "$ENV_PREFIX/bin/python" "$MASK_VALIDATOR" \
        --dataset-root "$MASKED_DATASET_ROOT" --scene "$scene"; then
        record "$scene" "dataset" "FAILED" "masked COLMAP validation failed"
        failed=$((failed + 1))
        continue
    fi
    record "$scene" "dataset" "VALID" "180 cameras and aligned RGBA masks validated"

    scene_output="$SUGAR_OUTPUT_ROOT/$scene"
    refinement_result="$scene_output/refinement_run_result.json"
    coarse_result="$scene_output/coarse_pilot_result.json"
    if valid_refinement_result "$refinement_result"; then
        record "$scene" "pipeline" "SKIPPED_COMPLETE" "validated final refined result"
        completed=$((completed + 1))
        continue
    fi

    rgb_scene="$RGB_DATASET_ROOT/$scene/undistorted"
    gs_output="$RGB_OUTPUT_ROOT/$scene/vanilla_gs"
    gs_checkpoint="$gs_output/point_cloud/iteration_7000/point_cloud.ply"
    if [[ -s "$gs_checkpoint" && -s "$gs_output/cameras.json" ]]; then
        record "$scene" "vanilla_gs" "REUSED" "validated iteration-7000 checkpoint"
    else
        if [[ -e "$gs_output" ]]; then
            record "$scene" "vanilla_gs" "FAILED" "partial output exists without a valid 7k checkpoint"
            failed=$((failed + 1))
            continue
        fi
        record "$scene" "vanilla_gs" "STARTED" "7000 iterations on RGB-only fresh COLMAP images"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PREFIX/bin/python" gaussian_splatting/train.py \
            -s "$rgb_scene" \
            -m "$gs_output" \
            --iterations 7000 \
            --eval \
            -w \
            --port "$network_port"; then
            record "$scene" "vanilla_gs" "FAILED" "training command failed"
            failed=$((failed + 1))
            continue
        fi
        if [[ ! -s "$gs_checkpoint" || ! -s "$gs_output/cameras.json" ]]; then
            record "$scene" "vanilla_gs" "FAILED" "training ended without required checkpoint files"
            failed=$((failed + 1))
            continue
        fi
        record "$scene" "vanilla_gs" "COMPLETE" "iteration-7000 checkpoint validated"
    fi

    if valid_coarse_result "$coarse_result"; then
        record "$scene" "coarse_sugar" "REUSED" "validated masked/bounded coarse result"
    else
        if [[ -e "$scene_output" ]]; then
            record "$scene" "coarse_sugar" "FAILED" "partial output exists without a valid coarse result"
            failed=$((failed + 1))
            continue
        fi
        record "$scene" "coarse_sugar" "STARTED" "masked/bounded scale-capped coarse training"
        if ! "$COARSE_SCRIPT" "$scene" "$gpu"; then
            record "$scene" "coarse_sugar" "FAILED" "coarse training command failed"
            failed=$((failed + 1))
            continue
        fi
        if ! valid_coarse_result "$coarse_result"; then
            record "$scene" "coarse_sugar" "FAILED" "coarse result manifest or artifacts are invalid"
            failed=$((failed + 1))
            continue
        fi
        record "$scene" "coarse_sugar" "COMPLETE" "coarse model and mesh validated"
    fi

    record "$scene" "refinement" "STARTED" "15000 iterations and final mesh export"
    if ! "$REFINE_SCRIPT" "$scene" "$gpu"; then
        record "$scene" "refinement" "FAILED" "refinement command failed"
        failed=$((failed + 1))
        continue
    fi
    if ! valid_refinement_result "$refinement_result"; then
        record "$scene" "refinement" "FAILED" "final result manifest or artifacts are invalid"
        failed=$((failed + 1))
        continue
    fi
    record "$scene" "pipeline" "COMPLETE" "vanilla GS, coarse SuGaR, refinement, and mesh export validated"
    completed=$((completed + 1))
done

echo
echo "Worker $worker_index finished: assigned=$assigned complete_or_skipped=$completed failed=$failed"
echo "Summary: $summary"
(( failed == 0 ))
