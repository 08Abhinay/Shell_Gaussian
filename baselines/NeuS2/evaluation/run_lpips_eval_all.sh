#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SHOE_LIST="${NEUS2_SHOE_LIST:-${NEUS2_ROOT}/bash_scripts/shoes.txt}"
N_STEPS="${NEUS2_N_STEPS:-10000}"
RUN_ID="${NEUS2_RUN_ID:-lpips_eval_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${NEUS2_LOG_DIR:-${NEUS2_ROOT}/output/batch_runs/${RUN_ID}}"
EVAL_SCRIPT="${NEUS2_EVAL_SCRIPT:-${NEUS2_ROOT}/bash_scripts/eval_shoe.sh}"
SUMMARY_SCRIPT="${SCRIPT_DIR}/evaluate_metrics.py"
PYTHON_BIN="${NEUS2_PYTHON:-${NEUS2_ROOT}/neus2_env/bin/python}"

if [[ -n "${NEUS2_GPUS:-}" ]]; then
  read -r -a GPUS <<< "$(printf '%s' "${NEUS2_GPUS}" | tr ',' ' ')"
else
  GPUS=(0 3 4)
fi

mapfile -t SHOES < <(awk 'NF && $1 !~ /^#/ { print $1 }' "${SHOE_LIST}")
if [[ "${#SHOES[@]}" -eq 0 ]]; then
  echo "No shoes found in ${SHOE_LIST}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "RUN_ID=${RUN_ID}"
echo "LOG_DIR=${LOG_DIR}"
echo "GPUS=${GPUS[*]}"
echo "SHOES=${#SHOES[@]}"

PIDS=()
for WORKER_IDX in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[${WORKER_IDX}]}"
  (
    set -euo pipefail
    for ((i=WORKER_IDX; i<${#SHOES[@]}; i+=${#GPUS[@]})); do
      SHOE_NAME="${SHOES[$i]}"
      LOG_FILE="${LOG_DIR}/${SHOE_NAME}.log"

      echo "[$(date -u +%H:%M:%S)] START ${SHOE_NAME} gpu=${GPU_ID}"
      if bash "${EVAL_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" "${N_STEPS}" > "${LOG_FILE}" 2>&1; then
        AVG_LINE="$(grep 'AVERAGE_TEST' "${NEUS2_ROOT}/output/${SHOE_NAME}_neus2_${N_STEPS}/eval_log.txt" || true)"
        echo "[$(date -u +%H:%M:%S)] DONE ${SHOE_NAME} ${AVG_LINE}"
      else
        echo "[$(date -u +%H:%M:%S)] FAIL ${SHOE_NAME}; see ${LOG_FILE}" >&2
        exit 1
      fi
    done
  ) &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more LPIPS eval jobs failed. Logs are in ${LOG_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
  --output-dir "${NEUS2_ROOT}/output" \
  --shoes-file "${SHOE_LIST}" \
  --summary "${LOG_DIR}/summary.tsv"

echo "ALL_DONE ${LOG_DIR}"
