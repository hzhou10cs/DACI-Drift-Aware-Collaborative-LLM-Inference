#!/usr/bin/env bash
# Exp4-C: Robust slack lambda sweep (§5.5)
set -euo pipefail

EXP_NAME="exp4_sensitivity/lambda_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

LAMBDA_VALUES=(0 0.5 1.0 1.5)

run_one() {
    local L="$1"
    local tag="lambda_${L//./_}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${tag}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --lambda_slack "${L}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/${tag}.log" 2>&1
    echo "[done] lambda=${L}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${LAMBDA_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== lambda_sweep complete ==="
