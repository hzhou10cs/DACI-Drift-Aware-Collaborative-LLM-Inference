#!/usr/bin/env bash
# Exp4-B: Horizon ceiling H_max sweep (§5.5)
set -euo pipefail

EXP_NAME="exp4_sensitivity/H_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

H_VALUES=(1 3 5 8 16)

run_one() {
    local H="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "H_${H}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --H_max "${H}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/H_${H}.log" 2>&1
    echo "[done] H_max=${H}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${H_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== H_sweep complete ==="
