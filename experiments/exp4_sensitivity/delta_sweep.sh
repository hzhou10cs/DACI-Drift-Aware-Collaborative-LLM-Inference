#!/usr/bin/env bash
# Exp4-D: Delta_max sweep (§5.5, new axis for Eq.27 boundary shift constraint)
set -euo pipefail

EXP_NAME="exp4_sensitivity/delta_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

# Delta values: 1, 3, 5, and effectively unrestricted (L=28 for Gemma-7B)
DELTA_VALUES=(1 3 5 28)

run_one() {
    local D="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "delta_${D}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --delta_max "${D}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/delta_${D}.log" 2>&1
    echo "[done] delta_max=${D}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${DELTA_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== delta_sweep complete ==="
