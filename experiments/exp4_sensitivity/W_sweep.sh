#!/usr/bin/env bash
# Exp4-A: Window length W sweep (§5.5)
set -euo pipefail

EXP_NAME="exp4_sensitivity/W_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

W_VALUES=(5 10 20 50 100)

run_one() {
    local W="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "W_${W}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --W_tokens "${W}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/W_${W}.log" 2>&1
    echo "[done] W=${W}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${W_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== W_sweep complete ==="
