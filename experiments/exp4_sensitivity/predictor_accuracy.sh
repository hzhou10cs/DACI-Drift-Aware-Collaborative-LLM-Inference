#!/usr/bin/env bash
# Exp4-E: Predictor accuracy (§5.5)
# Need FULL logs with per-window phi_hat_curr and phi_true (already in windows log).
# Aggregation script computes RMSE vs lead k from these traces.
set -euo pipefail

EXP_NAME="exp4_sensitivity/predictor_accuracy"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

# Note: current window log records only phi_hat_curr (k=0). To compare at multiple
# leads, a patch to Predictor.forecast() to dump {phi_hat_horizon, phi_true_horizon}
# would be needed. For now, this script captures phi_true per window (in windows log)
# and runs forecast separately via the aggregation script (offline RMSE).

run_one() {
    local seed="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "seed_${seed}" \
        --schemes DACI \
        --n_traces 1 \
        --seed_start "${seed}" \
        --regime default \
        --log_level full \
        > "${OUTPUT_DIR}/seed_${seed}.log" 2>&1
    echo "[done] seed=${seed}"
}

export -f run_one
export OUTPUT_DIR PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
SEEDS=()
for i in $(seq 0 $((N_TRACES-1))); do
    SEEDS+=($((SEED_START+i)))
done

printf '%s\n' "${SEEDS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== predictor_accuracy complete ==="
echo "Next: python experiments/aggregate.py predictor ${OUTPUT_DIR}"
