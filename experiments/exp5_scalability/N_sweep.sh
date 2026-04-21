#!/usr/bin/env bash
# Exp5-A: Cluster size N sweep (§5.6)
# Tier mix held proportional: N=4 (1,1,2), N=6 (1,2,3), N=8 (1,3,4), N=10 (2,3,5), N=12 (2,4,6)
set -euo pipefail

EXP_NAME="exp5_scalability/N_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

# Format: "N_total:high,mid,low"
CONFIGS=(
    "4:1,1,2"
    "6:1,2,3"
    "8:1,3,4"
    "10:2,3,5"
    "12:2,4,6"
)

run_one() {
    local spec="$1"
    local N="${spec%%:*}"
    local mix="${spec##*:}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "N_${N}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --cluster_mix "${mix}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/N_${N}.log" 2>&1
    echo "[done] N=${N} mix=${mix}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${CONFIGS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== N_sweep complete ==="
