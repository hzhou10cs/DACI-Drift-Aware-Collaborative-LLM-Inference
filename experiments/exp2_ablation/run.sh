#!/usr/bin/env bash
# Exp2: §5.3 Ablation Study
# 4-panel comparison: TTLT, #Reconf, Ovhd, P99 TPOT
# Variants: Full DACI, w/o Predictor, w/o Lazy, w/o Bottleneck-DP, w/o Adaptive-H
set -euo pipefail

EXP_NAME="exp2_ablation"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

# ablation_mode labels
VARIANTS=(
    "full:none"
    "no_predictor:no_predictor"
    "no_lazy:no_lazy"
    "no_bottleneck:no_bottleneck"
    "no_adaptive_H:no_adaptive_H"
)

run_one() {
    local spec="$1"
    local label="${spec%%:*}"
    local mode="${spec##*:}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${label}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --ablation "${mode}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/${label}.log" 2>&1
    echo "[done] ${label}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${VARIANTS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== ${EXP_NAME} complete. Outputs: ${OUTPUT_DIR} ==="
