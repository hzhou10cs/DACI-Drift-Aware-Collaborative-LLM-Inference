#!/usr/bin/env bash
# Exp5-B: Model depth L sweep (§5.6)
# Gemma-7B (L=28), LLaMA-2-13B (L=40), Qwen-2.5-32B (L=64)
set -euo pipefail

EXP_NAME="exp5_scalability/L_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

MODELS=(gemma3-4b llama-3.2-8b qwen3-14b)

run_one() {
    local model="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${model}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${model}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/${model}.log" 2>&1
    echo "[done] model=${model}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${MODELS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== L_sweep complete ==="