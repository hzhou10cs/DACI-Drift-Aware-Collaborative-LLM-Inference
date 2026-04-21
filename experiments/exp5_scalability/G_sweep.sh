#!/usr/bin/env bash
# Exp5-C: Request length G_hat sweep (§5.6)
# Compares DACI vs SDA at G_hat in {256, 1024, 2048, 4096}
# Both schemes run to compute relative improvement.
set -euo pipefail

EXP_NAME="exp5_scalability/G_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

G_VALUES=(256 1024 2048 4096)
SCHEMES=(DACI SDA)

run_one() {
    local spec="$1"
    local G="${spec%%:*}"
    local scheme="${spec##*:}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "G_${G}_${scheme}" \
        --schemes "${scheme}" \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --G_hat "${G}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/G_${G}_${scheme}.log" 2>&1
    echo "[done] G=${G} scheme=${scheme}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"

JOBS=()
for G in "${G_VALUES[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
        JOBS+=("${G}:${scheme}")
    done
done
printf '%s\n' "${JOBS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo "=== G_sweep complete ==="
