#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-EleutherAI/pythia-160m}"
OLD_FILE="${OLD_FILE:-${GLOBAL:-}/data/wikitext103.txt}"
NEW_FILE="${NEW_FILE:-${GLOBAL:-}/data/fineweb_edu_100m.txt}"
BASE_OUT="${BASE_OUT:-aaai27_residual_write_protection/results/pythia160m_pilot}"
SEEDS="${SEEDS:-1 2 3}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16.0}"
PROTECT_LAMBDA="${PROTECT_LAMBDA:-1.0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

local_args=()
if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
  local_args+=(--local-files-only)
fi

footprint_args=(--footprint-device "${FOOTPRINT_DEVICE:-auto}")
if [[ "${SKIP_FOOTPRINT:-0}" == "1" || "${SKIP_FOOTPRINT:-0}" == "true" ]]; then
  footprint_args+=(--skip-footprint)
fi

mkdir -p "$BASE_OUT"

echo "model: $MODEL_ID"
echo "old: $OLD_FILE"
echo "new: $NEW_FILE"
echo "out: $BASE_OUT"

if [[ ! -s "$BASE_OUT/inventory/protected_subspaces.pt" ]]; then
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/write_basis_inventory.py \
    --model-id "$MODEL_ID" \
    --text-file "$OLD_FILE" \
    --output-dir "$BASE_OUT/inventory" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-tokens "${INVENTORY_MAX_TOKENS:-131072}" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --footprint-chunk-size "${FOOTPRINT_CHUNK_SIZE:-128}" \
    --top-k-per-layer "${TOP_K_PER_LAYER:-64}" \
    "${footprint_args[@]}" \
    "${local_args[@]}"
fi

for seed in $SEEDS; do
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/train_write_protected_lora.py \
    --model-id "$MODEL_ID" \
    --train-file "$NEW_FILE" \
    --old-eval-file "$OLD_FILE" \
    --new-eval-file "$NEW_FILE" \
    --output-dir "$BASE_OUT/standard_seed${seed}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --max-steps "${MAX_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-batches "${EVAL_BATCHES:-20}" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    "${local_args[@]}"

  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/train_write_protected_lora.py \
    --model-id "$MODEL_ID" \
    --train-file "$NEW_FILE" \
    --old-eval-file "$OLD_FILE" \
    --new-eval-file "$NEW_FILE" \
    --output-dir "$BASE_OUT/protected_seed${seed}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --max-steps "${MAX_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-batches "${EVAL_BATCHES:-20}" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    --protected-subspaces "$BASE_OUT/inventory/protected_subspaces.pt" \
    --protect-lambda "$PROTECT_LAMBDA" \
    "${local_args[@]}"
done

"$PYTHON_BIN" aaai27_residual_write_protection/scripts/summarize_pilot_results.py \
  "$BASE_OUT"/standard_seed* "$BASE_OUT"/protected_seed* \
  --output "$BASE_OUT/summary.csv"
