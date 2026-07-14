#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-EleutherAI/pythia-160m}"
GLOBAL_DIR="${GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"

FACT_SEED="${FACT_SEED:-1}"
FACT_NUM_TRAIN="${FACT_NUM_TRAIN:-32}"
FACT_NUM_HELDOUT="${FACT_NUM_HELDOUT:-128}"
FACT_SEEN_EVAL="${FACT_SEEN_EVAL:-32}"
FACT_TRAIN_REPEATS="${FACT_TRAIN_REPEATS:-128}"
FACT_ANSWER_MODE="${FACT_ANSWER_MODE:-word}"
FACT_DIR="${FACT_DIR:-$GLOBAL_DIR/data/residual_write_factoids_${FACT_ANSWER_MODE}_n${FACT_NUM_TRAIN}_r${FACT_TRAIN_REPEATS}_seed${FACT_SEED}}"

if [[ ! -s "$FACT_DIR/train_facts.jsonl" || ! -s "$FACT_DIR/eval_seen.jsonl" ]]; then
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/make_factoid_corpus.py \
    --output-dir "$FACT_DIR" \
    --seed "$FACT_SEED" \
    --num-train-facts "$FACT_NUM_TRAIN" \
    --num-heldout-facts "$FACT_NUM_HELDOUT" \
    --seen-eval-facts "$FACT_SEEN_EVAL" \
    --train-repeats "$FACT_TRAIN_REPEATS" \
    --answer-mode "$FACT_ANSWER_MODE"
fi

OLD_FILE="${OLD_FILE:-$GLOBAL_DIR/data/wikitext103.txt}"
BASE_OUT="${BASE_OUT:-aaai27_residual_write_protection/results/pythia160m_factoid_answer_${FACT_SEED}}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-aaai27_residual_write_protection/results/token_cache_pythia160m_factoid_answer}"
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

hard_args=()
if [[ "${HARD_PROJECT:-0}" == "1" || "${HARD_PROJECT:-0}" == "true" ]]; then
  hard_args+=(--hard-project)
fi

footprint_args=(--footprint-device "${FOOTPRINT_DEVICE:-auto}")
if [[ "${SKIP_FOOTPRINT:-0}" == "1" || "${SKIP_FOOTPRINT:-0}" == "true" ]]; then
  footprint_args+=(--skip-footprint)
fi

mkdir -p "$BASE_OUT"

echo "model: $MODEL_ID"
echo "old: $OLD_FILE"
echo "fact dir: $FACT_DIR"
echo "out: $BASE_OUT"
echo "token cache: $TOKEN_CACHE_DIR"
echo "objective: factoid answer-only"
echo "factoids: mode=$FACT_ANSWER_MODE n=$FACT_NUM_TRAIN repeats=$FACT_TRAIN_REPEATS seen_eval=$FACT_SEEN_EVAL"
echo "seeds: $SEEDS"
echo "rank/alpha: $RANK/$ALPHA"
echo "tokens: inventory=${INVENTORY_MAX_TOKENS:-131072} eval=${MAX_EVAL_TOKENS:-131072}"
echo "train: steps=${MAX_STEPS:-1000} batch=${BATCH_SIZE:-4} block=${BLOCK_SIZE:-512} lr=${LEARNING_RATE:-2e-4}"
echo "eval: interval=${EVAL_INTERVAL:-100} batches=${EVAL_BATCHES:-20} seed=${EVAL_SEED:-1234}"
echo "protection: lambda=$PROTECT_LAMBDA hard_project=${HARD_PROJECT:-0} skip_footprint=${SKIP_FOOTPRINT:-0} selection=${INVENTORY_SELECTION_MODE:-importance}"

if [[ ! -s "$BASE_OUT/inventory/protected_subspaces.pt" ]]; then
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/write_basis_inventory.py \
    --model-id "$MODEL_ID" \
    --text-file "$OLD_FILE" \
    --output-dir "$BASE_OUT/inventory" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-tokens "${INVENTORY_MAX_TOKENS:-131072}" \
    --token-cache-dir "$TOKEN_CACHE_DIR" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --footprint-chunk-size "${FOOTPRINT_CHUNK_SIZE:-128}" \
    --top-k-per-layer "${TOP_K_PER_LAYER:-64}" \
    --selection-mode "${INVENTORY_SELECTION_MODE:-importance}" \
    --selection-seed "${INVENTORY_SELECTION_SEED:-1}" \
    "${footprint_args[@]}" \
    "${local_args[@]}"
fi

for seed in $SEEDS; do
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/train_factoid_write_protected_lora.py \
    --model-id "$MODEL_ID" \
    --train-jsonl "$FACT_DIR/train_facts.jsonl" \
    --new-eval-jsonl "$FACT_DIR/eval_seen.jsonl" \
    --old-eval-file "$OLD_FILE" \
    --output-dir "$BASE_OUT/standard_seed${seed}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --dropout "${DROPOUT:-0.0}" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.0}" \
    --seed "$seed" \
    --max-steps "${MAX_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-batches "${EVAL_BATCHES:-20}" \
    --eval-seed "${EVAL_SEED:-1234}" \
    --max-eval-tokens "${MAX_EVAL_TOKENS:-131072}" \
    --max-fact-records "${MAX_FACT_RECORDS:-0}" \
    --token-cache-dir "$TOKEN_CACHE_DIR" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    "${local_args[@]}"

  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/train_factoid_write_protected_lora.py \
    --model-id "$MODEL_ID" \
    --train-jsonl "$FACT_DIR/train_facts.jsonl" \
    --new-eval-jsonl "$FACT_DIR/eval_seen.jsonl" \
    --old-eval-file "$OLD_FILE" \
    --output-dir "$BASE_OUT/protected_seed${seed}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --dropout "${DROPOUT:-0.0}" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.0}" \
    --seed "$seed" \
    --max-steps "${MAX_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --block-size "${BLOCK_SIZE:-512}" \
    --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-batches "${EVAL_BATCHES:-20}" \
    --eval-seed "${EVAL_SEED:-1234}" \
    --max-eval-tokens "${MAX_EVAL_TOKENS:-131072}" \
    --max-fact-records "${MAX_FACT_RECORDS:-0}" \
    --token-cache-dir "$TOKEN_CACHE_DIR" \
    --chars-per-token-budget "${CHARS_PER_TOKEN_BUDGET:-8}" \
    --protected-subspaces "$BASE_OUT/inventory/protected_subspaces.pt" \
    --protect-lambda "$PROTECT_LAMBDA" \
    "${hard_args[@]}" \
    "${local_args[@]}"
done

"$PYTHON_BIN" aaai27_residual_write_protection/scripts/summarize_pilot_results.py \
  "$BASE_OUT"/standard_seed* "$BASE_OUT"/protected_seed* \
  --output "$BASE_OUT/summary.csv"

"$PYTHON_BIN" aaai27_residual_write_protection/scripts/eval_factoid_lora.py \
  "$BASE_OUT"/standard_seed* "$BASE_OUT"/protected_seed* \
  --model-id "$MODEL_ID" \
  --manifest "$FACT_DIR/eval_seen.jsonl" \
  --output "$BASE_OUT/fact_eval_seen.csv" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-records "${FACT_EVAL_MAX_RECORDS:-0}" \
  --max-new-tokens "${FACT_MAX_NEW_TOKENS:-12}" \
  --include-base \
  --candidate-accuracy \
  "${local_args[@]}"
