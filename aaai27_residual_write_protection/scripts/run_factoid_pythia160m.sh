#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-EleutherAI/pythia-160m}"
GLOBAL_DIR="${GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"

FACT_SEED="${FACT_SEED:-1}"
FACT_NUM_TRAIN="${FACT_NUM_TRAIN:-512}"
FACT_NUM_HELDOUT="${FACT_NUM_HELDOUT:-128}"
FACT_SEEN_EVAL="${FACT_SEEN_EVAL:-128}"
FACT_TRAIN_REPEATS="${FACT_TRAIN_REPEATS:-16}"
FACT_ANSWER_MODE="${FACT_ANSWER_MODE:-word}"
FACT_DIR="${FACT_DIR:-$GLOBAL_DIR/data/residual_write_factoids_${FACT_ANSWER_MODE}_n${FACT_NUM_TRAIN}_r${FACT_TRAIN_REPEATS}_seed${FACT_SEED}}"

if [[ ! -s "$FACT_DIR/train.txt" || ! -s "$FACT_DIR/eval_seen.jsonl" ]]; then
  "$PYTHON_BIN" aaai27_residual_write_protection/scripts/make_factoid_corpus.py \
    --output-dir "$FACT_DIR" \
    --seed "$FACT_SEED" \
    --num-train-facts "$FACT_NUM_TRAIN" \
    --num-heldout-facts "$FACT_NUM_HELDOUT" \
    --seen-eval-facts "$FACT_SEEN_EVAL" \
    --train-repeats "$FACT_TRAIN_REPEATS" \
    --answer-mode "$FACT_ANSWER_MODE"
fi

export MODEL_ID
export OLD_FILE="${OLD_FILE:-$GLOBAL_DIR/data/wikitext103.txt}"
export NEW_FILE="$FACT_DIR/train.txt"
export NEW_EVAL_FILE="$FACT_DIR/eval_seen.txt"
export BASE_OUT="${BASE_OUT:-aaai27_residual_write_protection/results/pythia160m_factoid_${FACT_SEED}}"
export TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-aaai27_residual_write_protection/results/token_cache_pythia160m_factoid}"

export INVENTORY_MAX_TOKENS="${INVENTORY_MAX_TOKENS:-131072}"
export MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-1048576}"
export MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-131072}"
export SKIP_FOOTPRINT="${SKIP_FOOTPRINT:-1}"

export MAX_STEPS="${MAX_STEPS:-500}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_BATCHES="${EVAL_BATCHES:-20}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export BLOCK_SIZE="${BLOCK_SIZE:-512}"
export RANK="${RANK:-8}"
export ALPHA="${ALPHA:-16.0}"
export PROTECT_LAMBDA="${PROTECT_LAMBDA:-1.0}"
export SEEDS="${SEEDS:-1 2 3}"

bash aaai27_residual_write_protection/scripts/run_pilot_pythia160m.sh

local_args=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" || "${LOCAL_FILES_ONLY:-0}" == "true" ]]; then
  local_args+=(--local-files-only)
fi

"$PYTHON_BIN" aaai27_residual_write_protection/scripts/eval_factoid_lora.py \
  "$BASE_OUT"/standard_seed* "$BASE_OUT"/protected_seed* \
  --model-id "$MODEL_ID" \
  --manifest "$FACT_DIR/eval_seen.jsonl" \
  --output "$BASE_OUT/fact_eval_seen.csv" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-records "${FACT_EVAL_MAX_RECORDS:-0}" \
  --max-new-tokens "${FACT_MAX_NEW_TOKENS:-12}" \
  --include-base \
  --candidate-accuracy \
  "${local_args[@]}"
