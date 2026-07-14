#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SUITE_NAME="${SUITE_NAME:-pythia160m_factoid_word32_r8_1000step}"
BASE_PREFIX="${BASE_PREFIX:-aaai27_residual_write_protection/results/${SUITE_NAME}}"
RUN_SOFT="${RUN_SOFT:-1}"
RUN_HARD="${RUN_HARD:-1}"

export FACT_ANSWER_MODE="${FACT_ANSWER_MODE:-word}"
export FACT_NUM_TRAIN="${FACT_NUM_TRAIN:-32}"
export FACT_SEEN_EVAL="${FACT_SEEN_EVAL:-32}"
export FACT_TRAIN_REPEATS="${FACT_TRAIN_REPEATS:-128}"

export MAX_STEPS="${MAX_STEPS:-1000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_BATCHES="${EVAL_BATCHES:-20}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export SEEDS="${SEEDS:-1 2 3}"
export RANK="${RANK:-8}"
export ALPHA="${ALPHA:-16.0}"
export PROTECT_LAMBDA="${PROTECT_LAMBDA:-1.0}"

echo "suite: $SUITE_NAME"
echo "base prefix: $BASE_PREFIX"
echo "factoids: mode=$FACT_ANSWER_MODE n=$FACT_NUM_TRAIN repeats=$FACT_TRAIN_REPEATS seen_eval=$FACT_SEEN_EVAL"
echo "training: steps=$MAX_STEPS batch=$BATCH_SIZE rank=$RANK alpha=$ALPHA seeds=$SEEDS"
echo "run soft: $RUN_SOFT"
echo "run hard: $RUN_HARD"

if [[ "$RUN_SOFT" == "1" || "$RUN_SOFT" == "true" ]]; then
  export BASE_OUT="${SOFT_BASE_OUT:-${BASE_PREFIX}_soft}"
  export HARD_PROJECT=0
  echo
  echo "=== Running soft write protection: $BASE_OUT ==="
  bash aaai27_residual_write_protection/scripts/run_factoid_pythia160m.sh
fi

if [[ "$RUN_HARD" == "1" || "$RUN_HARD" == "true" ]]; then
  export BASE_OUT="${HARD_BASE_OUT:-${BASE_PREFIX}_hard}"
  export HARD_PROJECT=1
  echo
  echo "=== Running hard write projection: $BASE_OUT ==="
  bash aaai27_residual_write_protection/scripts/run_factoid_pythia160m.sh
fi

echo
echo "suite complete"
if [[ "$RUN_SOFT" == "1" || "$RUN_SOFT" == "true" ]]; then
  echo "soft summary: ${SOFT_BASE_OUT:-${BASE_PREFIX}_soft}/summary.csv"
  echo "soft fact eval: ${SOFT_BASE_OUT:-${BASE_PREFIX}_soft}/fact_eval_seen.csv"
fi
if [[ "$RUN_HARD" == "1" || "$RUN_HARD" == "true" ]]; then
  echo "hard summary: ${HARD_BASE_OUT:-${BASE_PREFIX}_hard}/summary.csv"
  echo "hard fact eval: ${HARD_BASE_OUT:-${BASE_PREFIX}_hard}/fact_eval_seen.csv"
fi
