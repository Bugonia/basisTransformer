#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x ".venv_cu128/bin/python" ]]; then
  PYTHON_BIN=".venv_cu128/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi
DATA_FILE="${DATA_FILE:-data/enwik8.txt}"
ENCODING="${ENCODING:-latin-1}"
OUT_DIR="${OUT_DIR:-runs/block_residuals}"
LOG_DIR="${LOG_DIR:-runs}"

GPUS_STRING="${GPUS:-0 1}"
SEEDS_STRING="${SEEDS:-1}"
VARIANTS_STRING="${VARIANTS:-standard_act_attn standard_act_ffn standard_act_both}"
INCLUDE_STANDARD="${INCLUDE_STANDARD:-1}"

read -r -a GPU_ARRAY <<< "$GPUS_STRING"
read -r -a SEED_ARRAY <<< "$SEEDS_STRING"
read -r -a VARIANT_ARRAY <<< "$VARIANTS_STRING"

if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
  echo "This launcher is tuned for exactly two GPU slots; set GPUS=\"0 1\"." >&2
  exit 1
fi

N_LAYER="${N_LAYER:-8}"
N_HEAD="${N_HEAD:-8}"
N_EMBD="${N_EMBD:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BASE_RUN="${BASE_RUN:-enwik8_residual_output_activation_muon_${N_LAYER}l_${N_EMBD}d_ctx${BLOCK_SIZE}_bs${BATCH_SIZE}}"

MAX_ITERS="${MAX_ITERS:-100000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-30000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
VAL_FRAC="${VAL_FRAC:-0.005}"
TEST_FRAC="${TEST_FRAC:-0.005}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"
ADAMW_FALLBACK_LEARNING_RATE="${ADAMW_FALLBACK_LEARNING_RATE:-2e-4}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DROPOUT="${DROPOUT:-0.1}"
DTYPE="${DTYPE:-bfloat16}"
COMPILE="${COMPILE:-1}"
RESUME="${RESUME:-1}"

mkdir -p "$OUT_DIR" "$LOG_DIR" reports

task_variants=()
task_activations=()
task_seeds=()

for seed in "${SEED_ARRAY[@]}"; do
  if [[ "$INCLUDE_STANDARD" == "1" || "$INCLUDE_STANDARD" == "true" ]]; then
    task_variants+=("standard")
    task_activations+=("identity")
    task_seeds+=("$seed")
  fi
  for variant in "${VARIANT_ARRAY[@]}"; do
    task_variants+=("$variant")
    task_activations+=("gelu")
    task_seeds+=("$seed")
  done
done

echo "GPUs: ${GPU_ARRAY[*]}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Residual-output activation: GELU (matched to the standard FFN)"
echo "Variants: ${VARIANT_ARRAY[*]}"
echo "Tasks: ${#task_variants[@]}"
echo "One independent process per H100; DDP is intentionally disabled."

run_task() {
  local task_idx="$1"
  local gpu="$2"
  local variant="${task_variants[$task_idx]}"
  local activation="${task_activations[$task_idx]}"
  local seed="${task_seeds[$task_idx]}"
  local run_name="${BASE_RUN}_seed${seed}_${variant}"
  local run_dir="${OUT_DIR}/${run_name}"
  local log_path="${LOG_DIR}/${run_name}.log"
  local train_args=(
    train_block_residuals.py
    --data-file "$DATA_FILE"
    --encoding "$ENCODING"
    --out-dir "$OUT_DIR"
    --variant "$variant"
    --residual-output-activation "$activation"
    --run-name "$run_name"
    --seed "$seed"
    --max-iters "$MAX_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --val-frac "$VAL_FRAC"
    --test-frac "$TEST_FRAC"
    --n-layer "$N_LAYER"
    --n-head "$N_HEAD"
    --n-embd "$N_EMBD"
    --batch-size "$BATCH_SIZE"
    --block-size "$BLOCK_SIZE"
    --dropout "$DROPOUT"
    --optimizer muon
    --learning-rate "$LEARNING_RATE"
    --min-lr "$MIN_LR"
    --adamw-fallback-learning-rate "$ADAMW_FALLBACK_LEARNING_RATE"
    --warmup-iters "$WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --dtype "$DTYPE"
  )

  if [[ "$COMPILE" == "1" || "$COMPILE" == "true" ]]; then
    train_args+=(--compile)
  fi

  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    if [[ -s "${run_dir}/summary.csv" ]]; then
      echo "Skipping completed ${run_name}."
      return 0
    fi
  fi

  echo "Launching ${run_name} on GPU ${gpu}."
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "${train_args[@]}" > "$log_path" 2>&1
  echo "Finished ${run_name} on GPU ${gpu}."
}

run_worker() {
  local worker_idx="$1"
  local gpu="${GPU_ARRAY[$worker_idx]}"
  local task_idx
  for task_idx in "${!task_variants[@]}"; do
    if (( task_idx % 2 != worker_idx )); then
      continue
    fi
    run_task "$task_idx" "$gpu"
  done
}

run_worker 0 &
worker0_pid=$!
run_worker 1 &
worker1_pid=$!

cleanup() {
  kill "$worker0_pid" "$worker1_pid" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
if ! wait "$worker0_pid"; then
  echo "Worker on GPU ${GPU_ARRAY[0]} failed; inspect its latest log." >&2
  status=1
fi
if ! wait "$worker1_pid"; then
  echo "Worker on GPU ${GPU_ARRAY[1]} failed; inspect its latest log." >&2
  status=1
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "All residual-output activation runs completed."
echo "Summarize with:"
echo "  $PYTHON_BIN summarize_runs.py '${OUT_DIR}/${BASE_RUN}_*/summary.csv'"
