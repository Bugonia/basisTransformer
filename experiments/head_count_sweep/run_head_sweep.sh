#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -n "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

BASE_RUN="${BASE_RUN:-enwik8_head_sweep_standard_8l_512d_ctx512_bs256_lr2e4_test005_30k}"
PYTHON_BIN="${PYTHON_BIN:-.venv_cu128/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_block_residuals.py}"
DATA_FILE="${DATA_FILE:-data/enwik8.txt}"
ENCODING="${ENCODING:-latin-1}"

HEADS_STRING="${HEADS:-1 2 4 8 16 32 64 128 256 512}"
SEEDS_STRING="${SEEDS:-1 2}"
read -r -a HEAD_ARRAY <<< "$HEADS_STRING"
read -r -a SEED_ARRAY <<< "$SEEDS_STRING"

N_LAYER="${N_LAYER:-8}"
N_EMBD="${N_EMBD:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
MAX_ITERS="${MAX_ITERS:-30000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
VAL_FRAC="${VAL_FRAC:-0.005}"
TEST_FRAC="${TEST_FRAC:-0.005}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
DTYPE="${DTYPE:-bfloat16}"
COMPILE="${COMPILE:-0}"
RESUME="${RESUME:-1}"

if [[ -n "${GPUS:-}" ]]; then
  read -r -a GPU_ARRAY <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARRAY=()
  while IFS= read -r gpu_id; do
    GPU_ARRAY+=("$gpu_id")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  GPU_ARRAY=(0)
fi

if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS=\"0\" or similar." >&2
  exit 1
fi

compile_args=()
if [[ "$COMPILE" == "1" || "$COMPILE" == "true" ]]; then
  compile_args+=(--compile)
fi

mkdir -p runs reports
wave_pids=()
wave_names=()

cleanup() {
  if [[ "${#wave_pids[@]}" -gt 0 ]]; then
    echo "Stopping ${#wave_pids[@]} active run(s)..." >&2
    kill "${wave_pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

wait_wave() {
  local idx
  for idx in "${!wave_pids[@]}"; do
    if ! wait "${wave_pids[$idx]}"; then
      echo "Run failed: ${wave_names[$idx]}" >&2
      exit 1
    fi
  done
  wave_pids=()
  wave_names=()
}

for seed in "${SEED_ARRAY[@]}"; do
  wave_pids=()
  wave_names=()
  slot=0

  for n_head in "${HEAD_ARRAY[@]}"; do
    if (( N_EMBD % n_head != 0 )); then
      echo "Skipping n_head=${n_head}: N_EMBD=${N_EMBD} is not divisible by n_head." >&2
      continue
    fi

    gpu="${GPU_ARRAY[$slot]}"
    run_name="${BASE_RUN}_seed${seed}_h${n_head}"
    run_dir="runs/block_residuals/${run_name}"
    log_path="runs/${run_name}.log"

    if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
      if [[ -s "${run_dir}/summary.csv" ]]; then
        echo "Skipping completed ${run_name}."
        continue
      fi
    fi

    echo "Launching ${run_name} on GPU ${gpu}."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$TRAIN_SCRIPT" \
      --data-file "$DATA_FILE" \
      --encoding "$ENCODING" \
      --variant standard \
      --run-name "$run_name" \
      --seed "$seed" \
      --max-iters "$MAX_ITERS" \
      --eval-interval "$EVAL_INTERVAL" \
      --eval-iters "$EVAL_ITERS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --val-frac "$VAL_FRAC" \
      --test-frac "$TEST_FRAC" \
      --n-layer "$N_LAYER" \
      --n-head "$n_head" \
      --n-embd "$N_EMBD" \
      --batch-size "$BATCH_SIZE" \
      --block-size "$BLOCK_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --min-lr "$MIN_LR" \
      --warmup-iters "$WARMUP_ITERS" \
      --dtype "$DTYPE" \
      "${compile_args[@]}" \
      > "$log_path" 2>&1 &

    wave_pids+=("$!")
    wave_names+=("$run_name")
    slot=$((slot + 1))

    if (( slot == ${#GPU_ARRAY[@]} )); then
      wait_wave
      slot=0
    fi
  done

  wait_wave
done

"$PYTHON_BIN" experiments/head_count_sweep/summarize_head_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --variant standard \
  --baseline-head 8 \
  --csv-output "reports/${BASE_RUN}_aggregate.csv" \
  --svg-output "reports/${BASE_RUN}_test_loss.svg"
