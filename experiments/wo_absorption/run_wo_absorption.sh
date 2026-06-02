#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -n "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

BASE_RUN="${BASE_RUN:-enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_30k}"
PYTHON_BIN="${PYTHON_BIN:-.venv_cu128/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_block_residuals.py}"
DATA_FILE="${DATA_FILE:-data/enwik8.txt}"
ENCODING="${ENCODING:-latin-1}"

VARIANTS_STRING="${VARIANTS:-block_af block_af_no_mid_ln block_af_no_mid_ln_no_wo}"
SEEDS_STRING="${SEEDS:-1 2}"
read -r -a VARIANT_ARRAY <<< "$VARIANTS_STRING"
read -r -a SEED_ARRAY <<< "$SEEDS_STRING"

N_LAYER="${N_LAYER:-8}"
N_HEAD="${N_HEAD:-8}"
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
DROPOUT="${DROPOUT:-0.0}"
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

slot=0
for seed in "${SEED_ARRAY[@]}"; do
  for variant in "${VARIANT_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$slot]}"
    run_name="${BASE_RUN}_seed${seed}_${variant}"
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
      --variant "$variant" \
      --run-name "$run_name" \
      --seed "$seed" \
      --max-iters "$MAX_ITERS" \
      --eval-interval "$EVAL_INTERVAL" \
      --eval-iters "$EVAL_ITERS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --val-frac "$VAL_FRAC" \
      --test-frac "$TEST_FRAC" \
      --n-layer "$N_LAYER" \
      --n-head "$N_HEAD" \
      --n-embd "$N_EMBD" \
      --batch-size "$BATCH_SIZE" \
      --block-size "$BLOCK_SIZE" \
      --dropout "$DROPOUT" \
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
done

wait_wave

"$PYTHON_BIN" summarize_runs.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --paired-baseline block_af_no_mid_ln \
  > "reports/${BASE_RUN}_aggregate.csv"

"$PYTHON_BIN" plot_results_svg.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline block_af_no_mid_ln \
  --title "Block-AF W_O Absorption Experiment" \
  --output "reports/${BASE_RUN}.svg"
