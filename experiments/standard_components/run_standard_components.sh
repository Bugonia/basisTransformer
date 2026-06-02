#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -n "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

BASE_RUN="${BASE_RUN:-enwik8_standard_components_sdpa_g1_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k}"
STANDARD_BASE_RUN="${STANDARD_BASE_RUN:-enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k}"
PYTHON_BIN="${PYTHON_BIN:-.venv_cu128/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_block_residuals.py}"
DATA_FILE="${DATA_FILE:-data/enwik8.txt}"
ENCODING="${ENCODING:-latin-1}"

VARIANTS_STRING="${VARIANTS:-standard_swiglu standard_gated_attn standard_swiglu_gated_attn}"
SEEDS_STRING="${SEEDS:-1 2}"
read -r -a VARIANT_ARRAY <<< "$VARIANTS_STRING"
read -r -a SEED_ARRAY <<< "$SEEDS_STRING"

NORM="${NORM:-pre}"
NORM_KIND="${NORM_KIND:-layernorm}"
OPTIMIZER="${OPTIMIZER:-muon}"
N_LAYER="${N_LAYER:-8}"
N_HEAD="${N_HEAD:-8}"
N_EMBD="${N_EMBD:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
MAX_ITERS="${MAX_ITERS:-100000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-30000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
VAL_FRAC="${VAL_FRAC:-0.005}"
TEST_FRAC="${TEST_FRAC:-0.005}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
ADAMW_FALLBACK_LEARNING_RATE="${ADAMW_FALLBACK_LEARNING_RATE:-2e-4}"
DROPOUT="${DROPOUT:-0.1}"
DTYPE="${DTYPE:-bfloat16}"
COMPILE="${COMPILE:-0}"
RESUME="${RESUME:-1}"

if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "muon" ]]; then
  echo "Unknown OPTIMIZER='${OPTIMIZER}'. Expected adamw or muon." >&2
  exit 1
fi

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

fallback_args=()
if [[ "$OPTIMIZER" == "muon" ]]; then
  fallback_args+=(--adamw-fallback-learning-rate "$ADAMW_FALLBACK_LEARNING_RATE")
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
      --norm "$NORM" \
      --norm-kind "$NORM_KIND" \
      --optimizer "$OPTIMIZER" \
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
      --lr-decay-iters "$LR_DECAY_ITERS" \
      --weight-decay "$WEIGHT_DECAY" \
      --beta1 "$BETA1" \
      --beta2 "$BETA2" \
      --muon-momentum "$MUON_MOMENTUM" \
      --muon-ns-steps "$MUON_NS_STEPS" \
      "${fallback_args[@]}" \
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

standard_pattern="runs/block_residuals/${STANDARD_BASE_RUN}_seed*_muon_lr2e3/summary.csv"
if ! compgen -G "$standard_pattern" >/dev/null; then
  echo "Missing same-budget standard Muon baseline summaries: ${standard_pattern}" >&2
  echo "Run experiments/optimizer_sweep/run_optimizer_sweep.sh or set STANDARD_BASE_RUN." >&2
  exit 1
fi

"$PYTHON_BIN" experiments/standard_components/summarize_standard_components.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  "$standard_pattern" \
  --output-dir "results/${BASE_RUN}"

"$PYTHON_BIN" plot_results_svg.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  "$standard_pattern" \
  --baseline standard \
  --title "Standard Transformer Component Ablation" \
  --output "reports/${BASE_RUN}.svg"
