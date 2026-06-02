#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -n "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-.venv_cu128/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_block_residuals.py}"
DATA_FILE="${DATA_FILE:-data/enwik8.txt}"
ENCODING="${ENCODING:-latin-1}"

NORM="${NORM:-pre}"
NORM_KIND="${NORM_KIND:-layernorm}"
OPTIMIZER="${OPTIMIZER:-muon}"
VARIANT="${VARIANT:-standard}"
N_LAYER="${N_LAYER:-8}"
N_UNIQUE_LAYERS="${N_UNIQUE_LAYERS:-$N_LAYER}"
N_HEAD="${N_HEAD:-8}"
N_EMBD="${N_EMBD:-512}"
QK_N_BANDS="${QK_N_BANDS:-4}"
QK_BAND_MODE="${QK_BAND_MODE:-fixed}"
if [[ "$QK_BAND_MODE" != "learned" && "$QK_BAND_MODE" != "fixed" ]]; then
  echo "Unknown QK_BAND_MODE='${QK_BAND_MODE}'. Expected learned or fixed." >&2
  exit 1
fi

if [[ "$QK_BAND_MODE" == "fixed" && -z "${QK_BAND_SCALES:-}" ]]; then
  case "$QK_N_BANDS" in
    4)
      QK_BAND_SCALES="0.8,0.6,0.4,0.2"
      ;;
    8)
      QK_BAND_SCALES="0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2"
      ;;
    *)
      echo "Set QK_BAND_SCALES for QK_N_BANDS=${QK_N_BANDS}." >&2
      exit 1
      ;;
  esac
fi

if [[ "$QK_BAND_MODE" == "fixed" ]]; then
  QK_SCALE_SLUG="${QK_BAND_SCALES//,/-}"
  QK_SCALE_SLUG="${QK_SCALE_SLUG//./p}"
  QK_SCALE_SLUG="${QK_SCALE_SLUG// /}"
  QK_TAG="fixed_bands${QK_N_BANDS}_s${QK_SCALE_SLUG}"
else
  QK_TAG="learned_bands${QK_N_BANDS}"
fi
BATCH_SIZE="${BATCH_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BASE_RUN="${BASE_RUN:-enwik8_band_qk_${VARIANT}_${NORM}_${NORM_KIND}_${OPTIMIZER}_${N_LAYER}l_${N_EMBD}d_ctx${BLOCK_SIZE}_bs${BATCH_SIZE}_${QK_TAG}_test005_100k_earlystop10_lrdecay30k}"

QK_SCORES_STRING="${QK_SCORES:-band}"
SEEDS_STRING="${SEEDS:-1 2}"
read -r -a QK_SCORE_ARRAY <<< "$QK_SCORES_STRING"
read -r -a SEED_ARRAY <<< "$SEEDS_STRING"

MAX_ITERS="${MAX_ITERS:-100000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-30000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
VAL_FRAC="${VAL_FRAC:-0.005}"
TEST_FRAC="${TEST_FRAC:-0.005}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
if [[ -z "${LEARNING_RATE:-}" ]]; then
  if [[ "$OPTIMIZER" == "muon" ]]; then
    LEARNING_RATE="2e-3"
  else
    LEARNING_RATE="2e-4"
  fi
fi
if [[ -z "${MIN_LR:-}" ]]; then
  if [[ "$OPTIMIZER" == "muon" ]]; then
    MIN_LR="2e-4"
  else
    MIN_LR="2e-5"
  fi
fi
if [[ -z "${WEIGHT_DECAY:-}" ]]; then
  if [[ "$OPTIMIZER" == "muon" ]]; then
    WEIGHT_DECAY="0.01"
  else
    WEIGHT_DECAY="0.1"
  fi
fi
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
  for qk_score in "${QK_SCORE_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$slot]}"
    run_name="${BASE_RUN}_seed${seed}_${qk_score}"
    run_dir="runs/block_residuals/${run_name}"
    log_path="runs/${run_name}.log"
    qk_args=(--qk-score "$qk_score" --qk-n-bands "$QK_N_BANDS")
    if [[ "$qk_score" == "band" ]]; then
      qk_args+=(--qk-band-mode "$QK_BAND_MODE")
      if [[ "$QK_BAND_MODE" == "fixed" ]]; then
        qk_args+=(--qk-band-scales "$QK_BAND_SCALES")
      fi
    fi

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
      --variant "$VARIANT" \
      --norm "$NORM" \
      --norm-kind "$NORM_KIND" \
      --optimizer "$OPTIMIZER" \
      "${qk_args[@]}" \
      --run-name "$run_name" \
      --seed "$seed" \
      --max-iters "$MAX_ITERS" \
      --eval-interval "$EVAL_INTERVAL" \
      --eval-iters "$EVAL_ITERS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --val-frac "$VAL_FRAC" \
      --test-frac "$TEST_FRAC" \
      --n-layer "$N_LAYER" \
      --n-unique-layers "$N_UNIQUE_LAYERS" \
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

"$PYTHON_BIN" experiments/band_qk/summarize_band_qk.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
