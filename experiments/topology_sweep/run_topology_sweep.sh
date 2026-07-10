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
N_LAYER="${N_LAYER:-8}"
N_UNIQUE_LAYERS="${N_UNIQUE_LAYERS:-$N_LAYER}"
N_HEAD="${N_HEAD:-8}"
N_EMBD="${N_EMBD:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BASE_RUN="${BASE_RUN:-enwik8_topology_sweep_${NORM}_${NORM_KIND}_${OPTIMIZER}_${N_LAYER}l_${N_EMBD}d_ctx${BLOCK_SIZE}_bs${BATCH_SIZE}_test005_100k_earlystop10_lrdecay30k}"

VARIANTS_STRING="${VARIANTS:-standard standard_fa parallel block_af block_fa block_af_carry block_fa_carry}"
SEEDS_STRING="${SEEDS:-1 2}"
read -r -a VARIANT_ARRAY <<< "$VARIANTS_STRING"
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

echo "scheduler: dynamic queue over ${#GPU_ARRAY[@]} GPU slot(s)."

compile_args=()
if [[ "$COMPILE" == "1" || "$COMPILE" == "true" ]]; then
  compile_args+=(--compile)
fi

fallback_args=()
if [[ "$OPTIMIZER" == "muon" ]]; then
  fallback_args+=(--adamw-fallback-learning-rate "$ADAMW_FALLBACK_LEARNING_RATE")
fi

mkdir -p runs reports
active_pids=()
active_names=()
active_gpus=()
available_gpus=("${GPU_ARRAY[@]}")
released_name=""
released_gpu=""

cleanup() {
  if [[ "${#active_pids[@]}" -gt 0 ]]; then
    echo "Stopping ${#active_pids[@]} active run(s)..." >&2
    kill "${active_pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

release_active_pid() {
  local pid="$1"
  local idx
  for idx in "${!active_pids[@]}"; do
    if [[ "${active_pids[$idx]}" == "$pid" ]]; then
      released_name="${active_names[$idx]}"
      released_gpu="${active_gpus[$idx]}"
      unset 'active_pids[idx]' 'active_names[idx]' 'active_gpus[idx]'
      set +u
      active_pids=("${active_pids[@]}")
      active_names=("${active_names[@]}")
      active_gpus=("${active_gpus[@]}")
      set -u
      available_gpus+=("$released_gpu")
      return 0
    fi
  done
  return 1
}

is_running_job() {
  local pid="$1"
  local running_pid
  while IFS= read -r running_pid; do
    if [[ "$running_pid" == "$pid" ]]; then
      return 0
    fi
  done < <(jobs -pr)
  return 1
}

wait_for_free_gpu() {
  local finished_pid=""
  local status=0
  local idx pid

  if [[ "${#active_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  if help wait 2>/dev/null | grep -q -- "-p"; then
    set +e
    wait -n -p finished_pid "${active_pids[@]}"
    status=$?
    set -e

    if [[ -z "$finished_pid" ]]; then
      echo "A training process finished, but bash did not report its pid." >&2
      exit 1
    fi
  else
    while true; do
      for idx in "${!active_pids[@]}"; do
        pid="${active_pids[$idx]}"
        if ! is_running_job "$pid"; then
          finished_pid="$pid"
          set +e
          wait "$finished_pid"
          status=$?
          set -e
          break 2
        fi
      done
      sleep 5
    done
  fi

  if ! release_active_pid "$finished_pid"; then
    echo "Finished process ${finished_pid}, but it was not tracked by the launcher." >&2
    exit 1
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "Run failed: ${released_name}" >&2
    exit "$status"
  fi

  echo "Finished ${released_name} on GPU ${released_gpu}."
}

for seed in "${SEED_ARRAY[@]}"; do
  for variant in "${VARIANT_ARRAY[@]}"; do
    run_name="${BASE_RUN}_seed${seed}_${variant}"
    run_dir="runs/block_residuals/${run_name}"
    log_path="runs/${run_name}.log"

    if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
      if [[ -s "${run_dir}/summary.csv" ]]; then
        echo "Skipping completed ${run_name}."
        continue
      fi
    fi

    while [[ "${#available_gpus[@]}" -eq 0 ]]; do
      wait_for_free_gpu
    done

    gpu="${available_gpus[0]}"
    available_gpus=("${available_gpus[@]:1}")

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

    active_pids+=("$!")
    active_names+=("$run_name")
    active_gpus+=("$gpu")
  done
done

while [[ "${#active_pids[@]}" -gt 0 ]]; do
  wait_for_free_gpu
done

"$PYTHON_BIN" experiments/topology_sweep/summarize_topology_sweep.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  --baseline-variant standard \
  --csv-output "reports/${BASE_RUN}_aggregate.csv"
