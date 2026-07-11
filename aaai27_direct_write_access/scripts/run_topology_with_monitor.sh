#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-aaai27_enwik8_topology_muon_8l_512d_ctx512_bs256_5seed}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
REPORT_DIR="${REPORT_DIR:-reports}"
RUNS_DIR="${RUNS_DIR:-runs/block_residuals}"
PLAIN_LOG_DIR="${PLAIN_LOG_DIR:-runs}"

mkdir -p "$REPORT_DIR" "$PLAIN_LOG_DIR"

launcher_log="${REPORT_DIR}/${BASE_RUN}_launcher.log"
monitor_html="${REPORT_DIR}/${BASE_RUN}_monitor.html"
monitor_txt="${REPORT_DIR}/${BASE_RUN}_monitor.txt"
aggregate_csv="${REPORT_DIR}/${BASE_RUN}_aggregate.csv"
report_svg="${REPORT_DIR}/${BASE_RUN}_report.svg"

echo "=== AAAI topology run with monitor ==="
echo "root: $ROOT_DIR"
echo "base_run: $BASE_RUN"
echo "python: $PYTHON_BIN"
echo "data_file: ${DATA_FILE:-data/enwik8.txt}"
echo "variants: ${VARIANTS:-standard standard_fa parallel block_af block_fa block_af_carry block_fa_carry}"
echo "seeds: ${SEEDS:-1 2}"
echo "write_rank: ${WRITE_RANK:-0}"
echo "write_alpha: ${WRITE_ALPHA:-1.0}"
echo "reports:"
echo "  launcher log: $launcher_log"
echo "  monitor html: $monitor_html"
echo "  monitor text: $monitor_txt"
echo "  aggregate csv: $aggregate_csv"
echo "  report svg: $report_svg"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "visible GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
  echo "nvidia-smi not found; the training script will fall back to GPUS=0 unless GPUS is set."
fi

bash experiments/topology_sweep/run_topology_sweep.sh >"$launcher_log" 2>&1 &
train_pid=$!
echo "training launcher pid: $train_pid"

summarize_if_ready() {
  "$PYTHON_BIN" experiments/topology_sweep/summarize_topology_sweep.py \
    "${RUNS_DIR}/${BASE_RUN}_seed*/summary.csv" \
    --baseline-variant standard \
    --csv-output "$aggregate_csv" \
    >"${REPORT_DIR}/${BASE_RUN}_aggregate_latest.txt" 2>&1 || true
}

plot_if_ready() {
  "$PYTHON_BIN" plot_results_svg.py \
    "${RUNS_DIR}/${BASE_RUN}_seed*/summary.csv" \
    --output "$report_svg" \
    --baseline standard \
    --title "$BASE_RUN" \
    >"${REPORT_DIR}/${BASE_RUN}_plot_latest.txt" 2>&1 || true
}

monitor_once() {
  {
    echo
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    fi
    echo
    "$PYTHON_BIN" monitor_runs.py \
      --base-run "$BASE_RUN" \
      --runs-dir "$RUNS_DIR" \
      --plain-log-dir "$PLAIN_LOG_DIR" \
      --html "$monitor_html"
    echo
    echo "recent launcher log:"
    tail -n 40 "$launcher_log" || true
    echo
    echo "generated files:"
    ls -lh "$monitor_html" "$monitor_txt" "$aggregate_csv" "$report_svg" 2>/dev/null || true
  } | tee "$monitor_txt"
}

while kill -0 "$train_pid" 2>/dev/null; do
  summarize_if_ready
  plot_if_ready
  monitor_once
  sleep "$MONITOR_INTERVAL"
done

set +e
wait "$train_pid"
train_status=$?
set -e

summarize_if_ready
plot_if_ready
monitor_once

echo "training launcher exit status: $train_status"
exit "$train_status"
