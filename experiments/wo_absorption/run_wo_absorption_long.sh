#!/usr/bin/env bash
set -euo pipefail

export BASE_RUN="${BASE_RUN:-enwik8_wo_absorption_8l_512d_ctx512_bs256_lr2e4_test005_drop0_100k_earlystop10}"
export MAX_ITERS="${MAX_ITERS:-100000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export EVAL_ITERS="${EVAL_ITERS:-20}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
export DROPOUT="${DROPOUT:-0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_wo_absorption.sh"
