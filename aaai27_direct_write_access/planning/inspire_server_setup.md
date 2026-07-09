# Inspire Studio Server Setup

Use the online CPU instance to place code, Python environment, package cache,
Hugging Face cache, and enwik8 data under the personal global directory:

```text
/inspire/hdd/global_user/zhongxiaoqiu-253108120179
```

The offline GPU instance can then reuse the same directory.

## 1. Initialize Global Paths

```bash
set -euo pipefail

GLOBAL=/inspire/hdd/global_user/zhongxiaoqiu-253108120179
mkdir -p "$GLOBAL"/{code,envs,cache/pip,cache/huggingface,cache/torch,cache/xdg,data,wheelhouse/cu128,wheelhouse/pypi}

cat > "$GLOBAL/basis_env.sh" <<'EOF'
export GLOBAL=/inspire/hdd/global_user/zhongxiaoqiu-253108120179
export PROJECT_HOME=$GLOBAL/code/basisTransformer
export HF_HOME=$GLOBAL/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$GLOBAL/cache/pip
export TORCH_HOME=$GLOBAL/cache/torch
export XDG_CACHE_HOME=$GLOBAL/cache/xdg
export DATA_FILE=$GLOBAL/data/enwik8.txt
export PYTHONPATH=$PROJECT_HOME:${PYTHONPATH:-}
EOF

source "$GLOBAL/basis_env.sh"
```

## 2. Clone or Update the Repository

```bash
if [ -d "$PROJECT_HOME/.git" ]; then
  cd "$PROJECT_HOME"
  git pull --ff-only
else
  git clone https://github.com/Bugonia/basisTransformer.git "$PROJECT_HOME"
  cd "$PROJECT_HOME"
fi
```

If HTTPS authentication is needed, use a GitHub personal access token or set up
SSH keys on the online CPU instance.

## 3. Build a Reusable Python Environment

Install CUDA PyTorch even on the CPU instance. A GPU is not required for
installing CUDA wheels; the later offline GPU instance should see CUDA if the
driver is compatible.

```bash
python3 -m venv "$GLOBAL/envs/basis-transformer-cu128"
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cu128.txt
python -m pip install -r aaai27_direct_write_access/requirements.txt
```

Sanity check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available now:", torch.cuda.is_available())
PY
```

On the CPU instance, `cuda available now` may be `False`; that is expected.

## 4. Prepare Offline Wheelhouse Fallback

This is useful if the offline GPU instance cannot reuse the venv cleanly.

```bash
python -m pip download -d "$GLOBAL/wheelhouse/cu128" -r requirements-cu128.txt
grep -v '^torch' aaai27_direct_write_access/requirements.txt > /tmp/basis_req_no_torch.txt
python -m pip download -d "$GLOBAL/wheelhouse/pypi" -r /tmp/basis_req_no_torch.txt
```

Offline reinstall fallback:

```bash
source "$GLOBAL/basis_env.sh"
python3 -m venv "$GLOBAL/envs/basis-transformer-cu128"
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
python -m pip install --no-index \
  --find-links "$GLOBAL/wheelhouse/cu128" \
  --find-links "$GLOBAL/wheelhouse/pypi" \
  torch transformers accelerate safetensors numpy pandas tqdm einops datasets
```

## 5. Download Data and Model Caches

```bash
source "$GLOBAL/basis_env.sh"
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
cd "$PROJECT_HOME"

python prepare_enwik8.py --data-dir "$GLOBAL/data"

if [ ! -e "$PROJECT_HOME/data" ]; then
  ln -s "$GLOBAL/data" "$PROJECT_HOME/data"
fi
```

Prepare robustness corpora for the second-dataset checks:

```bash
python aaai27_direct_write_access/scripts/prepare_robustness_corpora.py \
  --data-dir "$GLOBAL/data" \
  --corpus wikitext103

python aaai27_direct_write_access/scripts/prepare_robustness_corpora.py \
  --data-dir "$GLOBAL/data" \
  --corpus fineweb_edu \
  --fineweb-chars 100000000
```

This creates:

```text
$GLOBAL/data/wikitext103.txt
$GLOBAL/data/wikitext103.txt.meta.json
$GLOBAL/data/fineweb_edu_100m.txt
$GLOBAL/data/fineweb_edu_100m.txt.meta.json
```

Pre-download open models for offline diagnostics:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    "EleutherAI/pythia-70m",
    "openai-community/gpt2",
    "Qwen/Qwen2.5-0.5B",
]

for model_id in models:
    print("downloading", model_id)
    snapshot_download(model_id)
PY
```

Run a quick basis-inventory check:

```bash
python aaai27_direct_write_access/scripts/inspect_model_basis.py \
  --model-id EleutherAI/pythia-70m \
  --output aaai27_direct_write_access/results/pythia-70m_basis.json \
  --device cpu
```

## 6. Offline GPU Instance Usage

On the offline GPU instance:

```bash
GLOBAL=/inspire/hdd/global_user/zhongxiaoqiu-253108120179
source "$GLOBAL/basis_env.sh"
source "$GLOBAL/envs/basis-transformer-cu128/bin/activate"
cd "$PROJECT_HOME"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

Launch the submission-grade five-seed topology rerun:

```bash
GPUS="0 1 2 3" \
SEEDS="1 2 3 4 5" \
DATA_FILE="$GLOBAL/data/enwik8.txt" \
BASE_RUN="aaai27_topology_sweep_pre_layernorm_muon_8l_512d_ctx512_bs256_5seed" \
bash experiments/topology_sweep/run_topology_sweep.sh
```

Adjust `GPUS` to match the allocated GPU ids.
