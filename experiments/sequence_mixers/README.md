# Linear Attention and SSM Mixer Sweep

This experiment compares token mixers against the same-budget standard
Transformer Muon baseline. The block topology stays standard
Attention/Mixer-then-FFN with pre-LayerNorm; only the attention branch is
replaced.

Default variants:

- `standard_linear_attn`: FLA `LinearAttention`.
- `standard_gla`: FLA `GatedLinearAttention`.
- `standard_retnet`: FLA `MultiScaleRetention`.
- `standard_mamba2`: FLA `Mamba2`.
- `standard_hadamard_qkv`: local causal Hadamard mixer
  \(o_i = q_i \odot \sum_{j \le i} k_j \odot v_j\).
- `standard_hadamard_qv`: local causal Hadamard mixer
  \(o_i = q_i \odot \sum_{j \le i} v_j\).

The standard baseline is the Muon run from the optimizer sweep:

```text
enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k
```

## Dependency

The FLA variants require the optional `flash-linear-attention` package in the
training environment. The runner checks `import fla.layers` before launching
FLA jobs and exits early if the package is missing. The local Hadamard variants
do not require FLA.

Example installation in the cluster venv:

```bash
source .venv_cu128/bin/activate
pip install -U flash-linear-attention
```

## Run

```bash
GPUS="0 1 2 3" experiments/sequence_mixers/run_sequence_mixers.sh
```

To test only the two local Hadamard mixers:

```bash
VARIANTS="standard_hadamard_qkv standard_hadamard_qv" \
CHECK_FLA=0 \
GPUS="0" \
experiments/sequence_mixers/run_sequence_mixers.sh
```

The default budget matches the Muon optimizer sweep:

- max steps: `100000`
- LR decay steps: `30000`
- early stop patience: `10`
- Muon main LR / min LR: `2e-3 / 2e-4`
- AdamW fallback LR: `2e-4`
- batch/context: `256 / 512`
- seeds: `1 2`

With four GPUs, the default script launches the four mixer variants for seed 1
as the first wave and seed 2 as the second wave.

## Monitor

```bash
BASE_RUN=enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k

tail -n 40 -f runs/${BASE_RUN}_seed*_standard_*.log
```

## Summarize

The runner automatically writes:

```text
results/enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k/
```

Manual summary:

```bash
BASE_RUN=enwik8_sequence_mixers_fla_muon_8l_512d_ctx512_bs256_lr2e3_test005_100k_earlystop10_lrdecay30k
STANDARD_BASE_RUN=enwik8_optimizer_sweep_standard_pre_layernorm_8l_512d_ctx512_bs256_test005_100k_earlystop10_lrdecay30k

python experiments/sequence_mixers/summarize_sequence_mixers.py \
  "runs/block_residuals/${BASE_RUN}_seed*/summary.csv" \
  "runs/block_residuals/${STANDARD_BASE_RUN}_seed*_muon_lr2e3/summary.csv" \
  --output-dir "results/${BASE_RUN}"
```
