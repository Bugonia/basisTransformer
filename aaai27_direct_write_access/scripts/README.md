# Scripts

## `inspect_model_basis.py`

Locate architectural write-basis matrices in a Hugging Face causal language
model.

Example:

```bash
python aaai27_direct_write_access/scripts/inspect_model_basis.py \
  --model-id EleutherAI/pythia-70m \
  --output aaai27_direct_write_access/results/pythia-70m_basis.json \
  --device cpu
```

The script currently looks for common projection paths:

- GPT-NeoX/Pythia: Attention dense output and MLP dense-to-hidden output;
- GPT-2: attention `c_proj` and MLP `c_proj`;
- Llama/Qwen/Gemma-style models: Attention `o_proj` and MLP `down_proj`.

Future scripts:

- `capture_residual_writes.py`: hook Attention/FFN residual deltas;
- `basis_logit_attribution.py`: project writes through the unembedding;
- `coefficient_trajectory.py`: collect coefficient trajectory statistics;
- `basis_interventions.py`: zero, mask, or project away selected write bases.
