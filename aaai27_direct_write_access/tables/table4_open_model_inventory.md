# Table 4 Draft: Open-Model Basis Inventory

Status:

- waiting for `inspect_model_basis.py` runs.

| Model | Layers | Hidden size | Attention output basis path | FFN output basis path | Unembedding shape | Status |
| --- | ---: | ---: | --- | --- | --- | --- |
| `EleutherAI/pythia-70m` | TBD | TBD | TBD | TBD | TBD | not run |
| `openai-community/gpt2` | TBD | TBD | TBD | TBD | TBD | not run |
| `Qwen/Qwen2.5-0.5B` | TBD | TBD | TBD | TBD | TBD | optional |

Paper message:

> The same write-basis decomposition maps onto mainstream pretrained
> decoder-only models.

Command template:

```bash
python aaai27_direct_write_access/scripts/inspect_model_basis.py \
  --model-id EleutherAI/pythia-70m \
  --output aaai27_direct_write_access/results/pythia-70m_basis.json \
  --device cpu
```
