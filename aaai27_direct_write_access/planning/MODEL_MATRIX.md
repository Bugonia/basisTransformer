# Model Matrix

The first open-model experiments should prioritize models that are small,
popular, easy to load through Hugging Face Transformers, and architecturally
diverse enough to test whether the write-economy framework transfers.

## Recommended Models

| Model ID | Approx scale | Architecture style | Use first for | Notes |
| --- | ---: | --- | --- | --- |
| `openai-community/gpt2` | 124M | GPT-2 | smoke tests, logit attribution | Mature baseline; Conv1D projections need path handling. |
| `EleutherAI/pythia-70m` | 70M | GPT-NeoX/Pythia | fastest real open-model test | Good for tool validation and coefficient logging. |
| `EleutherAI/pythia-160m` | 160M | GPT-NeoX/Pythia | small scaling check | Same family as 70M, still lightweight. |
| `EleutherAI/pythia-410m` | 410M | GPT-NeoX/Pythia | stronger small baseline | Useful before moving to modern 1B-class models. |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Qwen/Llama-like | modern base model | Good main model for attribution and path experiments. |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Qwen/Llama-like | scale-up after 0.5B | More expensive but still manageable on a single good GPU. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Llama-like | chat behavior and refusal | Prefer base variant for pure mechanism if available locally. |
| `google/gemma-2-2b` | 2B | Gemma | optional stronger model | May require accepting model terms before download. |

## Model Selection Principles

- Start with base models for mechanistic claims.
- Use chat models only when the phenomenon is chat-specific: refusal, safety,
  instruction following, or hallucination behavior.
- Keep one clean family for scaling. Pythia is useful because the family has
  several small sizes with similar architecture.
- Keep one modern family for relevance. Qwen2.5 small models are good because
  they are compact and widely used.
- Do not mix too many model families in the first paper. Use breadth only after
  the method works.

## First Three Runs

1. `EleutherAI/pythia-70m`
   - task: basis inventory;
   - expected output: projection path map and shape table.
2. `openai-community/gpt2`
   - task: compatibility check for GPT-2 projection modules;
   - expected output: same JSON schema as Pythia.
3. `Qwen/Qwen2.5-0.5B`
   - task: modern architecture compatibility;
   - expected output: Attention `o_proj`, MLP `down_proj`, unembedding map.

## Hardware Notes

- CPU is enough for basis inventory on very small models, but slow for repeated
  generation.
- A single 16-24 GB GPU should handle 0.5B-1.5B inference experiments with
  careful batch size and `torch_dtype=auto`.
- For coefficient trajectory datasets, stream prompts in small batches and
  save compact statistics rather than full activation tensors whenever possible.

