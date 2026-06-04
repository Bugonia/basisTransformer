# LLM RL Alignment Reproduction

This experiment folder is for small, comparable reproductions of mainstream
LLM post-training methods:

- `sft`: supervised fine-tuning baseline.
- `dpo`: direct preference optimization on chosen/rejected pairs.
- `grpo`: outcome-reward RL for math/reasoning-style prompts.
- `ppo + reward model`: recommended as a second-stage extension once the
  reward-model baseline is stable.

The main question is not "can one run RL", but "what changes when the same base
model, data budget, and evaluation prompts are exposed to RL-style objectives?"

## Recommended Comparison

Start with a small causal LM and LoRA adapters so the experiment is cheap and
repeatable.

```bash
python3 -m venv .venv-llm-rl
.venv-llm-rl/bin/python -m pip install -r experiments/llm_rl_alignment/requirements-llm-rl.txt
```

If you are already inside a CUDA/Torch environment such as `.venv_cu128`, keep
that PyTorch installation and install only the LLM alignment packages:

```bash
python -m pip install -r experiments/llm_rl_alignment/requirements-llm-rl-no-torch.txt
```

On restricted networks, use a reachable PyPI mirror:

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r experiments/llm_rl_alignment/requirements-llm-rl-no-torch.txt
```

If the training node has no working PyPI access, build a wheelhouse on an
online machine with the same Python version, copy it to the training node, and
install from local files:

```bash
python -m pip download \
  -r experiments/llm_rl_alignment/requirements-llm-rl-no-torch.txt \
  -d wheelhouse_llm_rl

python -m pip install --no-index --find-links wheelhouse_llm_rl \
  -r experiments/llm_rl_alignment/requirements-llm-rl-no-torch.txt
```

Use the same base model for every method:

```bash
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export OUT_ROOT=runs/llm_rl_alignment/qwen25_05b
```

Run an SFT baseline:

```bash
.venv-llm-rl/bin/python experiments/llm_rl_alignment/scripts/train_sft.py \
  --model "$BASE_MODEL" \
  --dataset trl-lib/Capybara \
  --output-dir "$OUT_ROOT/sft" \
  --max-samples 2000 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16
```

Run DPO on preference pairs:

```bash
.venv-llm-rl/bin/python experiments/llm_rl_alignment/scripts/train_dpo.py \
  --model "$BASE_MODEL" \
  --dataset trl-lib/ultrafeedback_binarized \
  --output-dir "$OUT_ROOT/dpo" \
  --max-samples 2000 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --beta 0.1
```

Run GRPO on answerable reasoning prompts:

```bash
.venv-llm-rl/bin/python experiments/llm_rl_alignment/scripts/train_grpo_math.py \
  --model "$BASE_MODEL" \
  --dataset trl-lib/DeepMath-103K \
  --output-dir "$OUT_ROOT/grpo" \
  --max-samples 1000 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-generations 4
```

Generate from the base model and adapters on the same prompts:

```bash
.venv-llm-rl/bin/python experiments/llm_rl_alignment/scripts/evaluate_generations.py \
  --prompt-file experiments/llm_rl_alignment/eval/smoke_prompts.jsonl \
  --model base="$BASE_MODEL" \
  --model sft="$OUT_ROOT/sft" \
  --model dpo="$OUT_ROOT/dpo" \
  --model grpo="$OUT_ROOT/grpo" \
  --output "$OUT_ROOT/smoke_generations.jsonl"
```

Summarize exact-match and length shifts:

```bash
.venv-llm-rl/bin/python experiments/llm_rl_alignment/scripts/summarize_generations.py \
  "$OUT_ROOT/smoke_generations.jsonl"
```

## What To Measure

Use multiple metrics, because RL often improves one axis while hurting another.

- Helpfulness and instruction following: pairwise preference win rate, ideally
  with a held-out human or model judge.
- Reasoning: exact-match or verifier accuracy on held-out math/code tasks.
- Verbosity/style drift: length, refusal rate, repetition, and format breakage.
- Base capability retention: perplexity or task accuracy on a non-alignment
  validation set.
- Reward hacking symptoms: high reward-model score with worse factuality,
  overlong answers, copied templates, or brittle formatting.

## Experimental Controls

Keep these fixed across methods:

- Base model checkpoint.
- Tokenizer and chat template.
- Training-token budget.
- LoRA rank/target modules.
- Prompt set for evaluation.
- Decoding parameters.
- Random seeds, when possible.

For a first pass, prefer small controlled runs over a large one-off run. A
convincing result is a table like:

| model | training tokens | preference win rate | math exact match | avg tokens | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| base | 0 | baseline | baseline | baseline | raw model |
| sft | same | ... | ... | ... | imitation only |
| dpo | same | ... | ... | ... | preference objective |
| grpo | same | ... | ... | ... | outcome reward |

## PPO Extension

Classic RLHF usually needs three stages: SFT policy, reward model, then PPO.
Reproduce it after DPO/GRPO are working, because it has more moving parts:

1. Train an SFT model.
2. Train a reward model on chosen/rejected responses.
3. Run PPO with the SFT model as policy and a frozen reference model.
4. Evaluate the PPO policy against the same prompt suite.

The folder currently focuses on SFT, DPO, and GRPO because they are the fastest
to make comparable on a single workstation or small GPU node.
