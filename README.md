# Odia GSM8K — Sarvam-1 Fine-tuning & Evaluation

Fine-tune and evaluate [`ai4bharat/sarvam-1`](https://huggingface.co/ai4bharat/sarvam-1) (2B Indic-language base model) on the [`tripathysagar/odia-gsm8k`](https://huggingface.co/datasets/tripathysagar/odia-gsm8k) math reasoning dataset.

Three-stage pipeline:

| Stage | Notebook | Output model |
|-------|----------|-------------|
| Baseline eval | `notebooks/sarvam1_eval.ipynb` | — |
| SFT (QLoRA) | `notebooks/sarvam1_sft.ipynb` | `{HF_USERNAME}/sarvam-1-odia-gsm8k-sft` |
| GRPO (RL) | `notebooks/sarvam1_grpo.ipynb` | `{HF_USERNAME}/sarvam-1-odia-gsm8k-grpo` |

Each eval run is logged to Comet ML for side-by-side comparison across all three models.

---

## Metrics tracked

**Correctness**
- Final-answer accuracy (exact match on `#### N`)
- Accuracy among format-compliant outputs
- Format adherence (fraction producing `#### N`)
- Odia language adherence (Odia script ratio, pure-Odia %, English-drift %)

**Latency**
- TTFT (time to first token) — p50 / p95 / p99
- TPOT (time per output token) — p50 / p95
- E2E latency — p50 / p95 / p99
- Throughput (tokens/sec)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/pareshppp/odia-finetuning-inference.git
cd odia-finetuning-inference
pip install transformers datasets accelerate peft bitsandbytes trl \
            torch python-dotenv pandas matplotlib seaborn tqdm comet_ml opik
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in at minimum:
#   HF_TOKEN, HF_USERNAME
#   COMET_API_KEY, COMET_WORKSPACE   (optional but recommended)
```

Key variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `ai4bharat/sarvam-1` | Model to evaluate or start training from |
| `DATASET_ID` | `tripathysagar/odia-gsm8k` | HuggingFace dataset |
| `NUM_FEW_SHOT` | `3` | Few-shot examples for base model eval (set `0` for SFT/GRPO) |
| `NUM_EVAL_SAMPLES` | `100` | Samples to evaluate (`-1` = all) |
| `USE_QLORA` | `true` | 4-bit QLoRA for SFT (set `false` for full bf16 fine-tuning) |
| `EVAL_RUN_TAG` | *(auto)* | `base` / `sft` / `grpo` — inferred from `MODEL_ID` if blank |
| `COMET_EVAL_PROJECT` | `sarvam1-odia-gsm8k-eval` | Comet project for eval comparison dashboard |

---

## Run order (on RunPod A100 / H100)

### Step 1 — Base model baseline

```bash
# In .env: MODEL_ID=ai4bharat/sarvam-1, NUM_FEW_SHOT=3
jupyter nbconvert --to notebook --execute notebooks/sarvam1_eval.ipynb
```

### Step 2 — SFT fine-tuning

```bash
# Trains QLoRA, merges adapter, pushes to HF Hub
jupyter nbconvert --to notebook --execute notebooks/sarvam1_sft.ipynb
```

### Step 3 — Eval SFT model

```bash
# In .env: MODEL_ID={HF_USERNAME}/sarvam-1-odia-gsm8k-sft, NUM_FEW_SHOT=0
jupyter nbconvert --to notebook --execute notebooks/sarvam1_eval.ipynb
```

### Step 4 — GRPO fine-tuning

```bash
# Starts from the SFT model, trains with verifiable rewards
jupyter nbconvert --to notebook --execute notebooks/sarvam1_grpo.ipynb
```

### Step 5 — Eval GRPO model

```bash
# In .env: MODEL_ID={HF_USERNAME}/sarvam-1-odia-gsm8k-grpo, NUM_FEW_SHOT=0
jupyter nbconvert --to notebook --execute notebooks/sarvam1_eval.ipynb
```

After all three eval runs, open the `sarvam1-odia-gsm8k-eval` project in Comet ML and use **Compare** to view base / SFT / GRPO side-by-side.

---

## Training defaults (SFT)

Tuned for A100 40 GB / H100 80 GB with QLoRA:

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Learning rate | 2e-4 |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch = 16) |
| Max sequence length | 2048 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA targets | q/k/v/o/gate/up/down projections |

## Training defaults (GRPO)

| Hyperparameter | Value |
|----------------|-------|
| Base model | SFT model |
| Epochs | 1 |
| Learning rate | 5e-6 |
| Group size (G) | 8 |
| KL coefficient (beta) | 0.04 |
| Max prompt length | 512 |
| Max completion length | 512 |

---

## Experiment tracking

- **Training metrics** (loss, learning rate, rewards): Comet ML project `sarvam1-odia-gsm8k`
- **Eval metrics** (accuracy, latency, format/language adherence): Comet ML project `sarvam1-odia-gsm8k-eval`
- **LLM traces** (sanity-check rollouts): Opik (optional; set `OPIK_API_KEY`)

---

## Results directory

Each eval run writes to `results/` (configurable via `RESULTS_DIR`):

```
results/
  eval_results.csv       # per-sample predictions + latency
  eval_summary.json      # aggregate metrics
  eval_plots.png         # accuracy + latency distribution charts
```
