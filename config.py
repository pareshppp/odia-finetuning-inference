"""
Central configuration for the Odia fine-tuning / inference project.

Non-secret parameters live here as plain Python values so they are version
controlled and shared by every notebook (`notebooks/`) and script (`scripts/`).

Secrets are the only thing NOT stored here — `HF_TOKEN`, `COMET_API_KEY` and
`OPIK_API_KEY` are read from the environment (`.env`). See `.env.example`.

Usage:
    from config import *          # notebooks / scripts pull every value below
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets from .env (project root). find_dotenv-style search keeps this
# working whether imported from notebooks/ or scripts/.
load_dotenv()

# ---------------------------------------------------------------------------
# Secrets — read from the environment (.env), never hard-coded here
# ---------------------------------------------------------------------------
HF_TOKEN      = os.getenv("HF_TOKEN")
COMET_API_KEY = os.getenv("COMET_API_KEY")
OPIK_API_KEY  = os.getenv("OPIK_API_KEY")

# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------
HF_USERNAME = "pareshppp"          # used to construct default push targets

# ---------------------------------------------------------------------------
# Model + dataset (shared by eval / SFT / GRPO)
# ---------------------------------------------------------------------------
MODEL_ID       = "sarvamai/sarvam-1"
DATASET_ID     = "tripathysagar/odia-gsm8k"
DATASET_SPLIT  = "test"            # used by eval notebook
TRAIN_SPLIT    = "train"           # used by SFT + GRPO
FEW_SHOT_SPLIT = "train"
QUESTION_COL   = "odia_question"
ANSWER_COL     = "odia_answer"

# ---------------------------------------------------------------------------
# Evaluation (sarvam1_eval.ipynb)
# ---------------------------------------------------------------------------
NUM_FEW_SHOT       = 3             # Sarvam-1 is base; few-shot strongly recommended
MAX_NEW_TOKENS     = 1024          # Odia GSM8K solutions can run long; 512 truncated the '#### N' line on ~5%+ of samples
NUM_EVAL_SAMPLES   = 100           # -1 → evaluate all samples
DEVICE             = "cuda"        # cuda | cpu | mps
RESULTS_DIR        = Path("results")
EVAL_RUN_TAG       = ""            # base | sft | grpo — auto-inferred from MODEL_ID if blank
COMET_EVAL_PROJECT = "sarvam1-odia-gsm8k-eval"  # separate project → compare base/sft/grpo side-by-side

# ---------------------------------------------------------------------------
# SFT (sarvam1_sft.ipynb)
# ---------------------------------------------------------------------------
SFT_OUTPUT_DIR        = Path("/workspace/sarvam1-odia-gsm8k-sft")
SFT_HUB_MODEL_ID      = ""         # blank → {HF_USERNAME}/sarvam-1-odia-gsm8k-sft
SFT_ADAPTER_HUB_MODEL_ID = ""      # blank → {HF_USERNAME}/sarvam-1-odia-gsm8k-sft-adapter
PUSH_TO_HUB      = True
PRIVATE_REPO     = False
USE_QLORA        = True            # 4-bit base + LoRA; set False for full bf16 FT

NUM_EPOCHS    = 2
LEARNING_RATE = 3e-4              # bumped from 2e-4: packing puts ~5x more real tokens/step, so gradients are less noisy
BATCH_SIZE    = 16                # packing fills every seq to MAX_SEQ_LEN, so memory/seq is higher than unpacked — keep batch modest
GRAD_ACCUM    = 1                 # effective batch = BATCH_SIZE * GRAD_ACCUM = 16 packed sequences
MAX_SEQ_LEN   = 2048             # GSM8K solutions can exceed 1024 in Odia; truncating '#### N' kills SFT signal
USE_PACKING   = True             # pack multiple examples per seq (needs FlashAttention-2 for correct doc isolation)
WARMUP_RATIO  = 0.05             # packing cuts total steps sharply, so give warmup a slightly larger share
LOGGING_STEPS = 10
SAVE_STEPS    = 200
LORA_R        = 16
LORA_ALPHA    = 32

# ---------------------------------------------------------------------------
# GRPO (sarvam1_grpo.ipynb) — starts from the SFT model
# ---------------------------------------------------------------------------
SFT_MODEL_ID      = ""             # blank → SFT_HUB_MODEL_ID
GRPO_OUTPUT_DIR   = Path("/workspace/sarvam1-odia-gsm8k-grpo")
GRPO_HUB_MODEL_ID = ""             # blank → {HF_USERNAME}/sarvam-1-odia-gsm8k-grpo
GRPO_ADAPTER_HUB_MODEL_ID = ""     # blank → {HF_USERNAME}/sarvam-1-odia-gsm8k-grpo-adapter

USE_VLLM              = False      # True = much faster rollouts (requires vllm install, USE_QLORA=False)
GRPO_NUM_EPOCHS       = 1.0
GRPO_LEARNING_RATE    = 5e-6
GRPO_BATCH_SIZE       = 1
GRPO_GRAD_ACCUM       = 8
MAX_PROMPT_LEN        = 512
MAX_COMPLETION_LENGTH = 512
NUM_GENERATIONS       = 8          # group size G — must divide global batch
GRPO_BETA             = 0.04       # KL coefficient to reference (SFT) model
GRPO_LOGGING_STEPS    = 5          # GRPO logs more often than SFT (rollouts are slow/informative)
GRPO_SAVE_STEPS       = 100

# ---------------------------------------------------------------------------
# Tracking — Comet ML (metrics) + Opik (LLM traces). API keys are secrets above.
# ---------------------------------------------------------------------------
COMET_WORKSPACE    = "paresh-pradhan"
COMET_PROJECT_NAME = "odia-finetuning-inference"
GPU_TYPE           = "l40s"  # e.g. h100, l40, a100 — appended to run name and Comet tags
COMET_TAGS         = ""            # optional extra comma-separated tags (e.g. "experiment,v2")

OPIK_WORKSPACE     = COMET_WORKSPACE
OPIK_PROJECT_NAME  = COMET_PROJECT_NAME


# ---------------------------------------------------------------------------
# Derived defaults (resolved once, so every consumer sees the same values)
# ---------------------------------------------------------------------------
if not SFT_HUB_MODEL_ID:
    SFT_HUB_MODEL_ID = f"{HF_USERNAME}/sarvam-1-odia-gsm8k-sft" if HF_USERNAME else ""

if not SFT_ADAPTER_HUB_MODEL_ID:
    SFT_ADAPTER_HUB_MODEL_ID = f"{HF_USERNAME}/sarvam-1-odia-gsm8k-sft-adapter" if HF_USERNAME else ""

if not GRPO_HUB_MODEL_ID:
    GRPO_HUB_MODEL_ID = f"{HF_USERNAME}/sarvam-1-odia-gsm8k-grpo" if HF_USERNAME else ""

if not GRPO_ADAPTER_HUB_MODEL_ID:
    GRPO_ADAPTER_HUB_MODEL_ID = f"{HF_USERNAME}/sarvam-1-odia-gsm8k-grpo-adapter" if HF_USERNAME else ""

# GRPO starts from the SFT model — fall back to the SFT push target
if not SFT_MODEL_ID:
    SFT_MODEL_ID = SFT_HUB_MODEL_ID

# Infer the eval run tag from the model id when not set explicitly
if not EVAL_RUN_TAG:
    _mid = MODEL_ID.lower()
    EVAL_RUN_TAG = "grpo" if "grpo" in _mid else "sft" if "sft" in _mid else "base"


# Names exported by `from config import *` — only configuration values, so the
# import doesn't leak helpers (os, Path, load_dotenv) into notebook namespaces.
__all__ = [
    # secrets
    "HF_TOKEN", "COMET_API_KEY", "OPIK_API_KEY",
    # hugging face
    "HF_USERNAME",
    # model + dataset
    "MODEL_ID", "DATASET_ID", "DATASET_SPLIT", "TRAIN_SPLIT", "FEW_SHOT_SPLIT",
    "QUESTION_COL", "ANSWER_COL",
    # eval
    "NUM_FEW_SHOT", "MAX_NEW_TOKENS", "NUM_EVAL_SAMPLES", "DEVICE", "RESULTS_DIR",
    "EVAL_RUN_TAG", "COMET_EVAL_PROJECT",
    # sft
    "SFT_OUTPUT_DIR", "SFT_HUB_MODEL_ID", "SFT_ADAPTER_HUB_MODEL_ID", "PUSH_TO_HUB", "PRIVATE_REPO", "USE_QLORA",
    "NUM_EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "GRAD_ACCUM", "MAX_SEQ_LEN",
    "USE_PACKING", "WARMUP_RATIO", "LOGGING_STEPS", "SAVE_STEPS", "LORA_R", "LORA_ALPHA",
    # grpo
    "SFT_MODEL_ID", "GRPO_OUTPUT_DIR", "GRPO_HUB_MODEL_ID", "GRPO_ADAPTER_HUB_MODEL_ID", "USE_VLLM",
    "GRPO_NUM_EPOCHS", "GRPO_LEARNING_RATE", "GRPO_BATCH_SIZE", "GRPO_GRAD_ACCUM",
    "MAX_PROMPT_LEN", "MAX_COMPLETION_LENGTH", "NUM_GENERATIONS", "GRPO_BETA",
    "GRPO_LOGGING_STEPS", "GRPO_SAVE_STEPS",
    # tracking
    "COMET_WORKSPACE", "COMET_PROJECT_NAME", "GPU_TYPE", "COMET_TAGS",
    "OPIK_WORKSPACE", "OPIK_PROJECT_NAME",
]
