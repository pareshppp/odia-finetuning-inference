"""
Sarvam-1 GRPO on Odia GSM8K
-----------------------------
Group Relative Policy Optimization starting from the SFT-tuned model,
using verifiable rewards on the train split of tripathysagar/odia-gsm8k.

Usage (RunPod):
    screen -S grpo
    cd /workspace && python grpo_train.py
    # Ctrl-A D to detach; screen -r grpo to reattach

Non-secret config lives in config.py; secrets (HF_TOKEN, COMET_API_KEY,
OPIK_API_KEY) in .env — see .env.example.
Requires SFT_MODEL_ID / SFT_HUB_MODEL_ID (or HF_USERNAME) set in config.py.
"""

import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Make the repo root importable so `config.py` resolves regardless of CWD,
# then pull in every non-secret parameter (config.py also loads .env secrets).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *  # noqa: F401,F403

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

# Import comet_ml *before* torch so Comet can auto-instrument the framework.
import comet_ml

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login as hf_login, create_repo

print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


# ---------------------------------------------------------------------------
# Configuration — all values come from config.py (imported above).
# A few local aliases keep the names this script already uses downstream.
# ---------------------------------------------------------------------------
NUM_EPOCHS       = GRPO_NUM_EPOCHS
LEARNING_RATE    = GRPO_LEARNING_RATE
BATCH_SIZE       = GRPO_BATCH_SIZE
GRAD_ACCUM       = GRPO_GRAD_ACCUM
MAX_COMPLETION   = MAX_COMPLETION_LENGTH
BETA             = GRPO_BETA
LOGGING_STEPS    = GRPO_LOGGING_STEPS
SAVE_STEPS       = GRPO_SAVE_STEPS
COMET_PROJECT    = COMET_PROJECT_NAME
COMET_TAGS_EXTRA = COMET_TAGS

# Fail fast — don't waste hours on misconfiguration
assert SFT_MODEL_ID, "Set SFT_MODEL_ID / SFT_HUB_MODEL_ID (or HF_USERNAME) in config.py before running."

if PUSH_TO_HUB:
    assert GRPO_HUB_MODEL_ID, (
        "PUSH_TO_HUB=true but neither GRPO_HUB_MODEL_ID nor HF_USERNAME is set. "
        "Set one before training, or set PUSH_TO_HUB=false."
    )
    assert GRPO_ADAPTER_HUB_MODEL_ID, (
        "PUSH_TO_HUB=true but GRPO_ADAPTER_HUB_MODEL_ID could not be resolved. "
        "Set HF_USERNAME or GRPO_ADAPTER_HUB_MODEL_ID in config.py."
    )

# vLLM + QLoRA is unstable in current TRL — vLLM materializes its own model copy
# that conflicts with quantized weights and adapter merge during rollouts
if USE_VLLM and USE_QLORA:
    raise RuntimeError(
        "USE_VLLM=true with USE_QLORA=true is not supported in this script. "
        "vLLM rollouts require an un-quantized model for inference. "
        "Set USE_QLORA=false (full bf16 LoRA training) or USE_VLLM=false."
    )

GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"SFT model     : {SFT_MODEL_ID}")
print(f"Dataset       : {DATASET_ID} ({TRAIN_SPLIT})")
print(f"Output dir    : {GRPO_OUTPUT_DIR}")
print(f"Hub repo      : {GRPO_HUB_MODEL_ID or '(unset)'}  push={PUSH_TO_HUB}")
print(f"QLoRA         : {USE_QLORA}  vLLM rollouts: {USE_VLLM}")
print(f"G={NUM_GENERATIONS}  beta={BETA}  lr={LEARNING_RATE}  bs={BATCH_SIZE}x{GRAD_ACCUM}")


# ---------------------------------------------------------------------------
# Auth + tracking
# ---------------------------------------------------------------------------
if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("HF Hub : logged in")

if COMET_API_KEY:
    tags = ["grpo", "qlora" if USE_QLORA else "bf16"]
    if GPU_TYPE:
        tags.append(GPU_TYPE)
    if COMET_TAGS_EXTRA:
        tags.extend(t.strip() for t in COMET_TAGS_EXTRA.split(",") if t.strip())
    os.environ["COMET_PROJECT_NAME"] = COMET_PROJECT
    os.environ["COMET_TAGS"]         = ",".join(tags)
    if COMET_WORKSPACE:
        os.environ["COMET_WORKSPACE"] = COMET_WORKSPACE
    comet_ml.login(api_key=COMET_API_KEY)   # auth only — project/workspace/tags set via env vars above
    REPORT_TO = "comet_ml"
    print(f"Comet  : configured  project={COMET_PROJECT}  tags={tags}")
else:
    REPORT_TO = "none"
    print("Comet  : not configured (set COMET_API_KEY to enable)")

OPIK_ENABLED = bool(OPIK_API_KEY)
if OPIK_ENABLED:
    import opik
    os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT_NAME
    opik.configure(api_key=OPIK_API_KEY, workspace=OPIK_WORKSPACE)
    print(f"Opik   : configured  project={OPIK_PROJECT_NAME}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "ଆପଣ ଜଣେ ସହାୟକ ଗଣିତ ସହକାରୀ ଅଟନ୍ତି। "
    "ତଳେ ଦିଆଯାଇଥିବା ସମସ୍ୟାକୁ ପର୍ଯ୍ୟାୟକ୍ରମେ ସମାଧାନ କରନ୍ତୁ। "
    "ଶେଷରେ, ଆପଣଙ୍କର ଚୂଡ଼ାନ୍ତ ସାଂଖ୍ୟିକ ଉତ୍ତରକୁ ଏକ ନୂଆ ଧାଡ଼ିରେ '####' ସହିତ ଆରମ୍ଭ କରି ଲେଖନ୍ତୁ।"
)

raw_ds = load_dataset(DATASET_ID, split=TRAIN_SPLIT, token=HF_TOKEN)
print(f"Loaded {len(raw_ds)} train examples")

def to_grpo_example(ex):
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nପ୍ରଶ୍ନ: {ex[QUESTION_COL]}\nଉତ୍ତର:",
        "gold":   str(ex[ANSWER_COL]),
    }

train_ds = raw_ds.map(to_grpo_example, remove_columns=raw_ds.column_names)
print(f"Prepared {len(train_ds)} GRPO prompts")


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
ODIA_TO_ARABIC = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")


def extract_numerical_answer(text: str) -> Optional[float]:
    text = text.translate(ODIA_TO_ARABIC)
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", "").rstrip("."))
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:[,\.]\d+)*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", "").rstrip("."))
        except ValueError:
            return None
    return None


def _match(pred, gold, tol=1e-3):
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def correctness_reward(completions, gold, **_kwargs):
    """+1.0 if final numerical answer matches gold, else 0.0."""
    rewards = []
    for completion, g in zip(completions, gold):
        pred_num = extract_numerical_answer(completion)
        gold_num = extract_numerical_answer(g)
        rewards.append(1.0 if _match(pred_num, gold_num) else 0.0)
    return rewards


def format_reward(completions, **_kwargs):
    """Small bonus for following the '#### <answer>' convention."""
    return [0.1 if re.search(r"####\s*[\-\d]", c.translate(ODIA_TO_ARABIC)) else 0.0
            for c in completions]


# Self-test reward functions
_sample_completions = ["step ...\n#### 42", "#### ୭", "no answer here"]
_sample_gold        = ["step ...\n#### 42", "#### 7", "#### 100"]
assert correctness_reward(_sample_completions, _sample_gold) == [1.0, 1.0, 0.0]
assert format_reward(_sample_completions) == [0.1, 0.1, 0.0]
print("Reward functions OK")


# ---------------------------------------------------------------------------
# Tokenizer + model
# ---------------------------------------------------------------------------
print(f"Loading tokenizer from {SFT_MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # required for GRPO rollout generation

print(f"Loading model from {SFT_MODEL_ID} ...")
model_kwargs = dict(token=HF_TOKEN, trust_remote_code=True, device_map="auto")

if USE_QLORA:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:
    model_kwargs["torch_dtype"] = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_ID, **model_kwargs)
model.config.use_cache = False

if USE_QLORA:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
print("Model ready.")


# ---------------------------------------------------------------------------
# GRPO trainer
# ---------------------------------------------------------------------------
run_name = "-".join(filter(None, ["sarvam1-grpo", GPU_TYPE, str(int(time.time()))]))

grpo_config = GRPOConfig(
    output_dir=str(GRPO_OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_RATIO,   # float in (0,1) is interpreted as a ratio of total steps
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    bf16=True,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION,
    beta=BETA,
    use_vllm=USE_VLLM,
    temperature=0.9,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    report_to=REPORT_TO,
    run_name=run_name,
    push_to_hub=False,
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_ds,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    peft_config=lora_config,
)

trainable, total = trainer.model.get_nb_trainable_parameters()
print(f"Train prompts    : {len(train_ds)}")
print(f"Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"Run name         : {run_name}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer.train()
print("GRPO training complete.")

adapter_dir = GRPO_OUTPUT_DIR / "final-adapter"
trainer.save_model(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print(f"Adapter saved to {adapter_dir}")

if PUSH_TO_HUB:
    # Reset to right-padding before pushing — left-padding was needed for rollouts
    # but downstream users expect right-padded tokenizers.
    tokenizer.padding_side = "right"

    print(f"Pushing adapter → {GRPO_ADAPTER_HUB_MODEL_ID}")
    create_repo(GRPO_ADAPTER_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO, exist_ok=True)
    trainer.model.push_to_hub(GRPO_ADAPTER_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    tokenizer.push_to_hub(GRPO_ADAPTER_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    print(f"Pushed adapter: https://huggingface.co/{GRPO_ADAPTER_HUB_MODEL_ID}")


# ---------------------------------------------------------------------------
# Merge LoRA → push to Hub
# ---------------------------------------------------------------------------
merged_model = None

if PUSH_TO_HUB:
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    print("Reloading SFT base in bf16 to merge GRPO adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    print(f"Pushing merged model → {GRPO_HUB_MODEL_ID}")
    create_repo(GRPO_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO, exist_ok=True)
    merged_model.push_to_hub(GRPO_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    tokenizer.push_to_hub(GRPO_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    print(f"Pushed: https://huggingface.co/{GRPO_HUB_MODEL_ID}")
else:
    print("PUSH_TO_HUB=false — adapter saved locally only.")


# ---------------------------------------------------------------------------
# Sanity-check inference
# ---------------------------------------------------------------------------
test_ds = load_dataset(DATASET_ID, split="test", token=HF_TOKEN).select(range(3))

eval_model = merged_model if merged_model is not None else trainer.model
eval_model.eval()

_track = opik.track(name="grpo_sanity_check") if OPIK_ENABLED else (lambda f: f)

@_track
def run_sample(question: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nପ୍ରଶ୍ନ: {question}\nଉତ୍ତର:"
    inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)
    with torch.no_grad():
        out = eval_model.generate(
            **inputs,
            max_new_tokens=MAX_COMPLETION,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("\n--- Sanity check ---")
for ex in test_ds:
    pred = run_sample(ex[QUESTION_COL])
    print("Q   :", ex[QUESTION_COL][:120], "...")
    print("Gold:", str(ex[ANSWER_COL])[:200], "...")
    print("Pred:", pred[:300])
    print("-" * 60)


# ---------------------------------------------------------------------------
# Finalize tracking — flush + close the Comet experiment
# ---------------------------------------------------------------------------
if REPORT_TO == "comet_ml":
    comet_ml.end()
    print("Comet  : experiment ended — all metrics & code flushed")
