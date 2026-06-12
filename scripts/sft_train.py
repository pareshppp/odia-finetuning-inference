"""
Sarvam-1 SFT on Odia GSM8K
----------------------------
Supervised fine-tuning of ai4bharat/sarvam-1 on the train split of
tripathysagar/odia-gsm8k using TRL SFTTrainer + QLoRA.

Usage (RunPod):
    screen -S sft
    cd /workspace && python sft_train.py
    # Ctrl-A D to detach; screen -r sft to reattach

All config via environment variables — see .env.example.
"""

import gc
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login as hf_login, create_repo

print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN         = os.getenv("HF_TOKEN")
HF_USERNAME      = os.getenv("HF_USERNAME", "")
BASE_MODEL_ID    = os.getenv("MODEL_ID",    "ai4bharat/sarvam-1")
DATASET_ID       = os.getenv("DATASET_ID",  "tripathysagar/odia-gsm8k")
TRAIN_SPLIT      = os.getenv("TRAIN_SPLIT", "train")
QUESTION_COL     = os.getenv("QUESTION_COL", "question")
ANSWER_COL       = os.getenv("ANSWER_COL",   "answer")

SFT_OUTPUT_DIR   = Path(os.getenv("SFT_OUTPUT_DIR",  "/workspace/sarvam1-odia-gsm8k-sft"))
SFT_HUB_MODEL_ID = os.getenv("SFT_HUB_MODEL_ID",     f"{HF_USERNAME}/sarvam-1-odia-gsm8k-sft" if HF_USERNAME else "")
PUSH_TO_HUB      = os.getenv("PUSH_TO_HUB",  "true").lower() == "true"
PRIVATE_REPO     = os.getenv("PRIVATE_REPO", "false").lower() == "true"

USE_QLORA        = os.getenv("USE_QLORA",    "true").lower() == "true"
NUM_EPOCHS       = float(os.getenv("NUM_EPOCHS",    "3"))
LEARNING_RATE    = float(os.getenv("LEARNING_RATE", "2e-4"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",      "8"))
GRAD_ACCUM       = int(os.getenv("GRAD_ACCUM",      "2"))
MAX_SEQ_LEN      = int(os.getenv("MAX_SEQ_LEN",     "2048"))
WARMUP_RATIO     = float(os.getenv("WARMUP_RATIO",  "0.03"))
LOGGING_STEPS    = int(os.getenv("LOGGING_STEPS",   "10"))
SAVE_STEPS       = int(os.getenv("SAVE_STEPS",      "200"))
LORA_R           = int(os.getenv("LORA_R",          "16"))
LORA_ALPHA       = int(os.getenv("LORA_ALPHA",      "32"))

COMET_API_KEY    = os.getenv("COMET_API_KEY")
COMET_WORKSPACE  = os.getenv("COMET_WORKSPACE")
COMET_PROJECT    = os.getenv("COMET_PROJECT_NAME", "sarvam1-odia-gsm8k")

# Fail fast on missing push target — don't waste hours of training
if PUSH_TO_HUB:
    assert SFT_HUB_MODEL_ID, (
        "PUSH_TO_HUB=true but neither SFT_HUB_MODEL_ID nor HF_USERNAME is set. "
        "Set one before training, or set PUSH_TO_HUB=false."
    )

SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Base model : {BASE_MODEL_ID}")
print(f"Dataset    : {DATASET_ID} ({TRAIN_SPLIT})")
print(f"Output dir : {SFT_OUTPUT_DIR}")
print(f"Hub repo   : {SFT_HUB_MODEL_ID or '(unset)'}  push={PUSH_TO_HUB}")
print(f"QLoRA      : {USE_QLORA}")
print(f"Epochs={NUM_EPOCHS}  lr={LEARNING_RATE}  bs={BATCH_SIZE}x{GRAD_ACCUM}  max_seq={MAX_SEQ_LEN}")


# ---------------------------------------------------------------------------
# Auth + tracking
# ---------------------------------------------------------------------------
if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("HF Hub : logged in")
else:
    print("HF Hub : HF_TOKEN not set — push will fail")

if COMET_API_KEY:
    import comet_ml
    comet_ml.login(api_key=COMET_API_KEY, workspace=COMET_WORKSPACE, project_name=COMET_PROJECT)
    os.environ["COMET_MODE"] = "ONLINE"
    os.environ["COMET_PROJECT_NAME"] = COMET_PROJECT
    if COMET_WORKSPACE:
        os.environ["COMET_WORKSPACE"] = COMET_WORKSPACE
    REPORT_TO = "comet_ml"
    print("Comet  : configured")
else:
    REPORT_TO = "none"
    print("Comet  : not configured (set COMET_API_KEY to enable)")

OPIK_ENABLED = bool(os.getenv("OPIK_API_KEY"))
if OPIK_ENABLED:
    import opik
    opik.configure(api_key=os.getenv("OPIK_API_KEY"), workspace=os.getenv("OPIK_WORKSPACE"))
    print("Opik   : configured")
else:
    print("Opik   : not configured (optional)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "ଆପଣ ଜଣେ ସହାୟକ ଗଣିତ ସହକାରୀ ଅଟନ୍ତି। "
    "ତଳେ ଦିଆଯାଇଥିବା ସମସ୍ୟାକୁ ପର୍ଯ୍ୟାୟକ୍ରମେ ସମାଧାନ କରନ୍ତୁ। "
    "ଶେଷରେ, ଆପଣଙ୍କର ଚୂଡ଼ାନ୍ତ ସାଂଖ୍ୟିକ ଉତ୍ତରକୁ ଏକ ନୂଆ ଧାଡ଼ିରେ '####' ସହିତ ଆରମ୍ଭ କରି ଲେଖନ୍ତୁ।"
)

ds = load_dataset(DATASET_ID, split=TRAIN_SPLIT, token=HF_TOKEN)
print(f"Loaded {len(ds)} training examples")


# ---------------------------------------------------------------------------
# Tokenizer + model
# ---------------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
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

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
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
# Format dataset
# ---------------------------------------------------------------------------
EOS = tokenizer.eos_token

def format_example(example):
    q = example[QUESTION_COL]
    a = example[ANSWER_COL]
    return {"text": f"{SYSTEM_PROMPT}\n\nପ୍ରଶ୍ନ: {q}\nଉତ୍ତର: {a}{EOS}"}

train_ds = ds.map(format_example, remove_columns=ds.column_names)
print(f"Formatted {len(train_ds)} examples")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
run_name = f"sarvam1-sft-{int(time.time())}"

sft_config = SFTConfig(
    output_dir=str(SFT_OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    bf16=True,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataset_text_field="text",
    report_to=REPORT_TO,
    run_name=run_name,
    push_to_hub=False,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainable, total = trainer.model.get_nb_trainable_parameters()
print(f"Train examples   : {len(train_ds)}")
print(f"Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"Run name         : {run_name}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer.train()
print("Training complete.")

adapter_dir = SFT_OUTPUT_DIR / "final-adapter"
trainer.save_model(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print(f"Adapter saved to {adapter_dir}")


# ---------------------------------------------------------------------------
# Merge LoRA → push to Hub
# ---------------------------------------------------------------------------
merged_model = None

if PUSH_TO_HUB:
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    print("Reloading base model in bf16 to merge adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    print(f"Pushing merged model → {SFT_HUB_MODEL_ID}")
    create_repo(SFT_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO, exist_ok=True)
    merged_model.push_to_hub(SFT_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    tokenizer.push_to_hub(SFT_HUB_MODEL_ID, token=HF_TOKEN, private=PRIVATE_REPO)
    print(f"Pushed: https://huggingface.co/{SFT_HUB_MODEL_ID}")
else:
    print("PUSH_TO_HUB=false — adapter saved locally only.")


# ---------------------------------------------------------------------------
# Sanity-check inference
# ---------------------------------------------------------------------------
test_ds = load_dataset(DATASET_ID, split="test", token=HF_TOKEN).select(range(3))

eval_model = merged_model if merged_model is not None else trainer.model
eval_model.eval()

_track = opik.track if OPIK_ENABLED else (lambda f: f)

@_track
def run_sample(question: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nପ୍ରଶ୍ନ: {question}\nଉତ୍ତର:"
    inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)
    with torch.no_grad():
        out = eval_model.generate(
            **inputs,
            max_new_tokens=256,
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
