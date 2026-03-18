#!/usr/bin/env python3
"""
Unsloth训练脚本 - 支持Alpaca格式JSON数据集
"""

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

# ============ 配置参数 ============
MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_PATH = "/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed.json"
OUTPUT_DIR = "/home/aiscuser/jxlei/protcot/saves/Qwen-RAGft-unsloth"
MAX_SEQ_LENGTH = 1024

# LoRA参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# 训练参数
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000

print("=" * 70)
print("开始训练配置")
print("=" * 70)

# ============ 加载数据集 ============
print(f"\n加载数据集: {DATASET_PATH}")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
print(f"数据集大小: {len(dataset)} 样本")

# ============ 加载模型 ============
print(f"\n加载模型: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print("模型加载完成")

# ============ 配置LoRA ============
print(f"\n配置LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ============ 数据格式化 ============
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {'text': texts}

print("\n格式化数据集...")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ============ 训练配置 ============
print("\n配置训练参数:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Output dir: {OUTPUT_DIR}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to='wandb',
        run_name='qwen-ragft-with-structure',
        save_strategy='steps',
        save_steps=5000
    ),
)

# ============ 开始训练 ============
print("\n" + "=" * 70)
print("开始训练")
print("=" * 70)
trainer.train()

# ============ 保存模型 ============
print("\n保存模型...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"模型已保存到: {OUTPUT_DIR}/final")

print("\n" + "=" * 70)
print("训练完成！")
print("=" * 70)
