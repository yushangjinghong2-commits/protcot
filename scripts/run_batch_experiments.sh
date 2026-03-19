#!/bin/bash
# Batch Experiment Runner - 串行执行多个评估实验
# 用法: bash run_batch_experiments.sh
# 修改下方 EXPERIMENTS 数组来配置实验组合

set -e

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000

# ============================================================
# 在这里定义实验组合，每行格式：
#   "MODEL_PATH|LORA_PATH|DATASET_PATH|OUTPUT_NAME"
# LORA_PATH 留空则不使用 LoRA（填 ""）
# ============================================================
EXPERIMENTS=(
    # 示例（取消注释并修改）：
    # "/home/aiscuser/jxlei/models/Meta-Llama-3.1-8B-Instruct|/home/aiscuser/jxlei/protcot/saves/Llama-RAGft-nostructure-unsloth/checkpoint-25000|/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed_no_structure.json|llama3-lora-ragft-nostructure-ck25000"
)

# ============================================================

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "Error: EXPERIMENTS 数组为空，请先填写实验组合"
    exit 1
fi

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0
FAILED_NAMES=()

echo "=========================================="
echo "Batch Experiment Runner"
echo "Total experiments: $TOTAL"
echo "=========================================="

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r MODEL_PATH LORA_PATH DATASET_PATH OUTPUT_NAME <<< "${EXPERIMENTS[$i]}"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$OUTPUT_NAME"
    IDX=$((i + 1))

    echo ""
    echo "[$IDX/$TOTAL] Starting: $OUTPUT_NAME"
    echo "  Model:   $MODEL_PATH"
    echo "  LoRA:    ${LORA_PATH:-<none>}"
    echo "  Dataset: $DATASET_PATH"
    echo "  Output:  $OUTPUT_DIR"
    echo "------------------------------------------"

    LORA_ARG=""
    if [ -n "$LORA_PATH" ]; then
        LORA_ARG="--lora-path $LORA_PATH"
    fi

    if bash "$PIPELINE_SCRIPT" \
        --model-path "$MODEL_PATH" \
        $LORA_ARG \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE"; then
        echo "[$IDX/$TOTAL] PASSED: $OUTPUT_NAME"
        PASS=$((PASS + 1))
    else
        echo "[$IDX/$TOTAL] FAILED: $OUTPUT_NAME"
        FAIL=$((FAIL + 1))
        FAILED_NAMES+=("$OUTPUT_NAME")
    fi
done

echo ""
echo "=========================================="
echo "Batch completed: $PASS passed, $FAIL failed"
if [ ${#FAILED_NAMES[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  - $name"
    done
fi
echo "=========================================="
