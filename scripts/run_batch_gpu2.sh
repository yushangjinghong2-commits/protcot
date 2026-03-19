#!/bin/bash
# GPU2 - Qwen3-8B: Group C (Qwen-RAGft-unsloth) x 2 datasets
# Total: 6 experiments

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000
QWEN="Qwen/Qwen3-8B"

DS_RAG_STRUCTURE="/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed.json"
DS_STRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced.json"

LORA_C="/home/aiscuser/jxlei/protcot/saves/Qwen-RAGft-unsloth"

EXPERIMENTS=(
    "$QWEN|$LORA_C/checkpoint-15000|$DS_RAG_STRUCTURE|qwen3-ragft-unsloth-ck15000-rag-structure"
    "$QWEN|$LORA_C/checkpoint-25000|$DS_RAG_STRUCTURE|qwen3-ragft-unsloth-ck25000-rag-structure"
    "$QWEN|$LORA_C/checkpoint-35000|$DS_RAG_STRUCTURE|qwen3-ragft-unsloth-ck35000-rag-structure"
    "$QWEN|$LORA_C/checkpoint-15000|$DS_STRUCTURE|qwen3-ragft-unsloth-ck15000-structure"
    "$QWEN|$LORA_C/checkpoint-25000|$DS_STRUCTURE|qwen3-ragft-unsloth-ck25000-structure"
    "$QWEN|$LORA_C/checkpoint-35000|$DS_STRUCTURE|qwen3-ragft-unsloth-ck35000-structure"
)

TOTAL=${#EXPERIMENTS[@]}; PASS=0; FAIL=0; FAILED_NAMES=()
echo "=========================================="; echo "GPU2 - Qwen3-8B Group C | Total: $TOTAL"; echo "=========================================="

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r MODEL_PATH LORA_PATH DATASET_PATH OUTPUT_NAME <<< "${EXPERIMENTS[$i]}"
    IDX=$((i + 1))
    echo ""; echo "[$IDX/$TOTAL] $OUTPUT_NAME"
    echo "  LoRA:    $LORA_PATH"; echo "  Dataset: $DATASET_PATH"
    if bash "$PIPELINE_SCRIPT" --model-path "$MODEL_PATH" --lora-path "$LORA_PATH" \
        --dataset-path "$DATASET_PATH" --output-dir "$BASE_OUTPUT_DIR/$OUTPUT_NAME" \
        --batch-size "$BATCH_SIZE"; then
        echo "[$IDX/$TOTAL] PASSED: $OUTPUT_NAME"; PASS=$((PASS + 1))
    else
        echo "[$IDX/$TOTAL] FAILED: $OUTPUT_NAME"; FAIL=$((FAIL + 1)); FAILED_NAMES+=("$OUTPUT_NAME")
    fi
done

echo ""; echo "=========================================="; echo "Done: $PASS passed, $FAIL failed"
[ ${#FAILED_NAMES[@]} -gt 0 ] && for n in "${FAILED_NAMES[@]}"; do echo "  FAILED: $n"; done
echo "=========================================="
