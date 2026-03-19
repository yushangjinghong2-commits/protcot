#!/bin/bash
# GPU1 - Qwen3-8B: Group A (LlamaFactory/Llama sft) + Group B (LlamaFactory/qwen3-8b-2 sft)
# Total: 6 experiments

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000
QWEN="Qwen/Qwen3-8B"

DS_NOSTRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json"
DS_STRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced.json"

LORA_A="/home/aiscuser/jxlei/LlamaFactory/saves/Llama/lora/sft"
LORA_B="/home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b-2/lora/sft"

EXPERIMENTS=(
    # Group A: Llama/lora/sft + nostructure
    "$QWEN|$LORA_A/checkpoint-15000|$DS_NOSTRUCTURE|qwen3-llamaft-ck15000-nostructure"
    "$QWEN|$LORA_A/checkpoint-25000|$DS_NOSTRUCTURE|qwen3-llamaft-ck25000-nostructure"
    "$QWEN|$LORA_A/checkpoint-35000|$DS_NOSTRUCTURE|qwen3-llamaft-ck35000-nostructure"
    # Group B: qwen3-8b-2/lora/sft + structure
    "$QWEN|$LORA_B/checkpoint-15000|$DS_STRUCTURE|qwen3-qwen3ft2-ck15000-structure"
    "$QWEN|$LORA_B/checkpoint-25000|$DS_STRUCTURE|qwen3-qwen3ft2-ck25000-structure"
    "$QWEN|$LORA_B/checkpoint-35000|$DS_STRUCTURE|qwen3-qwen3ft2-ck35000-structure"
)

TOTAL=${#EXPERIMENTS[@]}; PASS=0; FAIL=0; FAILED_NAMES=()
echo "=========================================="; echo "GPU1 - Qwen3-8B Group A+B | Total: $TOTAL"; echo "=========================================="

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
