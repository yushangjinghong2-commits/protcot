#!/bin/bash
# Llama3-8B + saves/Llama-2/lora/sft + no_structure

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000
LLAMA="/home/aiscuser/jxlei/models/Meta-Llama-3.1-8B-Instruct"
DS_NOSTRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json"
LORA="/home/aiscuser/jxlei/LlamaFactory/saves/Llama-2/lora/sft"

EXPERIMENTS=(
    "$LLAMA|$LORA/checkpoint-15000|$DS_NOSTRUCTURE|llama3-llamaft2-ck15000-nostructure"
    "$LLAMA|$LORA/checkpoint-25000|$DS_NOSTRUCTURE|llama3-llamaft2-ck25000-nostructure"
    "$LLAMA|$LORA/checkpoint-35000|$DS_NOSTRUCTURE|llama3-llamaft2-ck35000-nostructure"
)

TOTAL=${#EXPERIMENTS[@]}; PASS=0; FAIL=0; FAILED_NAMES=()
echo "=========================================="; echo "Llama3-8B llamaft2 nostructure | Total: $TOTAL"; echo "=========================================="

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
