#!/bin/bash
# GPU3 - Llama3-8B: Group D (qwen3-8b sft) + E (Llama-2 sft) + F (Llama-RAGft)
# Total: 12 experiments

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000
LLAMA="/home/aiscuser/jxlei/models/Meta-Llama-3.1-8B-Instruct"

DS_NOSTRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json"
DS_STRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced.json"
DS_RAG_NOSTRUCTURE="/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed_no_structure.json"

LORA_D="/home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b/lora/sft"
LORA_E="/home/aiscuser/jxlei/LlamaFactory/saves/Llama-2/lora/sft"
LORA_F="/home/aiscuser/jxlei/LlamaFactory/saves/Llama-RAGft/lora/sft"

EXPERIMENTS=(
    # Group D: qwen3-8b/lora/sft + nostructure
    "$LLAMA|$LORA_D/checkpoint-15000|$DS_NOSTRUCTURE|llama3-qwen3ft-ck15000-nostructure"
    "$LLAMA|$LORA_D/checkpoint-25000|$DS_NOSTRUCTURE|llama3-qwen3ft-ck25000-nostructure"
    "$LLAMA|$LORA_D/checkpoint-35000|$DS_NOSTRUCTURE|llama3-qwen3ft-ck35000-nostructure"
    # Group E: Llama-2/lora/sft + structure
    "$LLAMA|$LORA_E/checkpoint-15000|$DS_STRUCTURE|llama3-llamaft2-ck15000-structure"
    "$LLAMA|$LORA_E/checkpoint-25000|$DS_STRUCTURE|llama3-llamaft2-ck25000-structure"
    "$LLAMA|$LORA_E/checkpoint-35000|$DS_STRUCTURE|llama3-llamaft2-ck35000-structure"
    # Group F: Llama-RAGft + nostructure
    "$LLAMA|$LORA_F/checkpoint-9000|$DS_NOSTRUCTURE|llama3-ragft-ck9000-nostructure"
    "$LLAMA|$LORA_F/checkpoint-18000|$DS_NOSTRUCTURE|llama3-ragft-ck18000-nostructure"
    "$LLAMA|$LORA_F/checkpoint-27000|$DS_NOSTRUCTURE|llama3-ragft-ck27000-nostructure"
    # Group F: Llama-RAGft + rag-nostructure
    "$LLAMA|$LORA_F/checkpoint-9000|$DS_RAG_NOSTRUCTURE|llama3-ragft-ck9000-rag-nostructure"
    "$LLAMA|$LORA_F/checkpoint-18000|$DS_RAG_NOSTRUCTURE|llama3-ragft-ck18000-rag-nostructure"
    "$LLAMA|$LORA_F/checkpoint-27000|$DS_RAG_NOSTRUCTURE|llama3-ragft-ck27000-rag-nostructure"
)

TOTAL=${#EXPERIMENTS[@]}; PASS=0; FAIL=0; FAILED_NAMES=()
echo "=========================================="; echo "GPU3 - Llama3-8B Group D+E+F | Total: $TOTAL"; echo "=========================================="

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
