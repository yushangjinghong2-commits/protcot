#!/bin/bash
# GPU4 - Llama3-8B: Group G (Llama-RAGft-nostructure-unsloth) + H (Llama-RAGft-unsloth)
# Total: 12 experiments

PIPELINE_SCRIPT="/home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh"
BASE_OUTPUT_DIR="/home/aiscuser/jxlei/protcot/results"
BATCH_SIZE=1000
LLAMA="/home/aiscuser/jxlei/models/Meta-Llama-3.1-8B-Instruct"

DS_NOSTRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json"
DS_STRUCTURE="/home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced.json"
DS_RAG_NOSTRUCTURE="/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed_no_structure.json"
DS_RAG_STRUCTURE="/home/aiscuser/jxlei/Protcot/aa+di/PFUD_replaced_with_neighbors_filtered_top1ge20_fixed.json"

LORA_G="/home/aiscuser/jxlei/protcot/saves/Llama-RAGft-nostructure-unsloth"
LORA_H="/home/aiscuser/jxlei/protcot/saves/Llama-RAGft-unsloth"

EXPERIMENTS=(
    # Group G: Llama-RAGft-nostructure-unsloth + nostructure
    "$LLAMA|$LORA_G/checkpoint-15000|$DS_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck15000-nostructure"
    "$LLAMA|$LORA_G/checkpoint-25000|$DS_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck25000-nostructure"
    "$LLAMA|$LORA_G/checkpoint-35000|$DS_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck35000-nostructure"
    # Group G: Llama-RAGft-nostructure-unsloth + rag-nostructure
    "$LLAMA|$LORA_G/checkpoint-15000|$DS_RAG_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck15000-rag-nostructure"
    "$LLAMA|$LORA_G/checkpoint-25000|$DS_RAG_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck25000-rag-nostructure"
    "$LLAMA|$LORA_G/checkpoint-35000|$DS_RAG_NOSTRUCTURE|llama3-ragft-nostructure-unsloth-ck35000-rag-nostructure"
    # Group H: Llama-RAGft-unsloth + rag-structure
    "$LLAMA|$LORA_H/checkpoint-15000|$DS_RAG_STRUCTURE|llama3-ragft-unsloth-ck15000-rag-structure"
    "$LLAMA|$LORA_H/checkpoint-25000|$DS_RAG_STRUCTURE|llama3-ragft-unsloth-ck25000-rag-structure"
    "$LLAMA|$LORA_H/checkpoint-35000|$DS_RAG_STRUCTURE|llama3-ragft-unsloth-ck35000-rag-structure"
    # Group H: Llama-RAGft-unsloth + structure
    "$LLAMA|$LORA_H/checkpoint-15000|$DS_STRUCTURE|llama3-ragft-unsloth-ck15000-structure"
    "$LLAMA|$LORA_H/checkpoint-25000|$DS_STRUCTURE|llama3-ragft-unsloth-ck25000-structure"
    "$LLAMA|$LORA_H/checkpoint-35000|$DS_STRUCTURE|llama3-ragft-unsloth-ck35000-structure"
)

TOTAL=${#EXPERIMENTS[@]}; PASS=0; FAIL=0; FAILED_NAMES=()
echo "=========================================="; echo "GPU4 - Llama3-8B Group G+H | Total: $TOTAL"; echo "=========================================="

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
