#!/bin/bash

# Protein Function Prediction Evaluation Pipeline
# This script runs model inference and then calculates evaluation metrics

set -e  # Exit on error

# Default values
MODEL_PATH=""
LORA_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""
SPLIT="test"
BATCH_SIZE=10
MAX_NEW_TOKENS=5120
TEMPERATURE=0.0
TOP_P=1.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
ENABLE_THINKING=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --enable-thinking)
            ENABLE_THINKING="--enable-thinking"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --model-path MODEL --dataset-path DATASET --output-dir OUTPUT [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --model-path PATH           Model path or HuggingFace model ID"
            echo "  --dataset-path PATH         Path to dataset JSON file"
            echo "  --output-dir PATH           Directory to save results"
            echo ""
            echo "Optional arguments:"
            echo "  --lora-path PATH            Path to LoRA adapter"
            echo "  --split SPLIT               Dataset split (default: test)"
            echo "  --batch-size N              Batch size (default: 10)"
            echo "  --max-new-tokens N          Max new tokens (default: 5120)"
            echo "  --temperature T             Sampling temperature (default: 0.0)"
            echo "  --top-p P                   Top-p sampling (default: 1.0)"
            echo "  --tensor-parallel-size N    Number of GPUs (default: 1)"
            echo "  --gpu-memory-utilization F  GPU memory fraction (default: 0.9)"
            echo "  --enable-thinking           Enable thinking mode for Qwen models"
            echo ""
            echo "Example:"
            echo "  $0 --model-path Qwen/Qwen3-8B \\"
            echo "     --lora-path /path/to/lora \\"
            echo "     --dataset-path /path/to/dataset.json \\"
            echo "     --output-dir /path/to/output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$DATASET_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Use --help for usage information"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define output files
PREDICTIONS_FILE="$OUTPUT_DIR/predictions.jsonl"
METRICS_FILE="$OUTPUT_DIR/evaluation_metrics.txt"

echo "=========================================="
echo "Protein Function Prediction Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo "=========================================="

# Step 1: Run model inference
echo ""
echo "[Step 1/2] Running model inference..."
echo ""

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora-path $LORA_PATH"
    echo "Using LoRA adapter: $LORA_PATH"
fi

python /home/aiscuser/jxlei/protcot/scripts/evaluate_pfud.py \
    --model-path "$MODEL_PATH" \
    $LORA_ARG \
    --dataset-path "$DATASET_PATH" \
    --output-path "$PREDICTIONS_FILE" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    $ENABLE_THINKING

if [ $? -ne 0 ]; then
    echo "Error: Model inference failed"
    exit 1
fi

echo ""
echo "✓ Predictions saved to: $PREDICTIONS_FILE"

# Step 2: Calculate evaluation metrics
echo ""
echo "[Step 2/2] Calculating evaluation metrics..."
echo ""

python /home/aiscuser/jxlei/protcot/scripts/calculate_value.py "$PREDICTIONS_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Metric calculation failed"
    exit 1
fi

# Move the generated metrics file to output directory
GENERATED_METRICS=$(dirname "$PREDICTIONS_FILE")/f1_result_$(basename "$PREDICTIONS_FILE" .jsonl).txt
if [ -f "$GENERATED_METRICS" ]; then
    mv "$GENERATED_METRICS" "$METRICS_FILE"
    echo ""
    echo "✓ Metrics saved to: $METRICS_FILE"
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Results:"
echo "  - Predictions: $PREDICTIONS_FILE"
echo "  - Metrics: $METRICS_FILE"
echo "=========================================="
