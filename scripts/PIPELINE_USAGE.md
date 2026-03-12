# 蛋白质功能预测评估流程使用说明

## 概述

`run_evaluation_pipeline.sh` 是一个自动化脚本，整合了模型推理和评估指标计算两个步骤：

1. **步骤1**: 使用 `evaluate_pfud.py` 运行模型推理，生成预测结果
2. **步骤2**: 使用 `calculate_value.py` 计算评估指标（BLEU、ROUGE、BERTScore）

## 前置要求

```bash
# 给脚本添加执行权限
chmod +x /home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh
```

## 基本用法

```bash
./run_evaluation_pipeline.sh \
  --model-path <模型路径> \
  --dataset-path <数据集路径> \
  --output-dir <输出目录>
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | 模型路径或HuggingFace模型ID | `Qwen/Qwen3-8B` |
| `--dataset-path` | 数据集JSON文件路径 | `/path/to/dataset.json` |
| `--output-dir` | 结果保存目录 | `/path/to/output` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lora-path` | LoRA适配器路径 | 无 |
| `--split` | 数据集划分 | `test` |
| `--batch-size` | 批次大小 | `10` |
| `--max-new-tokens` | 最大生成token数 | `5120` |
| `--temperature` | 采样温度（0为贪婪解码） | `0.0` |
| `--top-p` | Top-p采样 | `1.0` |
| `--tensor-parallel-size` | GPU并行数量 | `1` |
| `--gpu-memory-utilization` | GPU显存使用率 | `0.9` |
| `--enable-thinking` | 启用Qwen思考模式 | 关闭 |

## 使用示例

### 示例1: 基础模型评估

```bash
./run_evaluation_pipeline.sh \
  --model-path Qwen/Qwen3-8B \
  --dataset-path /home/aiscuser/jxlei/Protcot/aa/PFUD_replaced_with_aa_neighbors.json \
  --output-dir /home/aiscuser/jxlei/protcot/results/qwen3-base
```

### 示例2: LoRA微调模型评估

```bash
./run_evaluation_pipeline.sh \
  --model-path Qwen/Qwen3-8B \
  --lora-path /home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b/lora/sft/checkpoint-10000 \
  --dataset-path /home/aiscuser/jxlei/Protcot/aa/PFUD_replaced_with_aa_neighbors.json \
  --output-dir /home/aiscuser/jxlei/protcot/results/qwen3-lora \
  --batch-size 16
```

### 示例3: 多GPU并行评估

```bash
./run_evaluation_pipeline.sh \
  --model-path meta-llama/Llama-3.1-8B \
  --lora-path /path/to/llama-lora \
  --dataset-path /home/aiscuser/jxlei/Protcot/aa/evaluation_dataset_test.json \
  --output-dir /home/aiscuser/jxlei/protcot/results/llama3-test \
  --tensor-parallel-size 2 \
  --batch-size 20
```

### 示例4: 启用Qwen思考模式

```bash
./run_evaluation_pipeline.sh \
  --model-path Qwen/Qwen3-8B \
  --dataset-path /home/aiscuser/jxlei/Protcot/aa/PFUD_replaced_with_aa_neighbors.json \
  --output-dir /home/aiscuser/jxlei/protcot/results/qwen3-thinking \
  --enable-thinking
```

## 输出文件

运行完成后，输出目录包含：

```
output-dir/
├── predictions.jsonl           # 模型预测结果（JSONL格式）
└── evaluation_metrics.txt      # 评估指标结果
```

### predictions.jsonl 格式

每行一个JSON对象：
```json
{
  "accesion": "afdb_accesion_A3CQH4",
  "split": "test",
  "instruction": "...",
  "input": "...",
  "prompt": "...",
  "prediction": "模型预测的功能描述",
  "reference": "标准答案"
}
```

### evaluation_metrics.txt 内容

```
Evaluation results for: /path/to/predictions.jsonl
Ground truth field: reference
Prediction field: prediction
Total records: 1059
Skipped records: 4596
============================================================

BLEU: 0.1234
ROUGE-1: 0.4567
ROUGE-2: 0.2345
ROUGE-L: 0.3456

BERTScore F1 (mean): 0.7890
```

## 常见问题

### 1. 显存不足

降低batch size或GPU显存使用率：
```bash
--batch-size 4 --gpu-memory-utilization 0.8
```

### 2. 数据集格式不兼容

确保数据集包含以下字段之一的组合：
- `reference` + `prediction`
- `true_function` + `predicted_function`
- `ground_truth` + `predicted_function`
- `label` + `function_predict`

### 3. 查看帮助信息

```bash
./run_evaluation_pipeline.sh --help
```

## 批量评估脚本示例

创建批量评估脚本 `batch_eval.sh`：

```bash
#!/bin/bash

DATASETS=(
    "/home/aiscuser/jxlei/Protcot/aa/evaluation_dataset_test.json"
    "/home/aiscuser/jxlei/Protcot/aa+di/evaluation_dataset_test.json"
    "/home/aiscuser/jxlei/Protcot/di/evaluation_dataset_test.json"
)

MODELS=(
    "Qwen/Qwen3-8B:/path/to/qwen-lora"
    "meta-llama/Llama-3.1-8B:/path/to/llama-lora"
)

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(basename $(dirname "$dataset"))

    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r model_path lora_path <<< "$model_info"
        model_name=$(basename "$model_path")

        output_dir="/home/aiscuser/jxlei/protcot/results/${model_name}_${dataset_name}"

        echo "Evaluating $model_name on $dataset_name..."

        ./run_evaluation_pipeline.sh \
            --model-path "$model_path" \
            --lora-path "$lora_path" \
            --dataset-path "$dataset" \
            --output-dir "$output_dir" \
            --batch-size 10
    done
done
```

## 注意事项

1. **Token长度限制**: BERTScore计算时会过滤超过511 tokens的样本
2. **依赖检查**: 确保已安装 `vllm`, `bert_score`, `evaluate`, `rouge_score` 等依赖
3. **路径问题**: 脚本中硬编码了 `/home/aiscuser/jxlei/protcot/scripts/` 路径，如需修改请编辑脚本
