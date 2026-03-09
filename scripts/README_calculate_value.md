# calculate_value.py 使用说明

## 📋 功能概述

`calculate_value.py` 是一个用于评估蛋白质功能预测模型性能的脚本，支持计算以下指标：

- **BLEU**: 机器翻译常用指标，衡量预测文本与参考文本的 n-gram 重叠度
- **ROUGE-1/2/L**: 文本摘要常用指标，衡量召回率
- **BERTScore F1**: 基于 BERT 嵌入的语义相似度指标（使用 BioBERT）

## 🔧 环境要求

### 依赖包

```bash
pip install bert-score transformers evaluate torch tqdm
```

### 模型文件

需要预先下载 BioBERT 模型到本地：

```bash
# 脚本中硬编码的路径
/remote-home/jxlei/models/biobert-large-cased-v1.1
```

如果路径不存在，会回退到 `bert-base-cased`。

## 📥 输入格式

### 支持的文件格式

1. **JSONL 格式**（推荐）：每行一个 JSON 对象
2. **JSON 格式**：单个 JSON 数组

### 支持的字段组合

脚本会自动检测以下字段组合（按优先级）：

| Ground Truth 字段 | Prediction 字段 | 来源说明 |
|------------------|----------------|---------|
| `ground_truth` | `predicted_function` | 标准预测文件 |
| `label` | `retrieved_functions` | 检索结果文件 |
| `label` | `function_predict` | RAG 预测文件 |
| `function` | `predicted_function` | 通用格式 |
| `reference` | `prediction` | vLLM 评估输出 ✨ |

### 示例输入文件

#### 格式 1: vLLM 评估输出 (JSONL)

```jsonl
{"accesion": "P12345", "split": "test", "instruction": "...", "input": "...", "prompt": "...", "prediction": "This protein functions as...", "reference": "The protein is involved in..."}
{"accesion": "P67890", "split": "test", "instruction": "...", "input": "...", "prompt": "...", "prediction": "This enzyme catalyzes...", "reference": "This enzyme is responsible for..."}
```

#### 格式 2: 标准预测文件 (JSONL)

```jsonl
{"accesion": "P12345", "ground_truth": "The protein is involved in...", "predicted_function": "This protein functions as..."}
{"accesion": "P67890", "ground_truth": "This enzyme is responsible for...", "predicted_function": "This enzyme catalyzes..."}
```

## 🚀 使用方法

### 基本用法

```bash
# 使用命令行参数指定输入文件
python calculate_value.py <input_file.jsonl>

# 示例
python calculate_value.py /path/to/predictions.jsonl
```

### 默认行为

如果不提供参数，使用默认路径：

```bash
python calculate_value.py
# 默认读取: /remote-home/jxlei/protcot/prot_function_predictions.jsonl
```

## 📊 输出说明

### 控制台输出

```
Loading predictions from /path/to/predictions.jsonl...
Detected JSONL format...
Detected fields: ground_truth='reference', prediction='prediction'
Filtering entries > 511 tokens...
Filtering: 100%|████████| 1000/1000 [00:05<00:00, 200.00it/s]
Loaded 950 records (skipped 50 due to length > 511).
Results will be saved to: /remote-home/jxlei/protcot/f1_result_predictions.txt

Computing BLEU and ROUGE...
BLEU: 0.3456
ROUGE-1: 0.4567
ROUGE-2: 0.2345
ROUGE-L: 0.3890

Computing BERTScore F1...
Using device: cuda
Model loaded. Starting inference...
Computing F1: 100%|████████| 15/15 [00:30<00:00,  2.00s/it]
BERTScore F1 (mean): 0.7823
```

### 输出文件

自动生成结果文件：`f1_result_<input_basename>.txt`

**示例**：输入 `predictions.jsonl` → 输出 `f1_result_predictions.txt`

**文件内容**：

```
Evaluation results for: /path/to/predictions.jsonl
Ground truth field: reference
Prediction field: prediction
Total records: 950
Skipped records: 50
============================================================

Computing BLEU and ROUGE...
BLEU: 0.3456
ROUGE-1: 0.4567
ROUGE-2: 0.2345
ROUGE-L: 0.3890

Computing BERTScore F1...
BERTScore F1 (mean): 0.7823
```

## ⚙️ 重要特性

### 1. 自动长度过滤

- **限制**：BERT 模型最大支持 512 tokens
- **行为**：自动跳过超过 511 tokens 的样本（预留 [CLS]/[SEP] 位置）
- **统计**：会报告跳过的样本数量

### 2. JSON 字段提取

对于 `function_predict` 字段，支持从 JSON 格式中提取 `description`：

```python
# 输入
{"function_predict": '{"description": "This protein...", "other": "..."}'}

# 自动提取为
"This protein..."
```

### 3. 批处理

- **BLEU/ROUGE**：一次性处理所有样本（CPU）
- **BERTScore**：批次大小 64（GPU），带进度条

### 4. 错误处理

- 空字符串自动跳过
- 编码错误自动跳过并记录
- 指标计算失败会打印详细错误信息

## 🔍 常见问题

### Q1: 提示 "Could not detect field names"

**原因**：输入文件的字段名不在支持列表中

**解决**：
1. 检查第一条数据的字段名
2. 修改 `detect_field_names()` 函数添加新的字段组合

### Q2: BERTScore 计算很慢

**原因**：BioBERT-large 模型较大，推理速度慢

**解决**：
1. 确保使用 GPU（自动检测）
2. 减少批次大小（修改第 216 行 `batch_size = 64`）
3. 使用更小的模型（修改第 203 行 `model_path`）

### Q3: 显存不足 (OOM)

**解决**：
1. 减小批次大小：`batch_size = 32` 或 `16`
2. 使用 CPU：设置 `device = "cpu"`（第 200 行）

### Q4: 跳过了很多样本

**原因**：样本超过 511 tokens

**解决**：
1. 检查数据质量（是否有异常长的文本）
2. 如果合理，可以修改限制（不推荐，会导致截断）

## 📝 完整工作流示例

### 步骤 1: 使用 vLLM 生成预测

```bash
python scripts/evaluate_pfud_test_vllm.py \
    --model-path /home/aiscuser/jxlei/models/Meta-Llama-3.1-8B-Instruct \
    --lora-path /home/aiscuser/jxlei/LlamaFactory/saves/Llama/lora/sft \
    --dataset-path /home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json \
    --output-path results/llama_predictions.jsonl \
    --split test \
    --batch-size 10 \
    --max-new-tokens 512
```

### 步骤 2: 计算评估指标

```bash
python scripts/calculate_value.py results/llama_predictions.jsonl
```

### 步骤 3: 查看结果

```bash
cat /remote-home/jxlei/protcot/f1_result_llama_predictions.txt
```

## 🛠️ 自定义修改

### 修改模型路径

```python
# 第 94 行和第 203 行
model_path = "/your/path/to/biobert-model"
```

### 修改批次大小

```python
# 第 216 行
batch_size = 32  # 默认 64
```

### 修改长度限制

```python
# 第 129 行
if len_gt > 1023 or len_pred > 1023:  # 默认 511
```

### 添加新的字段组合

```python
# 第 35-55 行，在 detect_field_names() 中添加
elif 'your_gt_field' in entry and 'your_pred_field' in entry:
    return 'your_gt_field', 'your_pred_field'
```

## 📌 注意事项

1. **路径硬编码**：脚本中有多处硬编码路径，使用前需要检查
2. **模型依赖**：需要预先下载 BioBERT 模型
3. **显存需求**：BERTScore 计算需要 GPU（推荐至少 8GB 显存）
4. **时间估算**：1000 样本约需 1-2 分钟（GPU）
5. **输出覆盖**：结果文件会自动覆盖同名文件

## 🔗 相关文件

- **输入生成**：`evaluate_pfud_test_vllm.py`
- **数据集**：`PFUD_replaced_no_structure.json`
- **模型**：BioBERT (`biobert-large-cased-v1.1`)

## 📚 参考资源

- [BERTScore 论文](https://arxiv.org/abs/1904.09675)
- [BLEU 论文](https://aclanthology.org/P02-1040/)
- [ROUGE 论文](https://aclanthology.org/W04-1013/)
- [BioBERT 论文](https://arxiv.org/abs/1901.08746)
