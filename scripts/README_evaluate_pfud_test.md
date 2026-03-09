# `evaluate_pfud_test.py` 使用说明
python protcot/scripts/evaluate_pfud.py --model-path  Qwen/Qwen3-8B  --lora-path /home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b/lora/sft/checkpoint-10000  --dataset-path /home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json --output-path /home/aiscuser/jxlei/protcot/predictions/qwen-ft-nostructure --split test --batch-size 10

这个脚本用于评测本地 Hugging Face 因果语言模型在 `PFUD_replaced_with_neighbors_filtered_top1ge20.json` 测试集上的表现。

支持两类模型形态：

- 直接加载基础模型，例如 `Qwen3`、`Llama 3`
- 加载基础模型后，再挂载通过 `LLaMA-Factory` 微调导出的 LoRA 权重

默认评测目标是：

- 数据集：`C:\Users\12240\Desktop\ssh\protdata\PFUD_replaced_with_neighbors_filtered_top1ge20.json`
- 分片：`test`

脚本位置：

- [evaluate_pfud_test.py](C:\Users\12240\Desktop\ssh\protdata\protcot\scripts\evaluate_pfud_test.py)

## 1. 环境依赖

至少需要：

- Python 3.10+
- `torch`
- `transformers`
- `sentencepiece`（部分模型 tokenizer 需要）
- `accelerate`（建议安装）
- `peft`（仅在使用 `--lora-path` 时需要）

示例安装：

```bash
pip install torch transformers accelerate sentencepiece peft
```

## 2. 脚本功能

脚本会执行以下流程：

1. 加载指定的基础模型和 tokenizer
2. 如果提供 `--lora-path`，在基础模型上加载 LoRA 适配器
3. 读取数据集 JSON
4. 按 `split` 过滤样本，默认只取 `test`
5. 将每条样本的 `instruction` 和 `input` 拼成提示词
6. 调用模型生成预测
7. 将逐条结果保存为 `JSONL`
8. 计算简单指标并保存为 `JSON`

当前输出的是轻量级文本匹配指标，不是生物学专用指标：

- `exact_match`
- `token_f1`
- `char_overlap_ratio`

## 3. 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `--model-path` | 是 | 无 | 基础模型路径，或 Hugging Face 模型名 |
| `--lora-path` | 否 | `None` | LoRA 权重路径，兼容 LLaMA-Factory 导出结果 |
| `--dataset-path` | 否 | 指向默认 PFUD 文件 | 数据集 JSON 路径 |
| `--output-path` | 是 | 无 | 逐条预测输出路径，保存为 `.jsonl` |
| `--metrics-path` | 否 | `output_path` 同名 `.metrics.json` | 汇总指标输出路径 |
| `--split` | 否 | `test` | 要评测的数据分片 |
| `--max-samples` | 否 | `None` | 最多评测多少条，用于快速试跑 |
| `--limit` | 否 | `None` | `--max-samples` 的别名；若两者同时设置，取更小值 |
| `--batch-size` / `--bathc-size` | 否 | `1` | 批量推理大小；兼容 `--bathc-size` 拼写 |
| `--max-new-tokens` | 否 | `256` | 每条样本最大生成 token 数 |
| `--temperature` | 否 | `0.0` | 生成温度，`0` 表示贪心解码 |
| `--top-p` | 否 | `1.0` | 当 `temperature > 0` 时生效 |
| `--device` | 否 | `auto` | 设备，例如 `auto`、`cuda:0`、`cpu` |
| `--dtype` | 否 | `auto` | 精度，可选 `auto`、`float16`、`bfloat16`、`float32` |
| `--trust-remote-code` | 否 | 关闭 | 加载自定义模型代码时开启 |
| `--disable-chat-template` | 否 | 关闭 | 不使用 tokenizer 的 chat template，改用普通 prompt |
| `--verbose` | 否 | 关闭 | 打印逐条评测进度 |

## 4. 最常用命令

### 4.1 评测基础模型

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path Qwen/Qwen3-8B ^
  --dataset-path C:\Users\12240\Desktop\ssh\protdata\PFUD_replaced_with_neighbors_filtered_top1ge20.json ^
  --output-path C:\Users\12240\Desktop\ssh\protdata\outputs\qwen3_pfud_test.jsonl ^
  --metrics-path C:\Users\12240\Desktop\ssh\protdata\outputs\qwen3_pfud_test.metrics.json ^
  --split test ^
  --batch-size 4 ^
  --device cuda:0 ^
  --dtype bfloat16
```
python protcot/scripts/evaluate_pfud_test.py --model-path Qwen/Qwen3-8B --dataset-path data/Protcot/PFUD_replaced_with_neighbors_filtered_top1ge20.json --output-path /home/aiscuser/jxlei/protcot/logs  --split test  --batch-size 1 --device cuda:0 --dtype bfloat16
说明：

- 这是最标准的用法
- 如果模型支持聊天模板，脚本会优先使用 tokenizer 自带的 chat template

### 4.2 评测基础模型 + LoRA

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path C:\models\Meta-Llama-3-8B-Instruct ^
  --lora-path C:\models\llamafactory\pfud_lora_ckpt ^
  --dataset-path C:\Users\12240\Desktop\ssh\protdata\PFUD_replaced_with_neighbors_filtered_top1ge20.json ^
  --output-path C:\Users\12240\Desktop\ssh\protdata\outputs\llama3_lora_pfud_test.jsonl ^
  --split test ^
  --batch-size 2 ^
  --device cuda:0 ^
  --dtype float16
```

说明：

- 这里的 `--lora-path` 指向 LoRA adapter 目录
- 基础模型必须和 LoRA 训练时使用的 base model 对应

### 4.3 小规模试跑

先跑少量样本，确认模型能正常生成：

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path Qwen/Qwen3-8B ^
  --output-path C:\Users\12240\Desktop\ssh\protdata\outputs\debug.jsonl ^
  --limit 20 ^
  --verbose
```

说明：

- 推荐第一次运行先用 `--max-samples 10` 或 `20`
- 可以先验证显存、prompt 格式、输出文件格式是否正常

### 4.4 不使用 chat template

如果某些模型 chat template 不稳定，或你希望完全按原始文本拼 prompt：

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path C:\models\your-model ^
  --output-path C:\Users\12240\Desktop\ssh\protdata\outputs\plain_prompt.jsonl ^
  --disable-chat-template
```

这时脚本会使用如下形式的 prompt：

```text
User: {instruction + input}
Assistant:
```

## 5. 输出文件说明

脚本会产出两个文件：

### 5.1 逐条结果文件：`JSONL`

每行是一条样本，包含：

- `index`
- `accesion`
- `split`
- `instruction`
- `input`
- `reference`
- `prediction`
- `generated_tokens`
- `metrics`

示例结构：

```json
{
  "index": 0,
  "accesion": "afdb_accesion_xxx",
  "split": "test",
  "instruction": "...",
  "input": "...",
  "reference": "...",
  "prediction": "...",
  "generated_tokens": 123,
  "metrics": {
    "exact_match": 0.0,
    "token_f1": 0.42,
    "char_overlap_ratio": 0.57
  }
}
```

### 5.2 汇总指标文件：`JSON`

包含：

- 模型路径
- LoRA 路径
- 数据集路径
- 评测参数
- 平均指标

示例结构：

```json
{
  "model_path": "Qwen/Qwen3-8B",
  "lora_path": null,
  "dataset_path": "C:\\Users\\12240\\Desktop\\ssh\\protdata\\PFUD_replaced_with_neighbors_filtered_top1ge20.json",
  "split": "test",
  "max_samples": null,
  "batch_size": 4,
  "max_new_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0,
  "metrics": {
    "num_samples": 5655,
    "exact_match": 0.0,
    "token_f1": 0.31,
    "char_overlap_ratio": 0.48,
    "avg_generated_tokens": 147.2
  }
}
```

## 6. 数据集说明

默认目标文件是：

- `PFUD_replaced_with_neighbors_filtered_top1ge20.json`

该文件的分片规模为：

- `train`: 368,288
- `valid`: 25,396
- `test`: 5,655

脚本默认只跑：

- `test`

如果要改跑验证集：

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path Qwen/Qwen3-8B ^
  --output-path C:\Users\12240\Desktop\ssh\protdata\outputs\qwen3_pfud_valid.jsonl ^
  --split valid
```

## 7. 注意事项

1. 当前脚本是逐条推理，不是 batch 推理，优点是简单稳妥，缺点是速度一般。
2. `token_f1` 和 `char_overlap_ratio` 只是通用文本重叠指标，不能替代人工评估或专业生物任务指标。
3. `PFUD` 是描述生成任务，`exact_match` 通常会很低，这很正常。
4. 如果使用 LoRA，必须保证 `--model-path` 和 LoRA 对应的基础模型一致。
5. 某些模型必须开启 `--trust-remote-code` 才能正确加载。
6. 某些模型的 chat template 可能不适配当前任务，此时可加 `--disable-chat-template`。
7. 默认数据文件较大，但脚本只筛选目标 split；首次调试仍建议加 `--max-samples`。

## 8. 常见问题

### Q1. 报错 `peft is required for --lora-path`

原因：

- 你传了 `--lora-path`，但环境里没有安装 `peft`

解决：

```bash
pip install peft
```

### Q2. 显存不足

可尝试：

- 换更小模型
- 使用 `--dtype float16` 或 `--dtype bfloat16`
- 改成 `--device cpu`（会慢很多）
- 先用 `--max-samples` 小规模测试

### Q3. 输出很短或为空

可尝试：

- 调大 `--max-new-tokens`
- 检查模型是否适合指令跟随
- 加上或去掉 `--disable-chat-template` 对比效果
- 确认 LoRA 是否与基础模型匹配

## 9. 推荐使用顺序

建议按这个顺序跑：

1. 先用 `--max-samples 10` 检查脚本和模型是否能正常生成
2. 再用完整 `test` 集跑正式评测
3. 最后对比基础模型和 LoRA 模型的 `.metrics.json`
## 10. 数据集格式要求

这个脚本默认读取的是一个 `JSON array` 文件，也就是：

```json
[
  { ... },
  { ... }
]
```

顶层必须是列表，列表里的每一项都是一个样本对象。

### 10.1 脚本实际依赖的字段

脚本运行时会读取这些字段：

| 字段名 | 是否必需 | 作用 |
|---|---|---|
| `instruction` | 是 | 主指令，会进入 prompt |
| `input` | 否 | 补充输入，会拼接到 `instruction` 后面；缺失时按空字符串处理 |
| `output` | 否 | 参考答案，用于计算文本匹配指标；缺失时会导致指标失真，实际评测应提供 |
| `accesion` | 否 | 蛋白 ID，仅用于结果记录 |
| `split` | 是 | 数据分片；脚本按它筛选 `train` / `valid` / `test` |

最小可用格式是：

```json
[
  {
    "instruction": "Please examine the protein and describe its functional role.",
    "input": "Examine the given protein and share a brief overview of its attributes.",
    "output": "This protein is likely involved in ...",
    "accesion": "afdb_accesion_xxx",
    "split": "test"
  }
]
```

### 10.2 默认 PFUD 数据集的样本格式

默认数据集是：

- `C:\\Users\\12240\\Desktop\\ssh\\protdata\\PFUD_replaced_with_neighbors_filtered_top1ge20.json`

这个文件里的每条样本通常长这样：

```json
{
  "instruction": "Please examine the protein and describe its functional role, potential involvement in cellular processes, and its subcellular location.\n\nReference proteins (top-2 nearest):\nNeighbor 1 accession: ...\n< aa sequence>...</ aa sequence>\n< 3Di sequence>...</ 3Di sequence>\nNeighbor 2 accession: ...\n< aa sequence>...</ aa sequence>\n< 3Di sequence>...</ 3Di sequence>",
  "input": "Examine the given protein and share a brief overview of its attributes.\n< protein sequence>...</ protein sequence>\n< protein structure>...</ protein structure>",
  "output": "Upon analysis of the specified structure, it is evident that the protein performs ...",
  "accesion": "afdb_accesion_A3CQH4",
  "split": "test",
  "task": "PFUD"
}
```

其中：

- `instruction` 里已经包含邻居参考蛋白信息
- `input` 是当前样本的问题，有些样本会附带目标蛋白的序列和结构，有些不会
- `output` 是自然语言参考答案
- `task` 字段存在与否都不影响当前脚本运行，脚本不会用它筛选

### 10.3 `split` 的取值要求

脚本通过 `--split` 参数筛选样本，所以数据里必须有可用的 `split` 字段。

常见取值：

- `train`
- `valid`
- `test`

另外，脚本内部会把：

- `validation` 视为 `valid`

也就是说，如果你的数据写的是：

```json
{ "split": "validation" }
```

传参时用：

```bash
--split valid
```

也能被正确选中。

### 10.4 不符合格式时会发生什么

如果数据格式不符合要求，常见问题如下：

1. 顶层不是 JSON 列表：
脚本会直接报错，提示文件不是 `JSON array`。

2. 没有 `split`：
样本无法被筛选，最终可能报 `No samples found for split=...`。

3. 没有 `instruction`：
模型拿不到有效 prompt，结果会不可用。

4. 没有 `output`：
脚本仍可能跑完，但指标没有参考意义。

### 10.5 自定义数据集建议

如果你想拿这个脚本评测别的文件，建议至少保证：

- 顶层是 JSON 列表
- 每条样本都有 `instruction`
- 每条样本都有 `split`
- 做正式评测时提供 `output`
- `instruction` 和 `input` 最终能拼成清晰的单轮 prompt

这样就可以直接通过：

```bash
python protcot/scripts/evaluate_pfud_test.py ^
  --model-path your_model ^
  --dataset-path your_dataset.json ^
  --output-path your_output.jsonl ^
  --split test
```

来复用这个脚本。
