# Protein Fine-tuning with LlamaFactory

基于 LlamaFactory 框架对 Qwen3-8B 模型进行蛋白质相关任务的微调。

## 项目概述

本项目使用 LoRA 方法对 Qwen3-8B 模型进行微调，训练数据为 PFUD (Protein Function Understanding Dataset) 数据集，用于蛋白质功能预测、结构分析等任务。

## 环境配置

### 安装 LlamaFactory

```bash
cd /home/aiscuser/jxlei/LlamaFactory
pip install -e ".[torch,metrics]"
```

### 依赖要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐)
- 显存: 至少 24GB (单卡 A100/A6000)

## 数据集说明

### 数据集位置

```
/home/aiscuser/jxlei/LlamaFactory/data/
├── PFUD_replaced_no_structure.json          # 主训练数据集 (无结构信息)
├── PFUD_replaced.json                        # 完整数据集
├── PFUD_replaced_with_neighbors.json         # 包含邻居信息
├── PFUD_replaced_with_neighbors_filtered.json
└── PFUD_replaced_with_neighbors_filtered_top1ge20.json
```

### 数据格式

数据集采用标准 Alpaca 格式：

```json
{
  "instruction": "任务指令",
  "input": "蛋白质序列和上下文",
  "output": "预期输出",
  "accesion": "蛋白质ID",
  "split": "train/test",
  "task": "PFUD"
}
```

### 数据集注册

所有 PFUD 数据集已在 `/home/aiscuser/jxlei/LlamaFactory/data/dataset_info.json` 中注册。

## 训练配置

### 配置文件: `train.yaml`

```yaml
### 模型配置
model_name_or_path: Qwen/Qwen3-8B
trust_remote_code: true

### 微调方法
stage: sft                    # 监督微调
finetuning_type: lora         # LoRA 方法
lora_rank: 16                 # LoRA 秩 (提升表达能力)
lora_alpha: 32                # LoRA alpha (通常是 rank 的 2 倍)
lora_target: all              # 对所有层应用 LoRA

### 数据集配置
dataset: PFUD_replaced_no_structure
template: qwen3_nothink
cutoff_len: 3096              # 最大序列长度 (适应长蛋白质序列)
max_samples: 400000           # 最大训练样本数
preprocessing_num_workers: 16
dataloader_num_workers: 4

### 输出配置
output_dir: saves/qwen3-8b/lora/sft
logging_steps: 10             # 每 10 步记录日志
save_steps: 5000              # 每 5000 步保存检查点
plot_loss: true               # 绘制损失曲线
report_to: tensorboard        # 使用 TensorBoard 监控

### 训练超参数
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # 有效批次大小 = 1 × 8 = 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true                    # 使用 bfloat16 精度
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `lora_rank` | 16 | LoRA 秩，控制适配器容量 |
| `lora_alpha` | 32 | LoRA 缩放因子 |
| `cutoff_len` | 3096 | 最大序列长度，适应长蛋白质序列 |
| `max_samples` | 400000 | 40万训练样本 |
| `num_train_epochs` | 3.0 | 总共训练 120万样本 |
| `save_steps` | 5000 | 每 5000 步保存一次 |

### 训练步数估算

- 总样本数: 400,000
- 有效批次大小: 8
- 每 epoch 步数: 400,000 / 8 = 50,000 步
- 总训练步数: 50,000 × 3 = 150,000 步
- 保存检查点数: 150,000 / 5000 = 30 个

## 运行训练

### 基础训练命令

```bash
cd /home/aiscuser/jxlei/LlamaFactory

# 单 GPU 训练
llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml
```

### 多 GPU 训练

```bash
# 使用 2 个 GPU
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml

# 使用 4 个 GPU
CUDA_VISIBLE_DEVICES=3,4,5,6 llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml
```

### 从检查点恢复训练

如果训练中断，可以从最新检查点恢复：

```yaml
# 在 train.yaml 中修改
resume_from_checkpoint: saves/qwen3-8b/lora/sft/checkpoint-50000
```

或使用命令行覆盖：

```bash
llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml \
    resume_from_checkpoint=saves/qwen3-8b/lora/sft/checkpoint-50000
```

## 监控训练

### 1. 实时日志

训练过程中，终端会显示：
- 当前步数和 epoch
- 训练损失
- 学习率
- 训练速度 (samples/s)

### 2. TensorBoard

启动 TensorBoard 监控：

```bash
tensorboard --logdir saves/qwen3-8b/lora/sft
```

然后在浏览器打开 `http://localhost:6006`

可以查看：
- 训练损失曲线
- 学习率变化
- 梯度统计
- 系统资源使用

### 3. 检查保存的模型

```bash
ls -lh saves/qwen3-8b/lora/sft/
```

每个检查点包含：
- `adapter_config.json` - LoRA 配置
- `adapter_model.safetensors` - LoRA 权重
- `trainer_state.json` - 训练状态
- `training_args.bin` - 训练参数

## 推理和评估

### 命令行推理

创建推理配置文件 `inference.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3-8B
adapter_name_or_path: saves/qwen3-8b/lora/sft
template: qwen3_nothink
finetuning_type: lora
```

运行推理：

```bash
llamafactory-cli chat /home/aiscuser/jxlei/protcot/inference.yaml
```

### API 服务

启动 API 服务器：

```bash
llamafactory-cli api /home/aiscuser/jxlei/protcot/inference.yaml
```

然后可以通过 HTTP 请求调用模型。

### 合并 LoRA 权重

如果需要将 LoRA 权重合并到基础模型：

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-8B \
    --adapter_name_or_path saves/qwen3-8b/lora/sft \
    --template qwen3_nothink \
    --finetuning_type lora \
    --export_dir saves/qwen3-8b-merged \
    --export_size 2 \
    --export_device cpu
```

## 性能优化

### 显存不足时

如果遇到 OOM (Out of Memory) 错误：

1. **减小批次大小**
   ```yaml
   per_device_train_batch_size: 1  # 已经是最小值
   ```

2. **减小序列长度**
   ```yaml
   cutoff_len: 2048  # 从 3096 降低到 2048
   ```

3. **使用梯度检查点**
   ```yaml
   gradient_checkpointing: true
   ```

4. **使用 QLoRA**
   ```yaml
   finetuning_type: qlora
   quantization_bit: 4
   ```

### 加速训练

1. **增加批次大小** (如果显存充足)
   ```yaml
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 4  # 保持总批次大小 = 8
   ```

2. **使用 Flash Attention**
   ```yaml
   flash_attn: fa2
   ```

3. **减少保存频率**
   ```yaml
   save_steps: 10000  # 从 5000 增加到 10000
   ```

## 常见问题

### Q1: 训练速度慢？

**A:** 检查以下几点：
- 确保使用了 `bf16: true`
- 增加 `dataloader_num_workers`
- 使用更快的存储设备
- 考虑使用多 GPU 训练

### Q2: 损失不下降？

**A:** 尝试：
- 调整学习率 (1e-5 到 5e-4)
- 增加 `lora_rank` (如 32 或 64)
- 检查数据质量
- 增加训练轮数

### Q3: 如何选择最佳检查点？

**A:**
- 查看 TensorBoard 中的损失曲线
- 选择验证损失最低的检查点
- 在测试集上评估多个检查点

### Q4: 训练中断如何恢复？

**A:**
```bash
llamafactory-cli train /home/aiscuser/jxlei/protcot/train.yaml \
    resume_from_checkpoint=saves/qwen3-8b/lora/sft/checkpoint-XXXXX
```

## 项目结构

```
/home/aiscuser/jxlei/
├── LlamaFactory/                    # LlamaFactory 框架
│   ├── data/
│   │   ├── dataset_info.json       # 数据集注册文件
│   │   └── PFUD_*.json             # PFUD 数据集
│   └── examples/                    # 示例配置
├── protcot/
│   ├── train.yaml                   # 训练配置
│   ├── inference.yaml               # 推理配置 (待创建)
│   └── README.md                    # 本文档
└── data/Protcot/                    # 原始数据备份
```

## 参考资源

- [LlamaFactory 官方文档](https://llamafactory.readthedocs.io/)
- [LlamaFactory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Qwen3 模型](https://huggingface.co/Qwen)

## 更新日志

- **2026-03-07**: 初始配置，注册 PFUD 数据集，优化训练参数
  - LoRA rank 从 8 提升到 16
  - 添加 lora_alpha: 32
  - save_steps 调整为 5000
  - 启用 TensorBoard 监控

## 联系方式

如有问题，请查阅 LlamaFactory 官方文档或提交 Issue。
