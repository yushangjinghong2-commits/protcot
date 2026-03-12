
### 基础训练命令

# 单 GPU 训练
source /home/aiscuser/jxlei/LlamaFactory/.venv/bin/activate
cd /home/aiscuser/jxlei/LlamaFactory
llamafactory-cli train /home/aiscuser/jxlei/protcot/train_llama_nostructure.yaml

# 推理 ：`evaluate_pfud_test.py` 使用说明
source /home/aiscuser/jxlei/.venv/bin/activate
python /home/aiscuser/jxlei/protcot/scripts/evaluate_pfud.py --model-path  Qwen/Qwen3-8B  --lora-path /home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b/lora/sft/checkpoint-10000  --dataset-path /home/aiscuser/jxlei/LlamaFactory/data/PFUD_replaced_no_structure.json --output-path /home/aiscuser/jxlei/protcot/predictions/qwen-ft-nostructure --split test --batch-size 1000

# 计算评价指标  calculate_value.py 使用说明
source /home/aiscuser/jxlei/.venv/bin/activate
python /home/aiscuser/jxlei/protcot/scripts/calculate_value.py /home/aiscuser/jxlei/protcot/qwen/logs


#  run_evaluation_pipeline.sh 脚本

bash /home/aiscuser/jxlei/protcot/scripts/run_evaluation_pipeline.sh --model-path Qwen/Qwen3-8B --lora-path /home/aiscuser/jxlei/LlamaFactory/saves/qwen3-8b/lora/sft/checkpoint-10000 --dataset-path /home/aiscuser/jxlei/Protcot/aa/PFUD_replaced_with_aa_neighbors.json --output-dir /home/aiscuser/jxlei/protcot/results/qwen3-lora --batch-size 1000


### 项目结构
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