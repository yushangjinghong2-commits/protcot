#!/bin/bash

# 多 GPU 训练启动脚本
# 使用方法: bash train_multi_gpu.sh

# ========== 配置区域 ==========
# 指定要使用的 GPU（逗号分隔）
export CUDA_VISIBLE_DEVICES=3,4,5,6

# GPU 数量（根据上面的 GPU 数量设置）
NPROC_PER_NODE=4

# 网络配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=32292

# NCCL 配置（根据你的服务器需求）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# 可选：NCCL 调试信息
# export NCCL_DEBUG=INFO

# 可选：超时设置（秒）
export NCCL_TIMEOUT=1800

# 训练配置文件路径
TRAIN_CONFIG="/home/aiscuser/jxlei/protcot/train.yaml"

# ========== 启动训练 ==========
echo "=========================================="
echo "多 GPU 训练配置"
echo "=========================================="
echo "GPU 设备: $CUDA_VISIBLE_DEVICES"
echo "GPU 数量: $NPROC_PER_NODE"
echo "Master 地址: $MASTER_ADDR:$MASTER_PORT"
echo "配置文件: $TRAIN_CONFIG"
echo "=========================================="
echo ""

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m llamafactory.cli train $TRAIN_CONFIG

echo ""
echo "训练完成或已退出"
