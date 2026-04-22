#!/bin/bash
# 快速训练脚本 - 使用flower_photos数据集

# 激活conda环境
source /home/master/software/3_Develop/3_0_environment/3_0_1_python/3_0_1_1_manager/anaconda3/etc/profile.d/conda.sh
conda activate vit_0421

# 进入工作目录
cd /home/master/file/vision_transformer/vision_transformer

# 训练参数说明:
# --data-path: 数据集路径
# --num_classes: 类别数(flower_photos有5类)
# --epochs: 训练轮数
# --batch-size: 批次大小(根据GPU显存调整)
# --lr: 学习率
# --fuse_token: 启用token融合
# --base_keep_rate: token保留率(0.7表示保留70%)
# --shrink_start_epoch: 从第几个epoch开始收缩token
# --shrink_epochs: token收缩持续的epoch数
# --exp_name: 实验名称

echo "开始训练 ViT with fuse_token..."
echo "数据集: flower_photos (5类)"
echo "配置: fuse_token=True, keep_rate=0.7"
echo "================================"

python train_fuse.py \
    --data-path ./data/flower_photos \
    --num_classes 5 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --fuse_token \
    --base_keep_rate 0.7 \
    --shrink_start_epoch 5 \
    --shrink_epochs 10 \
    --exp_name vit_fuse_flower \
    --device cuda:0

echo "================================"
echo "训练完成!"
echo "查看结果: tensorboard --logdir=./runs/vit_fuse_flower"
echo "最佳模型: ./weights/vit_fuse_flower_best.pth"
