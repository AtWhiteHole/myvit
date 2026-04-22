#!/bin/bash
# 对比实验脚本 - 测试不同配置

source /home/master/software/3_Develop/3_0_environment/3_0_1_python/3_0_1_1_manager/anaconda3/etc/profile.d/conda.sh
conda activate vit_0421
cd /home/master/file/vision_transformer/vision_transformer

echo "======================================"
echo "运行对比实验"
echo "======================================"

# 实验1: 标准ViT (baseline, 不使用fuse_token)
echo ""
echo "实验1: 标准ViT (baseline)"
python train_fuse.py \
    --data-path ./data/flower_photos \
    --num_classes 5 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --exp_name vit_baseline \
    --device cuda:0

# 实验2: ViT + fuse_token (keep_rate=0.7)
echo ""
echo "实验2: ViT + fuse_token (keep_rate=0.7)"
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
    --exp_name vit_fuse_0.7 \
    --device cuda:0

# 实验3: ViT + fuse_token (keep_rate=0.5)
echo ""
echo "实验3: ViT + fuse_token (keep_rate=0.5)"
python train_fuse.py \
    --data-path ./data/flower_photos \
    --num_classes 5 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --fuse_token \
    --base_keep_rate 0.5 \
    --shrink_start_epoch 5 \
    --shrink_epochs 10 \
    --exp_name vit_fuse_0.5 \
    --device cuda:0

echo ""
echo "======================================"
echo "所有实验完成!"
echo "======================================"
echo "查看对比结果:"
echo "tensorboard --logdir=./runs"
