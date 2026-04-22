# ViT with fuse_token 使用说明

## 概述

本项目成功将EViT的fuse_token机制集成到基础ViT实现中。fuse_token是一种信息保留的token剪枝技术,在降低计算成本的同时保持模型精度。

## 核心机制

### fuse_token工作原理

1. **Token剪枝**: 基于class token对patch token的注意力分数,选择top-k个最重要的token
2. **Token融合**: 将被剪枝的token通过注意力加权平均融合成一个token
3. **信息保留**: 融合后的token保留了被丢弃token的信息,提高鲁棒性

公式: `fused_token = sum(pruned_tokens * attention_weights)`

最终输出: `[cls_token, kept_tokens, fused_token]`

## 文件结构

```
vision_transformer/
├── vit_model_fuse.py      # 带fuse_token的增强ViT模型
├── helpers.py             # 辅助函数(complement_idx, adjust_keep_rate)
├── train_fuse.py          # 训练脚本
└── test_fuse_token.py     # 测试脚本
```

## 安装和测试

### 1. 环境要求

使用conda环境 `vit_0421`:
```bash
conda activate vit_0421
```

### 2. 运行测试

验证fuse_token实现:
```bash
cd /home/master/file/vision_transformer/vision_transformer
python test_fuse_token.py
```

测试包括:
- ✓ 输出形状验证
- ✓ 前向传播测试
- ✓ 不同keep_rate值测试
- ✓ 梯度反向传播验证
- ✓ 有无fuse_token对比

## 训练使用

### 基础训练命令

```bash
python train_fuse.py \
    --data-path ./data/your_dataset \
    --num_classes 1000 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --fuse_token \
    --base_keep_rate 0.7 \
    --shrink_start_epoch 10 \
    --shrink_epochs 20 \
    --exp_name vit_fuse_exp1
```

### 关键参数说明

#### fuse_token相关参数

- `--fuse_token`: 启用token融合机制(flag参数)
- `--base_keep_rate`: token保留率,范围(0, 1],默认0.7
  - 0.7表示保留70%的token
  - 1.0表示不剪枝(标准ViT)
- `--shrink_start_epoch`: 开始token收缩的epoch,默认0
- `--shrink_epochs`: token收缩持续的epoch数,默认10
  - 使用余弦退火从keep_rate=1.0逐渐降到base_keep_rate

#### 训练参数

- `--num_classes`: 分类类别数
- `--epochs`: 训练总epoch数
- `--batch-size`: batch大小
- `--lr`: 学习率
- `--lrf`: 学习率衰减因子,默认0.01
- `--weights`: 预训练权重路径
- `--freeze-layers`: 是否冻结除head外的层
- `--device`: 设备,如'cuda:0'或'cpu'
- `--exp_name`: 实验名称,用于保存模型和日志
- `--save_freq`: 保存checkpoint的频率(epoch)

### 训练示例

#### 1. 标准ViT训练(不使用fuse_token)

```bash
python train_fuse.py \
    --data-path ./data/imagenet \
    --num_classes 1000 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --exp_name vit_baseline
```

#### 2. 使用fuse_token训练(keep_rate=0.7)

```bash
python train_fuse.py \
    --data-path ./data/imagenet \
    --num_classes 1000 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --fuse_token \
    --base_keep_rate 0.7 \
    --shrink_start_epoch 10 \
    --shrink_epochs 20 \
    --exp_name vit_fuse_0.7
```

#### 3. 使用预训练权重微调

```bash
python train_fuse.py \
    --data-path ./data/your_dataset \
    --num_classes 5 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --weights ./pretrained/vit_base_patch16_224.pth \
    --freeze-layers True \
    --fuse_token \
    --base_keep_rate 0.7 \
    --exp_name vit_fuse_finetune
```

#### 4. 不同keep_rate对比实验

```bash
# keep_rate=0.9 (保留90%token)
python train_fuse.py --fuse_token --base_keep_rate 0.9 --exp_name vit_fuse_0.9

# keep_rate=0.7 (保留70%token)
python train_fuse.py --fuse_token --base_keep_rate 0.7 --exp_name vit_fuse_0.7

# keep_rate=0.5 (保留50%token)
python train_fuse.py --fuse_token --base_keep_rate 0.5 --exp_name vit_fuse_0.5
```

## 模型使用

### 加载模型进行推理

```python
import torch
from vit_model_fuse import vit_base_patch16_224_fuse

# 创建模型
model = vit_base_patch16_224_fuse(
    num_classes=1000,
    keep_rate=0.7,
    fuse_token=True
)

# 加载训练好的权重
checkpoint = torch.load('weights/vit_fuse_best.pth')
model.load_state_dict(checkpoint)
model.eval()

# 推理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x, keep_rate=0.7)
    pred = output.argmax(dim=1)
```

### 动态调整keep_rate

```python
# 训练时可以动态调整keep_rate
for epoch in range(epochs):
    for batch in dataloader:
        # 计算当前keep_rate
        keep_rate = adjust_keep_rate(
            iters=current_iter,
            epoch=epoch,
            warmup_epochs=10,
            total_epochs=30,
            ITERS_PER_EPOCH=len(dataloader),
            base_keep_rate=0.7
        )
        
        # 前向传播
        output = model(batch, keep_rate=keep_rate)
```

## 输出和日志

### 训练输出

训练过程会输出:
- 每100步的loss、accuracy、keep_rate
- 每个epoch的验证结果
- TensorBoard日志保存在`./runs/{exp_name}/`

### 模型保存

- 最佳模型: `./weights/{exp_name}_best.pth`
- 定期checkpoint: `./weights/{exp_name}_epoch{N}.pth`

### TensorBoard可视化

```bash
tensorboard --logdir=./runs
```

可视化内容:
- train/loss: 训练损失
- train/keep_rate: 当前token保留率
- train/lr: 学习率
- epoch/train_acc: 训练精度
- epoch/val_acc: 验证精度

## 性能预期

根据EViT论文结果:

| keep_rate | 精度变化 | 计算量节省 |
|-----------|---------|-----------|
| 1.0       | 基线    | 0%        |
| 0.9       | -0.1%   | ~10%      |
| 0.7       | -0.5%   | ~30%      |
| 0.5       | -1.5%   | ~50%      |

**注意**: 实际结果取决于数据集和训练配置。

## 实现细节

### 关键组件

1. **Attention类**: 增强的注意力机制
   - 计算class token对patch token的注意力
   - 执行top-k token选择
   - 返回token索引和注意力权重

2. **Block类**: Transformer block
   - 集成token融合逻辑
   - 根据fuse_token参数决定是否融合

3. **complement_idx函数**: 计算补集索引
   - 找出被剪枝的token索引
   - 用于token融合

4. **adjust_keep_rate函数**: 动态调整keep_rate
   - 余弦退火策略
   - 从1.0逐渐降到base_keep_rate

### 向后兼容性

- `keep_rate=1.0, fuse_token=False`: 等价于标准ViT
- 可以加载标准ViT预训练权重

## 常见问题

### Q1: fuse_token和不使用fuse_token的区别?

**A**: 
- 不使用fuse_token: 直接丢弃低注意力token,输出`[cls, kept_tokens]`
- 使用fuse_token: 将低注意力token融合,输出`[cls, kept_tokens, fused_token]`
- fuse_token保留更多信息,通常精度更高

### Q2: 如何选择keep_rate?

**A**: 
- 0.7是推荐的平衡点(精度损失小,计算节省明显)
- 可以通过对比实验找到最佳值
- 建议从0.7开始,根据需求调整

### Q3: 训练时间会增加吗?

**A**: 
- 使用fuse_token会略微增加计算量(融合操作)
- 但由于token数量减少,整体训练时间可能减少
- 推理时速度提升更明显

### Q4: 可以用于其他ViT变体吗?

**A**: 
- 可以,核心机制是通用的
- 需要修改对应的Attention和Block类
- 参考本实现的移植方式

## 下一步工作

当前已完成:
- ✓ fuse_token机制集成
- ✓ 训练脚本
- ✓ 测试验证

后续可以进行:
- 在ImageNet上训练并评估精度
- 测量实际推理速度提升
- 可视化被剪枝的token
- 实现知识蒸馏(使用fuse_token模型作为teacher)

## 参考

- EViT论文: "Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations"
- 原始ViT: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- EViT代码: `/home/master/file/vision_transformer/evit/`

## 联系和支持

如有问题,请检查:
1. 测试脚本是否全部通过
2. 训练日志中的keep_rate是否正常变化
3. TensorBoard可视化是否正常
