# 环境配置指南

## 快速开始

### 方法1: 使用pip安装 (推荐)

```bash
# 创建新的conda环境
conda create -n vit_fuse python=3.8 -y
conda activate vit_fuse

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 或 CPU版本
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

### 方法2: 使用conda安装

```bash
# 创建环境
conda create -n vit_fuse python=3.8 -y
conda activate vit_fuse

# 安装PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install timm tensorboard tensorboardX tqdm pyyaml
```

### 方法3: 使用现有环境 (vit_0421)

如果你已经有配置好的环境:

```bash
conda activate vit_0421
# 检查是否缺少依赖
pip install -r requirements.txt
```

## 验证安装

运行测试脚本验证环境配置:

```bash
cd /home/master/file/vision_transformer/vision_transformer
python test_fuse_token.py
```

如果看到以下输出,说明环境配置成功:
```
🎉 所有测试通过!
fuse_token实现验证成功,可以开始训练了!
```

## 依赖说明

### 必需依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=1.9.0 | 深度学习框架 |
| torchvision | >=0.10.0 | 图像处理和预训练模型 |
| timm | >=0.4.12 | Vision Transformer工具库 |
| tensorboard | >=2.4.0 | 训练可视化 |
| numpy | >=1.19.0 | 数值计算 |
| pillow | >=8.0.0 | 图像读取 |

### 可选依赖

| 包名 | 用途 |
|------|------|
| torchprofile | 计算FLOPs和模型复杂度 |
| lmdb | 大规模数据集加载 |
| pyarrow | 高效数据序列化 |
| einops | 张量操作简化 |
| matplotlib | 结果可视化 |
| opencv-python | 图像处理 |
| tqdm | 进度条显示 |

## 不同CUDA版本的PyTorch安装

### CUDA 11.8
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### CPU版本
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

### 检查CUDA版本
```bash
nvidia-smi
# 或
nvcc --version
```

## 常见问题

### Q1: ImportError: No module named 'torch'

**解决方案:**
```bash
pip install torch torchvision
```

### Q2: CUDA out of memory

**解决方案:**
- 减小batch size: `--batch-size 32` 或 `--batch-size 16`
- 使用梯度累积
- 使用混合精度训练

### Q3: timm版本不兼容

**解决方案:**
```bash
pip install timm==0.4.12
# 或使用最新版本
pip install timm --upgrade
```

### Q4: 找不到helpers模块

**解决方案:**
确保在正确的目录下运行:
```bash
cd /home/master/file/vision_transformer/vision_transformer
python train_fuse.py
```

### Q5: tensorboard无法启动

**解决方案:**
```bash
pip install tensorboard --upgrade
tensorboard --logdir=./runs --port=6006
```

## 性能优化建议

### 1. 使用混合精度训练

在train_fuse.py中添加:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环中
with autocast():
    output = model(images, keep_rate=keep_rate)
    loss = loss_function(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 数据加载优化

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,  # 增加worker数量
    pin_memory=True,  # 启用pin_memory
    persistent_workers=True  # 保持worker进程
)
```

### 3. 使用更快的数据格式

对于大规模数据集,考虑使用LMDB或WebDataset格式。

## 环境导出和复现

### 导出当前环境

```bash
# 导出conda环境
conda env export > environment.yml

# 导出pip依赖
pip freeze > requirements_freeze.txt
```

### 从导出文件创建环境

```bash
# 使用conda
conda env create -f environment.yml

# 使用pip
pip install -r requirements_freeze.txt
```

## 多GPU训练配置

如果使用多GPU训练,需要安装:

```bash
pip install torch.distributed
```

训练命令:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_fuse.py \
    --data-path ./data \
    --batch-size 256 \
    --fuse_token
```

## Docker环境 (可选)

创建Dockerfile:
```dockerfile
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train_fuse.py"]
```

构建和运行:
```bash
docker build -t vit_fuse .
docker run --gpus all -v $(pwd)/data:/workspace/data vit_fuse
```

## 系统要求

### 最低配置
- Python: 3.8+
- GPU: NVIDIA GPU with 6GB+ VRAM (推荐8GB+)
- RAM: 16GB+
- 存储: 50GB+ (用于数据集和模型)

### 推荐配置
- Python: 3.8-3.10
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB+
- 存储: 500GB+ SSD

## 联系支持

如果遇到环境配置问题:
1. 检查Python版本: `python --version`
2. 检查CUDA版本: `nvidia-smi`
3. 检查PyTorch安装: `python -c "import torch; print(torch.__version__)"`
4. 运行测试脚本: `python test_fuse_token.py`
