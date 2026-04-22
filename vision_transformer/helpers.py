import math
import torch

# ⭐新增代码说明: 从evit/helpers.py移植的辅助函数
# 用途: 支持fuse_token机制的核心工具函数
# 来源: /home/master/file/vision_transformer/evit/helpers.py

def adjust_keep_rate(iters, epoch, warmup_epochs, total_epochs,
                      ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1):
    """
    ⭐新增函数: 动态调整token保留率,使用余弦退火策略
    来源: evit/helpers.py 第7-18行

    功能说明:
    - 在warmup阶段保持keep_rate=1.0(不剪枝)
    - warmup后使用余弦退火从1.0逐渐降到base_keep_rate
    - 实现渐进式token剪枝,避免训练初期剪枝过多影响收敛

    Args:
        iters: 当前迭代次数
        epoch: 当前epoch
        warmup_epochs: 预热epoch数,在此期间keep_rate=1
        total_epochs: token收缩的总epoch数
        ITERS_PER_EPOCH: 每个epoch的迭代次数
        base_keep_rate: 最小保留率
        max_keep_rate: 最大保留率

    Returns:
        当前的keep_rate值
    """
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate


def complement_idx(idx, dim):
    """
    ⭐新增函数: 计算补集索引,用于token融合
    来源: evit/helpers.py 第52-73行

    功能说明:
    - 计算 set(range(dim)) - set(idx)
    - 用于找出被剪枝的token索引
    - fuse_token机制的核心函数,用于确定哪些token需要融合

    使用场景:
    在Block的forward中,当执行token剪枝后:
    1. idx保存了被保留的token索引
    2. complement_idx计算出被剪枝的token索引
    3. 使用这些索引提取被剪枝token的特征和注意力权重
    4. 进行加权融合

    Args:
        idx: 输入索引, shape: [N, *, K] (被保留的token索引)
        dim: 补集的最大索引 (总token数)

    Returns:
        补集索引, shape: [N, *, dim-K] (被剪枝的token索引)
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
