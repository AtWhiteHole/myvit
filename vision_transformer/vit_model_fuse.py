"""
Enhanced Vision Transformer with fuse_token support
Based on original ViT implementation with token fusion mechanism from EViT

⭐文件说明:
- 基础代码来源: vision_transformer/vit_model.py (原始ViT实现)
- 新增功能来源: evit/evit.py (fuse_token机制)
- 主要修改: Attention类、Block类、VisionTransformer类
- 新增参数: keep_rate, fuse_token
"""
from functools import partial
from collections import OrderedDict
import math

import torch
import torch.nn as nn

from helpers import complement_idx


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    # ⭐新增代码说明: 增强的Attention类,支持动态token剪枝
    # 来源: evit/evit.py 第178-225行
    # 主要修改:
    # 1. 新增keep_rate参数控制token保留率
    # 2. 计算class token对patch token的注意力分数
    # 3. 执行top-k选择保留重要token
    # 4. 返回token索引和注意力权重供后续融合使用

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0., keep_rate=1.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        # ⭐新增: keep_rate参数
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, f"keep_rate must > 0 and <= 1, got {keep_rate}"

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape

        # 标准的QKV计算和注意力机制(与原始ViT相同)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # ⭐新增: token剪枝逻辑
        # 当keep_rate < 1时,执行token选择
        left_tokens = N - 1
        if (self.keep_rate < 1 and keep_rate < 1) or tokens is not None:
            # 计算需要保留的token数量
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1

            # ⭐核心: 计算class token对所有patch token的注意力
            # cls_attn shape: [B, H, N-1] -> [B, N-1]
            cls_attn = attn[:, :, 0, 1:]  # 提取class token的注意力
            cls_attn = cls_attn.mean(dim=1)  # 对多头注意力求平均

            # ⭐核心: top-k选择,保留注意力分数最高的token
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            # 返回: (输出, 索引, token_id, 注意力分数, 保留数量)
            return x, index, idx, cls_attn, left_tokens

        return x, None, None, None, left_tokens


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    # ⭐新增代码说明: 增强的Transformer Block,支持token融合
    # 来源: evit/evit.py 第228-278行
    # 主要修改:
    # 1. 新增fuse_token参数控制是否融合被剪枝的token
    # 2. 在attention后执行token选择和融合
    # 3. 使用complement_idx找出被剪枝的token
    # 4. 对被剪枝token进行注意力加权融合

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=1., fuse_token=False):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              keep_rate=keep_rate)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # ⭐新增: keep_rate和fuse_token参数
        self.keep_rate = keep_rate
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape

        # 标准的attention计算(与原始ViT相同)
        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        # ⭐新增: token选择和融合逻辑
        if index is not None:
            non_cls = x[:, 1:]  # 去除class token
            # 取出保留的token
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                # ⭐核心: token融合机制
                # 步骤1: 计算补集索引(被剪枝的token)
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]

                # 步骤2: 提取被剪枝token的特征
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))

                # 步骤3: 提取被剪枝token的注意力权重
                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

                # 步骤4: 注意力加权求和,创建融合token
                # extra_token = sum(被剪枝token * 对应注意力权重)
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]

                # 步骤5: 拼接 [cls_token, kept_tokens, fused_token]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                # 不融合: 只保留 [cls_token, kept_tokens]
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        # 标准的MLP计算(与原始ViT相同)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1

        if get_idx and index is not None:
            return x, n_tokens, idx

        return x


class VisionTransformer(nn.Module):
    # ⭐新增代码说明: 增强的VisionTransformer,支持动态token剪枝和融合
    # 来源: evit/evit.py 第281-350行
    # 主要修改:
    # 1. 新增keep_rate参数(默认None,即不剪枝)
    # 2. 新增fuse_token参数(默认False,即不融合)
    # 3. 将参数传递给所有Block层
    # 4. 支持每层不同的keep_rate(可传入tuple)

    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 keep_rate=None, fuse_token=False):
        """
        Args:
            img_size: 输入图像大小
            patch_size: patch大小
            in_c: 输入通道数
            num_classes: 分类类别数
            embed_dim: embedding维度
            depth: transformer深度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: 是否在qkv中使用bias
            qk_scale: 覆盖默认的qk scale
            representation_size: pre-logits层维度
            distilled: 是否使用蒸馏token
            drop_ratio: dropout率
            attn_drop_ratio: 注意力dropout率
            drop_path_ratio: stochastic depth率
            embed_layer: patch embedding层
            norm_layer: 归一化层
            act_layer: 激活函数
            keep_rate: ⭐新增 token保留率,可以是单个值或每层的tuple
            fuse_token: ⭐新增 是否启用token融合
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # ⭐新增代码说明: 处理keep_rate参数
        # 支持三种输入格式:
        # 1. None -> 所有层keep_rate=1(不剪枝)
        # 2. 单个数值 -> 所有层使用相同的keep_rate
        # 3. tuple -> 每层使用不同的keep_rate
        if keep_rate is None:
            keep_rate = (1,) * depth
        elif isinstance(keep_rate, (int, float)):
            keep_rate = (keep_rate,) * depth
        assert len(keep_rate) == depth, f"keep_rate length {len(keep_rate)} != depth {depth}"

        # ⭐新增: 保存keep_rate和fuse_token参数
        self.keep_rate = keep_rate
        self.fuse_token = fuse_token

        # transformer encoder
        # ⭐新增代码说明: 创建Block时传入keep_rate和fuse_token参数
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  keep_rate=keep_rate[i], fuse_token=fuse_token)  # ⭐传入每层的keep_rate和fuse_token
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, keep_rate=None):
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        # 通过transformer blocks
        for block in self.blocks:
            x = block(x, keep_rate=keep_rate)

        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, keep_rate=None):
        x = self.forward_features(x, keep_rate=keep_rate)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_fuse(num_classes: int = 1000, keep_rate=1.0, fuse_token=False):
    """
    ViT-Base with fuse_token support
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes,
        keep_rate=keep_rate,
        fuse_token=fuse_token
    )
    return model
