"""
Training script for ViT with fuse_token support
⭐新增代码说明: 本文件基于原始train.py修改,增加了fuse_token机制的训练支持
"""
import os
import math
import argparse
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
# ⭐新增代码说明: 导入支持fuse_token的模型
from vit_model_fuse import vit_base_patch16_224_fuse as create_model
from utils import read_split_data
# ⭐新增代码说明: 导入动态keep_rate调整函数
from helpers import adjust_keep_rate


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, writer=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = data_loader
    ITERS_PER_EPOCH = len(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        # ⭐新增代码说明: 动态调整keep_rate
        # 使用余弦退火策略,从1.0逐渐降低到base_keep_rate
        # warmup阶段(shrink_start_epoch之前)保持keep_rate=1.0
        # shrink阶段(shrink_epochs期间)逐渐降低keep_rate
        it = epoch * ITERS_PER_EPOCH + step
        keep_rate = adjust_keep_rate(
            it, epoch,
            warmup_epochs=args.shrink_start_epoch,
            total_epochs=args.shrink_start_epoch + args.shrink_epochs,
            ITERS_PER_EPOCH=ITERS_PER_EPOCH,
            base_keep_rate=args.base_keep_rate
        )

        # ⭐新增代码说明: 前向传播时传入动态keep_rate
        pred = model(images, keep_rate=keep_rate)
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # 记录日志
        if writer is not None and step % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), it)
            writer.add_scalar('train/keep_rate', keep_rate, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], it)

        if step % 100 == 0:
            print(f"[epoch {epoch}] step {step}/{len(data_loader)}, "
                  f"loss: {accu_loss.item() / (step + 1):.3f}, "
                  f"acc: {accu_num.item() / sample_num:.3f}, "
                  f"keep_rate: {keep_rate:.3f}")

    return accu_loss.item() / len(data_loader), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, keep_rate=None):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images, keep_rate=keep_rate)
        loss = loss_function(pred, labels)
        accu_loss += loss.detach()

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

    print(f"[epoch {epoch}] val_loss: {accu_loss.item() / len(data_loader):.3f}, "
          f"val_acc: {accu_num.item() / sample_num:.3f}")

    return accu_loss.item() / len(data_loader), accu_num.item() / sample_num


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 创建模型
    print(f"Creating model with fuse_token={args.fuse_token}, base_keep_rate={args.base_keep_rate}")
    model = create_model(
        num_classes=args.num_classes,
        keep_rate=1.0,  # 初始keep_rate为1.0,训练时动态调整
        fuse_token=args.fuse_token
    ).to(device)

    # 加载预训练权重
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias']
        if model.has_logits:
            del_keys = ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']

        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结层
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # Cosine学习率调度
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            args=args,
            writer=tb_writer
        )

        scheduler.step()

        # 验证 - 使用base_keep_rate
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            keep_rate=args.base_keep_rate if args.fuse_token else None
        )

        # 记录到tensorboard
        tb_writer.add_scalar('epoch/train_loss', train_loss, epoch)
        tb_writer.add_scalar('epoch/train_acc', train_acc, epoch)
        tb_writer.add_scalar('epoch/val_loss', val_loss, epoch)
        tb_writer.add_scalar('epoch/val_acc', val_acc, epoch)
        tb_writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]["lr"], epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/{args.exp_name}_best.pth")
            print(f"Saved best model with acc: {best_acc:.3f}")

        # 定期保存checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), f"./weights/{args.exp_name}_epoch{epoch}.pth")

    print(f"Training finished. Best accuracy: {best_acc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # fuse_token相关参数
    parser.add_argument('--fuse_token', action='store_true', help='启用token融合')
    parser.add_argument('--base_keep_rate', type=float, default=0.7, help='token保留率')
    parser.add_argument('--shrink_start_epoch', type=int, default=0, help='开始token收缩的epoch')
    parser.add_argument('--shrink_epochs', type=int, default=10, help='token收缩持续的epoch数')

    # 数据和模型参数
    parser.add_argument('--data-path', type=str, default="./data/flower_photos/train")
    parser.add_argument('--weights', type=str, default='', help='预训练权重路径')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id')

    # 实验参数
    parser.add_argument('--exp_name', type=str, default='vit_fuse', help='实验名称')
    parser.add_argument('--save_freq', type=int, default=5, help='保存checkpoint的频率')

    opt = parser.parse_args()

    main(opt)
