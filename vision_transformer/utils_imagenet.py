import os
import sys
import json
import pickle

import torch
from tqdm import tqdm


def read_imagenet_data(root: str):
    """
    读取ImageNet-1K数据集，支持标准的train/val目录结构
    train: 1000个类别文件夹
    val: 可以是类别文件夹或单独的图片文件（需要val_labels.txt）
    """
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    assert os.path.exists(train_root), "train directory does not exist."
    assert os.path.exists(val_root), "val directory does not exist."

    # 读取训练集类别
    class_names = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]
    class_names.sort()

    # 生成类别索引映射
    class_indices = dict((k, v) for v, k in enumerate(class_names))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print(f"Found {len(class_names)} classes")

    # 读取训练集
    train_images_path = []
    train_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG", ".jpeg"]

    print("Loading training data...")
    for cla in tqdm(class_names):
        cla_path = os.path.join(train_root, cla)
        images = [os.path.join(train_root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    # 读取验证集
    val_images_path = []
    val_images_label = []

    print("Loading validation data...")
    # 检查val目录是否有子文件夹（类别文件夹）
    val_subdirs = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]

    if len(val_subdirs) > 0 and val_subdirs[0] in class_names:
        # val目录按类别组织
        print("Validation data organized by class folders")
        for cla in tqdm(class_names):
            cla_path = os.path.join(val_root, cla)
            if not os.path.exists(cla_path):
                continue
            images = [os.path.join(val_root, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            images.sort()
            image_class = class_indices[cla]

            for img_path in images:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
    else:
        # val目录所有图片在一起，需要标签文件
        print("Validation data in flat structure, looking for label file...")
        val_label_file = os.path.join(root, 'val_labels.txt')

        if os.path.exists(val_label_file):
            # 读取标签文件（格式：filename class_idx）
            with open(val_label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_idx = int(parts[1])
                        img_path = os.path.join(val_root, img_name)
                        if os.path.exists(img_path):
                            val_images_path.append(img_path)
                            val_images_label.append(class_idx)
        else:
            # 没有标签文件，尝试直接读取所有图片（用于测试，标签设为0）
            print("Warning: No val_labels.txt found, using dummy labels")
            val_images = [os.path.join(val_root, i) for i in os.listdir(val_root)
                          if os.path.splitext(i)[-1] in supported]
            val_images.sort()
            for img_path in val_images:
                val_images_path.append(img_path)
                val_images_label.append(0)  # dummy label

    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
