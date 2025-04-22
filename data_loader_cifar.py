#data_loader_cifar

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms, datasets
import os
import numpy as np
from PIL import Image

class TransformSubset(Dataset):
    """
    一个包装 Subset 的数据集类，可以为子集应用特定的 transform。
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def load_cifar10_data(batch_size=64, known_labels=[1,2,3,4,5,6,7,8,9], unknown_label=0,
                     save_dir='./data/cifar10', device=torch.device("cpu")):
    """
    使用本地已分好类的 CIFAR-10 数据集目录加载数据，并将 'airplane' 类别作为未知集，其余作为已知集。
    已知类进行9:1划分为train和test，未知类抽取1/10作为open_set。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 定义转换
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: torch.clamp(x, -1.5, 1.5))    # 硬截断
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: torch.clamp(x, -1.5, 1.5))    # 硬截断
    ])

    # 使用ImageFolder从本地加载全部数据
    full_dataset = datasets.ImageFolder(root=save_dir)

    # full_dataset.classes 应该为 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 对应的label: airplane=0, automobile=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9

    # 区分已知类和未知类的索引
    indices = list(range(len(full_dataset)))
    labels_list = [full_dataset.samples[i][1] for i in indices]

    known_indices = [i for i, label in enumerate(labels_list) if label in known_labels]
    unknown_indices = [i for i, label in enumerate(labels_list) if label == unknown_label]

    known_subset = Subset(full_dataset, known_indices)
    open_set_subset_full = Subset(full_dataset, unknown_indices)

    # 从开放集(unknown)中抽取1/10的数据
    print("从开放集中抽取十分之一的数据...")
    total_open_set = len(open_set_subset_full)
    subset_size = max(1, total_open_set // 10)  # 确保至少有一个样本

    torch.manual_seed(42)
    if subset_size < total_open_set:
        rand_indices = torch.randperm(total_open_set)[:subset_size]
        open_set_subset = Subset(open_set_subset_full, rand_indices)
    else:
        print("开放集子集大小大于或等于原始开放集大小，使用完整开放集。")
        open_set_subset = open_set_subset_full

    print(f"开放集总大小: {total_open_set}, 选择子集大小: {len(open_set_subset)}")

    # 已知类进一步划分为训练集和测试集（9:1）
    num_known = len(known_subset)
    known_idx_list = list(range(num_known))
    np.random.shuffle(known_idx_list)
    split = int(np.floor(0.1 * num_known))
    test_indices = known_idx_list[:split]
    train_indices = known_idx_list[split:]

    train_subset = Subset(known_subset, train_indices)
    test_subset = Subset(known_subset, test_indices)

    # 为训练集和测试集、开放集分别应用不同的transform
    train_subset = TransformSubset(train_subset, transform_train)
    test_subset = TransformSubset(test_subset, transform_test)
    open_set_subset = TransformSubset(open_set_subset, transform_test)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    open_set_loader = DataLoader(
        open_set_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 打印样本数量
    print(f"已知训练集样本数量: {len(train_subset)}")
    print(f"已知测试集样本数量: {len(test_subset)}")
    print(f"开放集样本数量: {len(open_set_subset)}")

    return train_loader, test_loader, open_set_loader
