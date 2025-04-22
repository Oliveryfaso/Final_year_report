# data_loader_MNIST.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from torchvision import transforms
import os
import numpy as np
from PIL import Image

class SignMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = self.data.iloc[:, 0].values  # 第一列为标签
        self.images = self.data.iloc[:, 1:].values.astype('float32') / 255.0  # pixel1 to pixel784, 归一化到 [0,1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将扁平的像素数据重塑为 28x28 的二维图像
        image = self.images[idx].reshape(28, 28)  # (28, 28)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            # 将 numpy 数组转换为 PIL 图像，再转换为 Tensor
            image = transforms.ToTensor()(Image.fromarray((image * 255).astype(np.uint8), mode='L'))
        return image, label

class CompleteDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # numpy arrays, shape: (N, 784)
        self.labels = labels  # numpy array, shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)  # 28x28
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class UnknownDataset(Dataset):
    def __init__(self, subset, unknown_label=0):
        """
        将子集标签设为 Unknown 类别。

        Args:
            subset (Subset): 原始子集。
            unknown_label (int): Unknown 类别的标签编号。
        """
        self.subset = subset
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        label = self.unknown_label
        return image, label

def save_combined_csv(train_csv, test_csv, combined_csv):
    """
    合并训练集和测试集的 CSV 文件，并保存为新的 CSV 文件。
    """
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_data.to_csv(combined_csv, index=False, mode='w')
    print(f"合并后的数据集已保存到 {combined_csv}")

def load_sign_mnist_data(batch_size=64, known_labels=range(1, 24), combined_csv='./data/handfigure/sign_mnist_combined.csv',
                        save_combined=True, train_csv='./data/handfigure/sign_mnist_train.csv',
                        test_csv='./data/handfigure/sign_mnist_test.csv', device=torch.device("cpu")):
    """
    加载 Sign MNIST 数据集，并根据已知标签划分训练集、测试集和开放集。

    Args:
        batch_size (int): 每个批次的样本数量。
        known_labels (iterable): 已知标签的范围（例如，1-23）。
        combined_csv (str): 合并后保存的 CSV 文件路径。
        save_combined (bool): 是否保存合并后的 CSV 文件。
        train_csv (str): 训练集 CSV 文件路径。
        test_csv (str): 测试集 CSV 文件路径。
        device (torch.device): 设备信息，用于设置 pin_memory。

    Returns:
        tuple: 包含 train_loader, test_loader, open_set_loader 的 DataLoader 对象。
    """
    if save_combined:
        save_combined_csv(train_csv, test_csv, combined_csv)
    else:
        if not os.path.exists(combined_csv):
            raise FileNotFoundError(f"合并后的 CSV 文件不存在：{combined_csv}")
        else:
            print(f"使用已存在的合并后的数据集 CSV 文件：{combined_csv}")

    # 定义转换，移除 Normalize 以保持 [0,1] 范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 ndarray 转换为 torch.Tensor
        # transforms.Normalize((0.5,), (0.5,))  # 移除这行
    ])

    # 加载合并后的数据集
    combined_dataset = SignMNISTDataset(csv_file=combined_csv, transform=transform)

    # 获取所有图像和标签
    # 假设 SignMNISTDataset 已正确返回图像和整数标签
    # 不需要额外的 all_images 和 all_labels

    # 创建完整的数据集
    complete_dataset = combined_dataset  # 不需要额外的 CompleteDataset，直接使用 combined_dataset

    # 划分为已知集和开放集
    known_indices = [i for i, (_, label) in enumerate(complete_dataset) if label in known_labels]
    unknown_indices = [i for i, (_, label) in enumerate(complete_dataset) if label not in known_labels]

    # 使用 Subset 创建对应的子集
    known_subset = Subset(complete_dataset, known_indices)
    open_set_subset = Subset(complete_dataset, unknown_indices)

    # 将开放集的标签统一映射为0
    open_set_subset = UnknownDataset(open_set_subset, unknown_label=0)

    # 从开放集中抽取十分之一的数据
    print("从开放集中抽取十分之一的数据...")
    total_open_set = len(open_set_subset)
    subset_size = max(1, total_open_set // 10)  # 确保至少有一个样本

    # 设置随机种子以保证可重复性
    torch.manual_seed(42)

    # 生成随机索引
    if subset_size < total_open_set:
        indices = torch.randperm(total_open_set)[:subset_size]
        open_set_subset = Subset(open_set_subset, indices)
    else:
        print("开放集子集大小大于或等于原始开放集大小，使用完整开放集。")

    print(f"开放集总大小: {total_open_set}, 选择子集大小: {len(open_set_subset)}")

    # 进一步划分已知集为训练集和测试集（9:1）
    num_known = len(known_subset)
    indices = list(range(num_known))
    split = int(np.floor(0.1 * num_known))
    np.random.shuffle(indices)
    test_indices = indices[:split]
    train_indices = indices[split:]

    train_subset = Subset(known_subset, train_indices)
    test_subset = Subset(known_subset, test_indices)

    # 创建 DataLoader，设置 num_workers=0 以避免 Windows 上的多进程问题
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 修改为0
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 修改为0
        pin_memory=True if device.type == 'cuda' else False
    )
    open_set_loader = DataLoader(
        open_set_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 修改为0
        pin_memory=True if device.type == 'cuda' else False
    )

    # 打印样本数量
    print(f"已知训练集样本数量: {len(train_subset)}")
    print(f"已知测试集样本数量: {len(test_subset)}")
    print(f"开放集样本数量: {len(open_set_subset)}")

    return train_loader, test_loader, open_set_loader

