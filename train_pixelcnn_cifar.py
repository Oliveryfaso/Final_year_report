# train_pixelcnn_cifar.py

import torch
from torch import optim
from torch.utils.data import DataLoader
from pixelcnn_cifar import PixelCNN, discretized_mix_logistic_loss
from data_loader_cifar import load_cifar10_data
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import numpy as np

def evaluate_pixelcnn(model, data_loader, device, mean, std):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            images = images * std + mean
            images = torch.clamp(images, 0.0, 1.0)
            output = model(images)
            loss = discretized_mix_logistic_loss(images, output, sum_all=True)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def plot_px_distribution(real_px, fake_px, save_path='loss_curves_gan/px_distribution.png'):
    plt.figure(figsize=(10,6))
    sns.kdeplot(real_px, label='Real Data p_x', shade=True)
    sns.kdeplot(fake_px, label='Fake Data p_x', shade=True)
    plt.xlabel('p_x')
    plt.ylabel('Density')
    plt.title('PixelCNN p_x Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"p_x 分布已保存到 {save_path}")

def train_pixelcnn_cifar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size = 64
    epochs = 120  # 增加训练轮数
    learning_rate = 5e-4  # 降低学习率
    save_model_path = './models/pixelcnn_pp.pth'
    save_loss_curve_path = 'loss_curves_gan/training_loss_curve_cifar.png'

    # CIFAR10官方mean和std
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

    # 加载数据集
    train_loader, test_loader, open_set_loader = load_cifar10_data(
        batch_size=batch_size,
        known_labels=[1,2,3,4,5,6,7,8,9],
        unknown_label=0,
        save_dir='./data/cifar10',
        device=device
    )

    print("初始化 PixelCNN 模型...")
    model = PixelCNN(
        nr_filters=160,  # 增加滤波器数量
        nr_resnet=5,
        nr_logistic_mix=10,
        disable_third=False,
        dropout_p=0.3,  # 增加 dropout 比例
        n_channel=3,
        image_wh=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 使用学习率调度器，每20个epoch下降一次，gamma=0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()

    loss_history = []
    px_history = []

    print("开始训练 PixelCNN++ 模型...")
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch [{epoch + 1}/{epochs}]", unit="batch",
                            total=len(train_loader))
        for i, (images, _) in progress_bar:
            images = images.to(device)

            # 逆标准化: x = x_norm * std + mean
            # 确保数据在[0,1]之间
            images = images * std + mean
            images = torch.clamp(images, 0.0, 1.0)

            optimizer.zero_grad()
            output = model(images)
            loss = discretized_mix_logistic_loss(images, output, sum_all=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix(loss=avg_loss)

        scheduler.step()
        epoch_avg_loss = running_loss / len(train_loader)
        loss_history.append(epoch_avg_loss)

        # 计算验证损失
        val_loss = evaluate_pixelcnn(model, test_loader, device, mean, std)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 计算 p_x 分布
        with torch.no_grad():
            # 使用训练集的一部分计算 p_x
            sample_images = next(iter(train_loader))[0].to(device)
            sample_images = sample_images * std + mean
            sample_images = torch.clamp(sample_images, 0.0, 1.0)
            output = model(sample_images)
            loss = discretized_mix_logistic_loss(sample_images, output, sum_all=True, return_per_image=True)
            px = torch.exp(-loss).cpu().numpy()
            px_history.extend(px)

        # 每10个epoch绘制一次分布图
        if (epoch + 1) % 10 == 0:
            # 生成假数据的 p_x 需要通过生成器生成并通过 PixelCNN 计算
            # 这里假设你有一个预训练的生成器，可以加载并生成样本
            # 下面是一个示例：
            # generator = ...  # 加载生成器
            # generator.eval()
            # noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # fake_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            # with torch.no_grad():
            #     fake_images = generator(noise, fake_labels)
            #     fake_images = fake_images * std + mean
            #     fake_images = torch.clamp(fake_images, 0.0, 1.0)
            #     output_fake = model(fake_images)
            #     loss_fake = discretized_mix_logistic_loss(fake_images, output_fake, sum_all=True, return_per_image=True)
            #     fake_px = torch.exp(-loss_fake).cpu().numpy()
            #     plot_px_distribution(px_history, fake_px, save_path=f'loss_curves_gan/px_distribution_epoch_{epoch + 1}.png')
            #     px_history = []
            pass  # 根据具体实现添加生成假数据的 p_x 计算

    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"PixelCNN++ 模型训练完成并已保存到 {save_model_path}")

    os.makedirs('loss_curves_gan', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PixelCNN++ Training Loss Curve on CIFAR-10')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_loss_curve_path)
    plt.close()
    print(f"训练过程损失曲线已保存为 {save_loss_curve_path}")

if __name__ == "__main__":
    train_pixelcnn_cifar()
