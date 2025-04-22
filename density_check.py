# density_check.py

import os
import torch
import torchvision
import torchvision.io as io
import torchvision.transforms as T
from pixelcnn_MNIST import PixelCNN, discretized_mix_logistic_loss_c1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from data_loader_MNIST import SignMNISTDataset
from model_convgan_MNIST import Generator
import random
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # 新增

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数配置
nr_filters = 64
nr_resnet = 5
nr_logistic_mix = 10
n_channel = 1  # MNIST 是单通道图像
image_wh = 28
nrow = 4
epsilon = 1e-4
nz = 100

# 分组：
# Group 1: Epoch 1-40
# Group 2: Epoch 41-80
# Group 3: Epoch 81-120
group1 = range(1, 41)
group2 = range(41, 81)
group3 = range(81, 121)

generated_dir = 'generated_images'

# 创建保存结果的目录
os.makedirs('./density_results', exist_ok=True)

# 加载 PixelCNN++
pixelcnn_pp = PixelCNN(
    nr_filters=nr_filters,
    nr_resnet=nr_resnet,
    nr_logistic_mix=nr_logistic_mix,
    disable_third=False,
    dropout_p=0.5,
    n_channel=n_channel,
    image_wh=image_wh
).to(device)

pixelcnn_pp_path = './models/pixelcnn_pp.pth'
if os.path.exists(pixelcnn_pp_path):
    pixelcnn_pp.load_state_dict(torch.load(pixelcnn_pp_path, map_location=device))
    pixelcnn_pp.eval()
    print("PixelCNN++ 模型已加载。")
else:
    raise FileNotFoundError(f"PixelCNN++ 模型文件未找到: {pixelcnn_pp_path}")

# 加载生成器
generator = Generator(nz=nz, ngf=64, nc=1, num_classes=24).to(device)
generator_path = './models/generator.pth'
if os.path.exists(generator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print("生成器已加载。")
else:
    raise FileNotFoundError(f"生成器模型文件未找到: {generator_path}")

def load_epoch_images(epoch):
    image_path = os.path.join(generated_dir, f'epoch_{epoch}.png')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件 {image_path} 未找到。")

    img = io.read_image(image_path)  # [C,H,W]
    img = img.float() / 255.0

    # 转单通道
    if img.size(0) == 3:
        img = img[0:1]

    expected_size = image_wh * nrow
    H, W = img.size(1), img.size(2)
    if (H != expected_size or W != expected_size):
        # 若有padding
        if H == expected_size + 10 and W == expected_size + 10:
            y_start = 2
            x_start = 2
            img = img[:, y_start:y_start + expected_size, x_start:x_start + expected_size]
        else:
            raise ValueError(f"图像大小不匹配，期望: {expected_size}x{expected_size}, 实际: {H}x{W}")

    sub_images = []
    for row in range(nrow):
        for col in range(nrow):
            y_start = row * image_wh
            x_start = col * image_wh
            sub_img = img[:, y_start:y_start + image_wh, x_start:x_start + image_wh]
            sub_images.append(sub_img)
    return sub_images

# 把1-120分成三组不同epoch的生成图像全部载入
all_fake_subimages = []
fake_groups = []  # 保存每张假图像的组别 (1,2,3)
for ep in range(1, 121):
    try:
        ep_subimgs = load_epoch_images(ep)
    except Exception as e:
        print(f"加载 epoch {ep} 的图像时发生错误: {e}")
        continue
    # 确定ep所属组别
    if ep in group1:
        group_id = 1
    elif ep in group2:
        group_id = 2
    else:
        group_id = 3

    all_fake_subimages.extend(ep_subimgs)
    fake_groups.extend([group_id] * len(ep_subimgs))

# 检查是否有生成图像被加载
if len(all_fake_subimages) == 0:
    raise ValueError("未加载到任何生成的图像子图。请检查生成图像目录和文件。")

fake_images = torch.stack(all_fake_subimages, dim=0).to(device)  # [B,1,28,28]
B_fake = fake_images.size(0)
print(f"总共载入 {B_fake} 张生成图像子图。")

# 从真实数据中抽取相同数量的图像
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))  # 根据训练时的归一化方式调整
])

dataset = SignMNISTDataset(csv_file='./data/handfigure/sign_mnist_combined.csv', transform=transform)
if B_fake > len(dataset):
    raise ValueError(f"请求的真实图像数量 {B_fake} 超过了数据集的大小 {len(dataset)}。")
indices = random.sample(range(len(dataset)), B_fake)
real_subset = torch.utils.data.Subset(dataset, indices)
real_loader = DataLoader(real_subset, batch_size=B_fake, shuffle=False, num_workers=0)
real_images, _ = next(iter(real_loader))
real_images = real_images.to(device)

print(f"已从真实数据中抽取 {real_images.size(0)} 张图像用于对比。")

def compute_px(model, images):
    with torch.no_grad():
        l = model(images)
        loss = discretized_mix_logistic_loss_c1(images, l, sum_all=True)
        B = images.size(0)
        H = images.size(2)
        W = images.size(3)
        total_log_p = -loss * (B * H * W)
        log_p_x_per_image = total_log_p / B
        p_x_per_image = torch.exp(log_p_x_per_image.unsqueeze(-1).unsqueeze(-1))
    return p_x_per_image.mean().item(), log_p_x_per_image.mean().item()

p_real, _ = compute_px(pixelcnn_pp, real_images)
p_fake, _ = compute_px(pixelcnn_pp, fake_images)

print(f"真实图像平均 p(x): {p_real:.8f}, 生成图像平均 p(x): {p_fake:.8f}")
if p_fake < epsilon:
    print("平均来看，这些生成的图像位于低密度区域。")
else:
    print("平均来看，这些生成的图像不在低密度区域。")

def images_to_features(imgs):
    return imgs.view(imgs.size(0), -1).cpu().numpy()

real_features = images_to_features(real_images)
fake_features = images_to_features(fake_images)

# 为简单起见，使用平均 p(x) 作为所有 fake 和 real 的近似 p 值
real_p_values = np.full((real_features.shape[0],), p_real)
fake_p_values = np.full((fake_features.shape[0],), p_fake)

# 创建标签： real=0, fake group1=1, group2=2, group3=3
real_labels = np.zeros(real_features.shape[0], dtype=int)
fake_labels = np.array(fake_groups, dtype=int)

all_features = np.concatenate([real_features, fake_features], axis=0)
all_p_values = np.concatenate([real_p_values, fake_p_values], axis=0)
all_groups = np.concatenate([real_labels, fake_labels], axis=0)  # 0 for real, 1/2/3 for fake

# PCA降维
pca = PCA(n_components=3)  # 修改为3以支持3D降维
proj = pca.fit_transform(all_features)

real_count = real_features.shape[0]
real_proj = proj[:real_count]
fake_proj = proj[real_count:]

# 根据不同的 PCA 组合进行分割
# 2D PCA
pca_2d = proj[:, :2]
real_proj_2d = pca_2d[:real_count]
fake_proj_2d = pca_2d[real_count:]

fake_proj_g1_2d = fake_proj_2d[all_groups[real_count:] == 1]
fake_proj_g2_2d = fake_proj_2d[all_groups[real_count:] == 2]
fake_proj_g3_2d = fake_proj_2d[all_groups[real_count:] == 3]

# 3D PCA
real_proj_3d = proj[:real_count]
fake_proj_3d = proj[real_count:]

fake_proj_g1_3d = fake_proj_3d[all_groups[real_count:] == 1]
fake_proj_g2_3d = fake_proj_3d[all_groups[real_count:] == 2]
fake_proj_g3_3d = fake_proj_3d[all_groups[real_count:] == 3]

# 绘制原有的分组可视化图 (2D PCA)
plt.figure(figsize=(10, 8))
plt.scatter(real_proj_2d[:, 0], real_proj_2d[:, 1],
            c='blue', alpha=0.6, s=20, marker='o', label='Real')
plt.scatter(fake_proj_g1_2d[:, 0], fake_proj_g1_2d[:, 1],
            c='red', alpha=0.6, s=20, marker='s', label='Fake Group 1 (Ep1-40)')
plt.scatter(fake_proj_g2_2d[:, 0], fake_proj_g2_2d[:, 1],
            c='green', alpha=0.6, s=20, marker='^', label='Fake Group 2 (Ep41-80)')
plt.scatter(fake_proj_g3_2d[:, 0], fake_proj_g3_2d[:, 1],
            c='orange', alpha=0.6, s=20, marker='d', label='Fake Group 3 (Ep81-120)')

plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC2', fontsize=14)
plt.title('Real vs Fake Images by Epoch Groups in 2D (PCA)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./density_results/real_fake_pca_pc1_pc2.png')
plt.close()
print("已保存 PC1 vs PC2 可视化图像到 real_fake_pca_pc1_pc2.png")

# 绘制 PC1 vs PC3
plt.figure(figsize=(10, 8))
plt.scatter(real_proj[:, 0], real_proj[:, 2],
            c='blue', alpha=0.6, s=20, marker='o', label='Real')
plt.scatter(fake_proj_g1_2d[:, 0], fake_proj_g1_3d[:, 2],
            c='red', alpha=0.6, s=20, marker='s', label='Fake Group 1 (Ep1-40)')
plt.scatter(fake_proj_g2_2d[:, 0], fake_proj_g2_3d[:, 2],
            c='green', alpha=0.6, s=20, marker='^', label='Fake Group 2 (Ep41-80)')
plt.scatter(fake_proj_g3_2d[:, 0], fake_proj_g3_3d[:, 2],
            c='orange', alpha=0.6, s=20, marker='d', label='Fake Group 3 (Ep81-120)')

plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC3', fontsize=14)
plt.title('PC1 vs PC3 of Real and Fake Images (PCA)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./density_results/real_fake_pca_pc1_pc3.png')
plt.close()
print("已保存 PC1 vs PC3 可视化图像到 real_fake_pca_pc1_pc3.png")

# 绘制 PC2 vs PC3
plt.figure(figsize=(10, 8))
plt.scatter(real_proj[:, 1], real_proj[:, 2],
            c='blue', alpha=0.6, s=20, marker='o', label='Real')
plt.scatter(fake_proj_g1_2d[:, 1], fake_proj_g1_3d[:, 2],
            c='red', alpha=0.6, s=20, marker='s', label='Fake Group 1 (Ep1-40)')
plt.scatter(fake_proj_g2_2d[:, 1], fake_proj_g2_3d[:, 2],
            c='green', alpha=0.6, s=20, marker='^', label='Fake Group 2 (Ep41-80)')
plt.scatter(fake_proj_g3_2d[:, 1], fake_proj_g3_3d[:, 2],
            c='orange', alpha=0.6, s=20, marker='d', label='Fake Group 3 (Ep81-120)')

plt.xlabel('PC2', fontsize=14)
plt.ylabel('PC3', fontsize=14)
plt.title('PC2 vs PC3 of Real and Fake Images (PCA)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./density_results/real_fake_pca_pc2_pc3.png')
plt.close()
print("已保存 PC2 vs PC3 可视化图像到 real_fake_pca_pc2_pc3.png")

# 在此基础上添加 K-means 聚类分析
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(proj[:, :2])  # 使用前两主成分进行聚类
cluster_labels = kmeans.labels_

# 将数据再次拆分：前 real_count 为真实数据，后面为生成数据
real_clusters = cluster_labels[:real_count]
fake_clusters = cluster_labels[real_count:]

# 按组划分的 fake 数据的聚类标签
fake_clusters_g1 = fake_clusters[all_groups[real_count:] == 1]
fake_clusters_g2 = fake_clusters[all_groups[real_count:] == 2]
fake_clusters_g3 = fake_clusters[all_groups[real_count:] == 3]

# 绘制 K-means 聚类可视化图
cmap = plt.get_cmap('tab10')
plt.figure(figsize=(10, 8))

# 创建图例标识（无需传数据点）
plt.scatter([], [], c='k', marker='o', edgecolors='black', s=50, label='Real Data')
plt.scatter([], [], c='k', marker='s', edgecolors='none', s=50, label='Fake Group 1 (Ep1-40)')
plt.scatter([], [], c='k', marker='^', edgecolors='none', s=50, label='Fake Group 2 (Ep41-80)')
plt.scatter([], [], c='k', marker='d', edgecolors='none', s=50, label='Fake Group 3 (Ep81-120)')

# 绘制真实数据，根据簇标签着色，使用 'o' 圆点且黑边框
for c in range(num_clusters):
    mask = (real_clusters == c)
    plt.scatter(real_proj_2d[mask, 0], real_proj_2d[mask, 1],
                c=[cmap(c)], alpha=0.6, s=20, marker='o', edgecolors='black')

# 绘制生成数据点：根据簇标签着色，并根据组别使用不同的 marker
for c in range(num_clusters):
    # Group1
    c_mask_g1 = (fake_clusters_g1 == c)
    if np.any(c_mask_g1):
        plt.scatter(fake_proj_g1_2d[c_mask_g1, 0], fake_proj_g1_2d[c_mask_g1, 1],
                    c=[cmap(c)], alpha=0.6, s=20, marker='s', edgecolors='none')

    # Group2
    c_mask_g2 = (fake_clusters_g2 == c)
    if np.any(c_mask_g2):
        plt.scatter(fake_proj_g2_2d[c_mask_g2, 0], fake_proj_g2_2d[c_mask_g2, 1],
                    c=[cmap(c)], alpha=0.6, s=20, marker='^', edgecolors='none')

    # Group3
    c_mask_g3 = (fake_clusters_g3 == c)
    if np.any(c_mask_g3):
        plt.scatter(fake_proj_g3_2d[c_mask_g3, 0], fake_proj_g3_2d[c_mask_g3, 1],
                    c=[cmap(c)], alpha=0.6, s=20, marker='d', edgecolors='none')

plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC2', fontsize=14)
plt.title('Clusters (K-means) of Real vs Fake in 2D (PCA)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(fontsize=12)
plt.savefig('./density_results/real_fake_pca_with_kmeans.png')
plt.close()
print("已保存K-means聚类可视化图像到 real_fake_pca_with_kmeans.png")

# 绘制 3D PCA 图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(real_proj_3d[:, 0], real_proj_3d[:, 1], real_proj_3d[:, 2],
           c='blue', alpha=0.6, s=20, marker='o', label='Real')

ax.scatter(fake_proj_g1_3d[:, 0], fake_proj_g1_3d[:, 1], fake_proj_g1_3d[:, 2],
           c='red', alpha=0.6, s=20, marker='s', label='Fake Group 1 (Ep1-40)')

ax.scatter(fake_proj_g2_3d[:, 0], fake_proj_g2_3d[:, 1], fake_proj_g2_3d[:, 2],
           c='green', alpha=0.6, s=20, marker='^', label='Fake Group 2 (Ep41-80)')

ax.scatter(fake_proj_g3_3d[:, 0], fake_proj_g3_3d[:, 1], fake_proj_g3_3d[:, 2],
           c='orange', alpha=0.6, s=20, marker='d', label='Fake Group 3 (Ep81-120)')

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
ax.set_title('Real vs Fake Images by Epoch Groups in 3D (PCA)', fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('./density_results/real_fake_pca_epoch_groups_3d.png')
plt.close()
print("已保存3D PCA可视化图像到 real_fake_pca_epoch_groups_3d.png")
