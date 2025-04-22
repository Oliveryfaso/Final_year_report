##train_cifar

import torch
from torch import optim
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import os
from model_convgan_cifar import get_optimizer
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import pickle
# 新增以下导入语句
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from model_r3gan import R3Generator, R3Discriminator


def train_kway_network(model, data_loader, generator, device, epochs=5, learning_rate=0.0001, activation_save_path='./models/activations.pkl', loss_save_path='./models/kway_losses.pkl'):
    model.train()
    generator.eval()  # 确保生成器处于评估模式
    class_weights = torch.ones(model.num_classes).to(device)
    class_weights[0] = 1.0  # 假设 0 代表未知类别
    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_metric = TripletMarginLoss(margin=1.0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    activations = defaultdict(list)

    loss_history = {
        'total_loss': [],
        'cls_loss': [],
        'metric_loss': [],
        'accuracy': []
    }

    nz = generator.nz  # 噪声向量维度

    for epoch in range(epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_metric_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch [{epoch + 1}/{epochs}]', unit='batch')

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            # 生成假数据
            batch_size = images.size(0)
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 'Unknown' 类别标签为 0

            with torch.no_grad():  # 不需要计算生成器的梯度
                # fake_images = generator(noise)
                fake_images = generator(noise, fake_labels)

            # 合并真实数据和假数据
            combined_images = torch.cat([images, fake_images], dim=0)
            combined_labels = torch.cat([labels, fake_labels], dim=0)

            # 前向传播
            features, logits = model(combined_images)
            cls_loss = criterion_cls(logits, combined_labels)

            # Triplet 损失
            label_to_indices = defaultdict(list)
            for idx, lab in enumerate(combined_labels):
                lab = lab.item()
                label_to_indices[lab].append(idx)

            triplet_anchors = []
            triplet_positives = []
            triplet_negatives = []

            unique_labels = list(label_to_indices.keys())
            if len(unique_labels) > 1:
                for lab, idx_list in label_to_indices.items():
                    if len(idx_list) < 2 or lab == 0:
                        continue  # 忽略未知类别或样本数不足的类别
                    anchor_idx, positive_idx = np.random.choice(idx_list, 2, replace=False)
                    neg_candidates = [i for u_lab, u_idx_list in label_to_indices.items() if u_lab != lab and u_lab != 0 for i in u_idx_list]
                    if len(neg_candidates) == 0:
                        continue
                    negative_idx = np.random.choice(neg_candidates)

                    triplet_anchors.append(anchor_idx)
                    triplet_positives.append(positive_idx)
                    triplet_negatives.append(negative_idx)

            if len(triplet_anchors) == 0:
                metric_loss = torch.tensor(0.0, device=device)
            else:
                anchor_feat = features[triplet_anchors]
                positive_feat = features[triplet_positives]
                negative_feat = features[triplet_negatives]
                metric_loss = criterion_metric(anchor_feat, positive_feat, negative_feat)

            loss = cls_loss + 0.1 * metric_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_metric_loss += metric_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += combined_labels.size(0)
            correct += (predicted == combined_labels).sum().item()

            progress_bar.set_postfix({
                'Loss': running_loss / (progress_bar.n + 1),
                'Cls Loss': running_cls_loss / (progress_bar.n + 1),
                'Metric Loss': running_metric_loss / (progress_bar.n + 1),
                'Accuracy': 100 * correct / total
            })

            for feat, lab in zip(features.detach(), combined_labels.detach()):
                activations[lab.item()].append(feat.cpu().numpy())

        epoch_loss = running_loss / len(data_loader)
        epoch_cls_loss = running_cls_loss / len(data_loader)
        epoch_metric_loss = running_metric_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}] Total Loss: {epoch_loss:.4f} | Cls Loss: {epoch_cls_loss:.4f} | Metric Loss: {epoch_metric_loss:.4f} | Accuracy: {epoch_acc:.2f}%')

        loss_history['total_loss'].append(epoch_loss)
        loss_history['cls_loss'].append(epoch_cls_loss)
        loss_history['metric_loss'].append(epoch_metric_loss)
        loss_history['accuracy'].append(epoch_acc)

    print('K-way 网络(含度量学习)训练完成。')

    with open(activation_save_path, 'wb') as f:
        pickle.dump(activations, f)
    print(f"激活数据已保存到 {activation_save_path}")

    plot_kway_losses(loss_history, save_dir='loss_curves_kway')

def plot_kway_losses(loss_history, save_dir='loss_curves_kway'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(loss_history['total_loss']) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_history['total_loss'], 'r-', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Total Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss_history['cls_loss'], 'b-', label='Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Classification Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, loss_history['metric_loss'], 'g-', label='Metric Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Metric Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, loss_history['accuracy'], 'm-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('K-way Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kway_loss_curve.png'))
    plt.close()
    print(f"K-way 网络的损失曲线已保存到 {os.path.join(save_dir, 'kway_loss_curve.png')}")

def save_fake_images(epoch, generator, device, nz=100, num_images=16, save_dir='generated_images'):
    generator.eval()
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    labels = torch.randint(1, 10, (num_images,), device=device)
    with torch.no_grad():
        fake_images = generator(noise, labels).cpu()
    grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
    os.makedirs(save_dir, exist_ok=True)
    torchvision.utils.save_image(grid, os.path.join(save_dir, f'epoch_{epoch + 1}.png'))
    generator.train()

def plot_losses(d_losses, g_losses, loss_dir='loss_curves_gan'):
    os.makedirs(loss_dir, exist_ok=True)
    plt.figure()
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(loss_dir, 'loss_curve.png'))
    plt.close()

def train_gan(generator, discriminator, data_loader, device, pixelcnn_pp, epochs=90, learning_rate=2e-4, nz=100,
              save_dir='generated_images', loss_dir='loss_curves_gan', epsilon=1e-4):
    from model_convgan_cifar import get_optimizer
    from pixelcnn_cifar import discretized_mix_logistic_loss

    torch.autograd.set_detect_anomaly(True)
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_feature = torch.nn.MSELoss()

    optimizer_g = get_optimizer(generator, lr=learning_rate, beta1=0.5, beta2=0.999)
    optimizer_d = get_optimizer(discriminator, lr=0.5*learning_rate, beta1=0.5, beta2=0.999)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.5)

    d_losses, g_losses = [], []

    # CIFAR10 mean/std
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        progress_bar = tqdm(data_loader, desc=f'Epoch [{epoch + 1}/{epochs}]', unit='batch')

        current_epoch = epoch + 1
        if current_epoch <= 20:
            use_feature_matching = False
            use_diversity = False
            use_low_density = False
            weight_gan = 1.0
            weight_class = 1.0
            weight_feature_matching = 0.0
            weight_diversity = 0.1
            weight_low_density = 0.0
        elif current_epoch <= 40:
            use_feature_matching = True
            use_diversity = True
            use_low_density = False
            weight_gan = 1.0
            weight_class = 1.0
            weight_feature_matching = 0.5
            weight_diversity = 0.1
            weight_low_density = 0.0
        else:
            use_feature_matching = True
            use_diversity = True
            use_low_density = True
            weight_gan = 1.0
            weight_class = 1.0
            weight_feature_matching = 0.5
            weight_diversity = 0.1
            weight_low_density = 0.5

        # use_feature_matching = True
        # use_diversity = True
        # use_low_density = True
        # weight_gan = 1.0
        # weight_class = 1.0
        # weight_feature_matching = 1.0
        # weight_diversity = 0.1
        # weight_low_density = 1.0

        for i, (real_images, labels) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device, dtype=torch.long)

            if labels.max() >= 10 or labels.min() < 0:
                continue

            # 隔批更新判别器：仅在i为偶数时更新判别器
            if i % 2 == 0:
                # 判别器训练
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images_batch = generator(noise, labels).detach()
                outputs_real, _, _ = discriminator(real_images)
                outputs_fake, _, _ = discriminator(fake_images_batch)

                d_loss_real = torch.mean(F.relu(1.0 - outputs_real))
                d_loss_fake = torch.mean(F.relu(1.0 + outputs_fake))
                d_loss = d_loss_real + d_loss_fake

                discriminator.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
            else:
                # 如果当前批次不更新判别器，d_loss就用上一次（偶数批次）的损失或设置为0
                d_loss = torch.tensor(0.0, device=device)

            # 生成器训练
            generator.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(noise, labels)

            outputs_fake_gen, class_output, fake_features = discriminator(fake_images)

            g_loss_gan = -torch.mean(outputs_fake_gen)
            g_loss_class = criterion_class(class_output, labels)

            g_loss_feature_matching = 0.0
            g_loss_diversity = 0.0
            g_loss_low_density = 0.0

            if use_feature_matching or use_low_density or use_diversity:
                with torch.no_grad():
                    _, _, real_features = discriminator(real_images)

            if use_feature_matching:
                g_loss_feature_matching = criterion_feature(real_features.mean(dim=0), fake_features.mean(dim=0))

            if use_diversity:
                g_loss_diversity = torch.mean((fake_images - fake_images.mean(dim=0)) ** 2)

            if use_low_density:
                # 获取判别器中间层特征
                _, _, features = discriminator(fake_images)
                
                # 计算能量（基于logsumexp）
                energy = torch.logsumexp(features, dim=1)
                
                # 设定动态阈值（例如使用特征均值）
                threshold = features.mean()
                
                # 能量正则化损失
                g_loss_low_density = F.relu(energy - threshold).mean()
            else:
                g_loss_low_density = 0.0

            g_loss = (weight_gan * g_loss_gan +
                      weight_class * g_loss_class +
                      weight_feature_matching * g_loss_feature_matching +
                      weight_diversity * g_loss_diversity +
                      weight_low_density * g_loss_low_density)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            progress_bar.set_postfix({'Loss D': d_loss.item(), 'Loss G': g_loss.item()})

            if (i + 1) % 100 == 0:
                save_fake_images(epoch, generator, device, nz=nz, save_dir=save_dir)

        g_losses.append(g_loss_epoch / len(data_loader))
        d_losses.append(d_loss_epoch*2 / len(data_loader))

        print(f"Epoch [{epoch + 1}/{epochs}] Loss D: {d_losses[-1]:.4f}, Loss G: {g_losses[-1]:.4f}")

        scheduler_g.step()
        scheduler_d.step()

    plot_losses(d_losses, g_losses, loss_dir=loss_dir)

    print("\n=== Generator Losses per Epoch ===")
    for idx, loss in enumerate(g_losses, 1):
        print(f"Epoch {idx}: {loss:.4f}")

    print("\n=== Discriminator Losses per Epoch ===")
    for idx, loss in enumerate(d_losses, 1):
        print(f"Epoch {idx}: {loss:.4f}")

    torch.save(generator.state_dict(), './models/generator.pth')
    torch.save(discriminator.state_dict(), './models/discriminator.pth')
    print("GAN模型训练完成，模型已保存。")





def relativistic_loss(real_pred, fake_pred):
    """RpGAN损失核心实现[2,4](@ref)"""
    real_logits = real_pred - fake_pred.mean(0, keepdim=True)
    fake_logits = fake_pred - real_pred.mean(0, keepdim=True)
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

def gradient_penalty(D, real, fake):
    """数值稳定版梯度惩罚"""
    device = real.device
    batch_size = real.size(0)
    
    # 生成插值样本（添加抖动噪声）
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha + 0.1 * torch.randn_like(alpha)  # 添加10%噪声
    alpha = torch.clamp(alpha, 0.0, 1.0)
    
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    interpolates = torch.clamp(interpolates, -1.5, 1.5)  # 放宽截断范围
    
    # 混合精度前向
    with autocast():
        d_interpolates = D(interpolates)
        d_interpolates = torch.clamp(d_interpolates, -5.0, 5.0)  # 输出截断
    
    # 梯度计算（添加梯度截断）
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )[0]
    
    # 梯度归一化（防止除零）
    gradients = gradients.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-8)
    penalty = torch.mean((grad_norm - 1.0)**2)
    
    return penalty

# 在原有代码最后添加以下内容

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_r3gan(generator, discriminator, train_loader, device, epochs=200):
    """相对少量改动后的 train_r3gan, 修复梯度裁剪字典和多次混合精度前向问题."""
    scaler = GradScaler()

    # 优化器
    opt_g = torch.optim.RAdam(
        generator.parameters(), 
        lr=1e-4, 
        betas=(0.0, 0.999),
        weight_decay=1e-4
    )
    opt_d = torch.optim.RAdam(
        discriminator.parameters(),
        lr=4e-4,
        betas=(0.0, 0.999),
        weight_decay=1e-4
    )
    
    # 学习率预热（前10个epoch）
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        opt_g, 
        lr_lambda=lambda ep: min(1.0, (ep+1)/10.0)
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        opt_d,
        lr_lambda=lambda ep: min(1.0, (ep+1)/10.0)
    )

    # 分层梯度裁剪的最大范数，根据名称简单区分
    max_grad_norm = {
        'conv': 1.0,
        'fc': 0.5,
        'default': 1.0
    }

    for epoch in range(epochs):
        for real, _ in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            real = real.to(device)

            # ========== 1. 判别器训练 ========== #
            opt_d.zero_grad()
            with autocast():
                # 生成器产生假样本 (z 缩小到 0.1 以减少初始爆炸)
                z = torch.randn(real.size(0), 128, device=device) * 0.1
                fake = generator(z)

                # 判别器对真实/假样本的预测
                real_pred = discriminator(real)
                fake_pred = discriminator(fake.detach())

                # WGAN-GP: 这里10.0是梯度惩罚系数
                gp = 10.0 * gradient_penalty(discriminator, real, fake)
                # WGAN 判别器损失 (相对略改成: fake_pred.mean() - real_pred.mean() + gp)
                loss_d = fake_pred.mean() - real_pred.mean() + gp

            # 判别器 backward + 分层裁剪 + step
            scaler.scale(loss_d).backward()
            for name, param in discriminator.named_parameters():
                if param.grad is not None:
                    # 只要名字里包含 'fc' 就按照 fc 的范数限制，否则如果包含 'conv' 就是卷积
                    if 'fc' in name:
                        max_norm = max_grad_norm['fc']
                    elif 'conv' in name:
                        max_norm = max_grad_norm['conv']
                    else:
                        max_norm = max_grad_norm['default']
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            scaler.step(opt_d)

            # ========== 2. 生成器训练 ========== #
            opt_g.zero_grad()
            with autocast():
                # 重新生成假样本 (也可以复用上面那批 z, 影响不大)
                fake = generator(z)
                fake_pred_g = discriminator(fake)
                # 生成器损失 WGAN: 希望 fake_pred_g 越大越好 → -mean
                loss_g = -fake_pred_g.mean()

            # 生成器 backward + 分层裁剪 + step
            scaler.scale(loss_g).backward()
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    # 这里原文里给 l1 层和其他层设了不同 max_norm
                    if 'l1' in name:
                        max_norm = 0.3  # 初始化层更严格
                    elif 'conv' in name:
                        max_norm = 1.0
                    else:
                        max_norm = 0.5
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            scaler.step(opt_g)
            scaler.update()

            # 简单数值检查：若出现 NaN 立即中止
            if (torch.isnan(loss_d) or torch.isnan(loss_g) or 
                torch.isinf(loss_d) or torch.isinf(loss_g)):
                print(f"Epoch {epoch+1} 检测到 NaN/Inf, 终止训练.")
                return
        
        # 学习率预热
        scheduler_g.step()
        scheduler_d.step()

    print("R3GAN 训练完成！")

    # ============ 新增：cWGAN-GP的训练示例 =================
def gradient_penalty_cgan(discriminator, real_images, fake_images, labels, device, gp_lambda=10.0):
    """
    计算 cWGAN-GP 的梯度惩罚，考虑条件 (labels)。
    """
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_images)

    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad_(True)

    disc_interpolates, _, _ = discriminator(interpolates)

    grads = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gradient_penalty = gp_lambda * ((grad_norm - 1.0)**2).mean()
    return gradient_penalty


def train_cwgan_gp(
    generator, discriminator, data_loader, device,
    nz=100, gp_lambda=10.0, n_critic=5,  # n_critic=5常见
    epochs=50, lr_g=1e-4, lr_d=4e-4,
):
    """
    基于条件WGAN-GP的训练示例。
    """
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))

    generator.train()
    discriminator.train()

    step = 0
    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch+1}/{epochs}]")):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # ==== (1) 更新判别器 ====
            for _ in range(n_critic):
                z = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_imgs = generator(z, labels)

                d_real, _, _ = discriminator(real_imgs)
                d_fake, _, _ = discriminator(fake_imgs.detach())

                d_loss_real = -d_real.mean()
                d_loss_fake = d_fake.mean()

                gp = gradient_penalty_cgan(discriminator, real_imgs, fake_imgs, labels, device, gp_lambda=gp_lambda)
                d_loss = d_loss_real + d_loss_fake + gp

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # ==== (2) 更新生成器 ====
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            gen_imgs = generator(z, labels)
            g_score, _, _ = discriminator(gen_imgs)

            g_loss = -g_score.mean()

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            step += 1
            
            if i % 50 == 0:
                tqdm.write(f"Epoch [{epoch+1}/{epochs}] Step {i}: D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

        print(f"=> Epoch {epoch+1} done: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    # 训练结束后可保存
    torch.save(generator.state_dict(), "./models/generator_cwgan.pth")
    torch.save(discriminator.state_dict(), "./models/discriminator_cwgan.pth")
    print("cWGAN-GP 训练完成并保存模型！")
