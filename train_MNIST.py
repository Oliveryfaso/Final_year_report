# train_MNIST.py


import torch
from torch import optim
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import os
from model_convgan_MNIST import get_optimizer
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import pickle
from collections import defaultdict
import torchvision.utils as vutils


def train_kway_network(model, data_loader, generator, device, epochs=5, learning_rate=0.0001, activation_save_path='./models/activations.pkl', loss_save_path='./models/kway_losses.pkl'):
    """
    训练 K-way 网络，并记录每个epoch的损失变化。
    """
    model.train()
    generator.eval()  # 确保生成器处于评估模式，不更新其参数
    class_weights = torch.ones(model.num_classes).to(device)
    class_weights[0] = 1.2  # 增加 'Unknown' 类的权重
    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_metric = TripletMarginLoss(margin=1.0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化用于存储激活数据的字典
    activations = defaultdict(list)

    # 初始化用于记录损失的列表
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

            # ========== 修复假数据生成逻辑 ==========
            # ========== 生成假数据 ==========
            batch_size = images.size(0)
            
            # 生成噪声（保持2D维度）
            noise = torch.randn(batch_size, generator.nz, device=device)
            
            # 生成假标签（类别索引，整数类型）
            fake_labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 'Unknown'类别为0

            # 生成假图像（传入类别索引，而非one-hot）
            with torch.no_grad():
                fake_images = generator(noise, fake_labels)  # 关键修复点：直接传入整数标签

            # 合并真实数据和假数据
            combined_images = torch.cat([images, fake_images], dim=0)
            combined_labels = torch.cat([labels, fake_labels], dim=0)
            # ========== 后续代码保持不变 ==========
            
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
                        continue  # 忽略 'Unknown' 类别或样本数不足的类别
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

            # ... (其余代码保持不变) ...

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

            # 记录激活数据
            for feat, lab in zip(features.detach(), combined_labels.detach()):
                activations[lab.item()].append(feat.cpu().numpy())

        epoch_loss = running_loss / len(data_loader)
        epoch_cls_loss = running_cls_loss / len(data_loader)
        epoch_metric_loss = running_metric_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}] Total Loss: {epoch_loss:.4f} | Cls Loss: {epoch_cls_loss:.4f} | Metric Loss: {epoch_metric_loss:.4f} | Accuracy: {epoch_acc:.2f}%')

        # 记录当前epoch的损失
        loss_history['total_loss'].append(epoch_loss)
        loss_history['cls_loss'].append(epoch_cls_loss)
        loss_history['metric_loss'].append(epoch_metric_loss)
        loss_history['accuracy'].append(epoch_acc)

    print('K-way 网络(含度量学习)训练完成。')

    # 保存激活数据
    with open(activation_save_path, 'wb') as f:
        pickle.dump(activations, f)
    print(f"激活数据已保存到 {activation_save_path}")

    # 绘制损失曲线
    plot_kway_losses(loss_history, save_dir='loss_curves_kway')

def plot_kway_losses(loss_history, save_dir='loss_curves_kway'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(loss_history['total_loss']) + 1)

    plt.figure(figsize=(12, 8))

    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_history['total_loss'], 'r-', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Total Loss')
    plt.legend()

    # 分类损失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss_history['cls_loss'], 'b-', label='Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Classification Loss')
    plt.legend()

    # 度量损失
    plt.subplot(2, 2, 3)
    plt.plot(epochs, loss_history['metric_loss'], 'g-', label='Metric Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('K-way Metric Loss')
    plt.legend()

    # 准确率
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
    labels = torch.randint(1, 24, (num_images,), device=device)
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



def   train_gan(generator, discriminator, data_loader, device, pixelcnn_pp, epochs=90, learning_rate=2e-4, nz=100,
              save_dir='generated_images', loss_dir='loss_curves_gan', epsilon=1e-4):

    import torch.nn.functional as F
    from model_convgan_MNIST import get_optimizer

    torch.autograd.set_detect_anomaly(True)
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_feature = torch.nn.MSELoss()  # 定义特征匹配损失函数

    optimizer_g = get_optimizer(generator, lr=learning_rate, beta1=0.5, beta2=0.999)
    optimizer_d = get_optimizer(discriminator, lr=learning_rate, beta1=0.5, beta2=0.999)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.64)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=15, gamma=0.8)

    # lambda_gp = 10  # 梯度惩罚的权重

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        progress_bar = tqdm(data_loader, desc=f'Epoch [{epoch + 1}/{epochs}]', unit='batch')
        for i, (real_images, labels) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device, dtype=torch.long)

            if labels.max() >= 24 or labels.min() < 1:
                continue

            ### 判别器训练 ###
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images_batch = generator(noise, labels).detach()
            outputs_real, _, _ = discriminator(real_images)
            outputs_fake, _, _ = discriminator(fake_images_batch)

            # 使用 Hinge Loss
            d_loss_real = torch.mean(F.relu(1.0 - outputs_real))
            d_loss_fake = torch.mean(F.relu(1.0 + outputs_fake))
            # gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images_batch, device, lambda_gp)
            d_loss = d_loss_real + d_loss_fake

            # 反向传播和优化判别器
            discriminator.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

            ### 生成器训练 ###
            generator.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(noise, labels)
            if torch.isnan(fake_images).any():
                continue

            # 获取生成样本的判别器输出和特征
            outputs_fake_gen, class_output, fake_features = discriminator(fake_images)
            if torch.isnan(outputs_fake_gen).any() or torch.isnan(class_output).any():
                continue

            # 对抗性损失（Hinge Loss）
            g_loss_gan = -torch.mean(outputs_fake_gen)

            # 分类损失
            g_loss_class = criterion_class(class_output, labels)

            # 低密度区域损失
            with torch.no_grad():
                l = pixelcnn_pp(fake_images)
                log_p_x = l.mean(dim=(1,2,3))
                p_x = torch.exp(log_p_x.unsqueeze(-1).unsqueeze(-1))

            mask = (p_x > epsilon).float()
            g_loss_low_density = -torch.mean(log_p_x * mask.squeeze())

            # 多样性损失
            diversity_loss = torch.mean((fake_images - fake_images.mean(dim=0))**2)

            # 获取真实样本的特征表示
            with torch.no_grad():
                _, _, real_features = discriminator(real_images)

            # 计算特征匹配损失
            feature_matching_loss = criterion_feature(real_features.mean(dim=0), fake_features.mean(dim=0))

            # 定义损失权重
            weight_gan = 1.0
            weight_class = 1.0
            weight_low_density = 1.0
            weight_feature_matching = 1.0  # 可以根据需要调整
            weight_diversity = 0.1  # 根据需要调整

            # 总生成器损失
            g_loss = (weight_gan * g_loss_gan +
                      weight_class * g_loss_class +
                      weight_low_density * g_loss_low_density +
                      weight_feature_matching * feature_matching_loss +
                      weight_diversity * diversity_loss)

            if torch.isnan(g_loss).any():
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            progress_bar.set_postfix({'Loss D': d_loss.item(), 'Loss G': g_loss.item()})

            if (i + 1) % 100 == 0:
                save_fake_images(epoch, generator, device, nz=nz, save_dir=save_dir)

        g_losses.append(g_loss_epoch / len(data_loader))
        d_losses.append(d_loss_epoch / len(data_loader))

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






def gradient_penalty(critic, real_imgs, fake_imgs, labels, device):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates.requires_grad_(True)
    
    # 修正后的梯度计算（已删除不可见字符）
    disc_interpolates = critic(interpolates, labels)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True, retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty=((gradients.norm(2,dim=1)-1)**2).mean()
    return gradient_penalty

def save_sample_images_cwgan_gp(epoch, fixed_noise, fixed_labels, generator, device, save_dir='generated_images_cwgan'):
    """
    保存生成的样本图像做可视化.
    你可随时在训练过程定期调用该函数。
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise.to(device), fixed_labels.to(device))
    grid = vutils.make_grid(fake_imgs.cpu(), nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f"Epoch {epoch}")
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()
    generator.train()

def train_cwgan_gp(generator, critic, dataloader, device,
                   nz=100,
                   num_epochs=50,
                   lr=1e-4,
                   beta1=0.5,
                   beta2=0.9,
                   lambda_gp=10,
                   n_critic=5):
    import torch
    import torch.nn.functional as F
    from torch import optim
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os

    # 优化器
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=lr)

    generator.train()
    critic.train()

    g_losses, c_losses = [], []

    # 固定噪声和标签，用于每个epoch可视化
    fixed_noise = torch.randn(64, nz, device=device)
    fixed_labels = torch.randint(low=0, high=24, size=(64,), device=device)

    # 定义梯度裁剪阈值
    critic_clip_value = 1.0   # Critic梯度最大范数
    gen_clip_value = 0.5      # 生成器梯度最大范数

    for epoch in range(num_epochs):
        # 用tqdm包装dataloader实现batch级别进度显示
        progress_bar = tqdm(dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')
        
        for i, (real_imgs, real_labels) in enumerate(progress_bar):
            real_imgs, real_labels = real_imgs.to(device), real_labels.to(device)
            
            # ================== 训练Critic ==================
            optimizer_C.zero_grad()
            
            # 生成假图像
            noise = torch.randn(real_imgs.size(0), nz, device=device)
            fake_imgs = generator(noise, real_labels)  # 这里假设使用真实标签生成
            fake_output = critic(fake_imgs.detach(), real_labels)
            
            # 计算Critic损失
            real_output = critic(real_imgs, real_labels)
            gp = gradient_penalty(critic, real_imgs, fake_imgs, real_labels, device)
            c_loss = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp * gp
            
            # 反向传播与梯度裁剪
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_clip_value)  # 关键修改点
            optimizer_C.step()
            
            # ================== 训练生成器 ==================
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                # 生成器前向
                gen_imgs = generator(noise, real_labels)
                g_output = critic(gen_imgs, real_labels)
                g_loss = -torch.mean(g_output)
                
                # 反向传播与梯度裁剪
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), gen_clip_value)  # 关键修改点
                optimizer_G.step()

            # 记录loss并显示在tqdm进度条上
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
            progress_bar.set_postfix({
                'C_Loss': f"{c_loss.item():.4f}",
                'G_Loss': f"{g_loss.item():.4f}"
            })

        # 每个epoch做可视化
        save_sample_images_cwgan_gp(epoch+1, fixed_noise, fixed_labels, generator, device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Critic Loss: {c_loss.item():.4f} | Generator Loss: {g_loss.item():.4f}")

    # 画训练损失曲线
    plt.figure()
    plt.plot(g_losses, label="G Loss")
    plt.plot(c_losses, label="C Loss")
    plt.legend()
    plt.title("cWGAN-GP Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("cWGAN_GP_loss_curve.png")
    plt.close()

    # 保存模型
    # torch.save(generator.state_dict(), "cWGAN_GP_Generator.pth")
    # torch.save(critic.state_dict(), "cWGAN_GP_Critic.pth")
    print("cWGAN-GP 模型训练完成，已保存。")
