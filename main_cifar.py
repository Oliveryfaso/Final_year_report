# main_cifar.py

import time
import pickle
import os
from data_loader_cifar import load_cifar10_data
from model_convgan_cifar import KwayNetwork, Generator, Discriminator, weights_init_normal
# main_cifar.py
from train_cifar import train_kway_network, train_gan, train_r3gan, train_cwgan_gp

from ots_cifar import evaluate_model
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pixelcnn_cifar import PixelCNN, discretized_mix_logistic_loss
from openmax_cifar import OpenMax
from train_pixelcnn_cifar import train_pixelcnn_cifar
from sklearn.metrics import classification_report
from model_r3gan import R3Generator, R3Discriminator


def save_results(metrics, timestamp, class_names, save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)

    # 保存文本结果
    result_filename = os.path.join(save_dir, f"result_{timestamp}.txt")
    with open(result_filename, 'w') as f:
        f.write(f"Training and Evaluation Results - {timestamp}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro Average): {metrics['precision']:.4f}\n")
        f.write(f"Recall (Macro Average): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score (Macro Average): {metrics['f1_score']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
        f.write("Classification Report:\n")
        f.write(classification_report_to_string(metrics['classification_report']))
    print(f"Results saved to {result_filename}")

    # 保存混淆矩阵图像
    confusion_matrix_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(metrics['confusion_matrix'], classes=class_names, save_path=confusion_matrix_path)
    print(f"混淆矩阵热力图已保存到 {confusion_matrix_path}。")

    # 保存各类别F1分数图像
    f1_scores_path = os.path.join(save_dir, f"class_f1_scores_{timestamp}.png")
    plot_class_f1_scores(metrics['classification_report'], classes=class_names, save_path=f1_scores_path)
    print(f"各类别的F1分数图已保存到 {f1_scores_path}。")


def classification_report_to_string(report):
    report_str = ""
    for class_label, metrics in report.items():
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            report_str += f"{class_label}: {metrics}\n"
        else:
            report_str += f"Class {class_label}:\n"
            for metric, value in metrics.items():
                report_str += f"  {metric}: {value}\n"
    return report_str


def plot_confusion_matrix(cm, classes, save_path='evaluation_results/confusion_matrix.png'):
    plt.figure(figsize=(12, 10))
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_class_f1_scores(class_report, classes, save_path='evaluation_results/class_f1_scores.png'):
    f1_scores = []
    for cls in classes:
        cls_str = str(cls)
        if cls_str in class_report:
            f1_scores.append(class_report[cls_str]['f1-score'])
        else:
            f1_scores.append(0.0)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=f1_scores, palette='viridis')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores per Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    torch.autograd.set_detect_anomaly(True)
    batch_size = 512
    known_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 已知标签为1-9
    unknown_label = 0  # 'airplane' 类别为未知类
    save_dir = './data/cifar10'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载数据集并划分已知集和开放集...")
    try:
        train_loader, test_loader, open_set_loader = load_cifar10_data(
            batch_size=batch_size,
            known_labels=known_labels,
            unknown_label=unknown_label,
            save_dir=save_dir,
            device=device
        )
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        return

    # 1. 训练 PixelCNN（如果需要的话）
    # print("训练 PixelCNN 模型...")
    # train_pixelcnn_cifar()
    print("PixelCNN 模型训练完成。（如已训练，这里暂不重复执行）")

    # 2. 初始化并加载 PixelCNN
    print("加载训练好的 PixelCNN 模型...")
    pixelcnn_pp = PixelCNN(
        nr_filters=160,  # 根据之前的修改
        nr_resnet=5,
        nr_logistic_mix=10,
        disable_third=False,
        dropout_p=0.3,  # 根据之前的修改
        n_channel=3,
        image_wh=32
    ).to(device)

    pixelcnn_pp_pth_path = './models/pixelcnn_pp.pth'
    if os.path.exists(pixelcnn_pp_pth_path):
        try:
            pixelcnn_pp.load_state_dict(torch.load(pixelcnn_pp_pth_path, map_location=device))
            pixelcnn_pp.eval()
            print("PixelCNN++ 模型已加载。")
        except RuntimeError as e:
            print(f"加载 PixelCNN++ 时发生错误: {e}")
            return
    else:
        print("PixelCNN++ 模型文件未找到，请确保已训练 PixelCNN++或修改路径。")
        return

    # 3. 初始化 GAN（不重新训练，只加载）
    print("初始化 GAN 网络...")
    generator_pth_path = './models/generator.pth'
    discriminator_pth_path = './models/discriminator.pth'

    generator = Generator(nz=100, ngf=64, nc=3, num_classes=10, image_size=32).to(device)
    discriminator = Discriminator(nc=3, ndf=64, num_classes=10, input_size=32).to(device)

    # 如果你还有 R3GAN 的文件，就保持在这里：
    # generator = R3Generator(nz=128, ngf=64).to(device)
    # discriminator = R3Discriminator(img_ch=3).to(device)

    # 如果你想加载它们曾经训练的权重，那么执行以下操作：
    if os.path.exists(generator_pth_path):
        generator.load_state_dict(torch.load(generator_pth_path, map_location=device))
        generator.eval()
        print("已加载预训练的 Generator.")
    else:
        print("generator.pth 不存在，使用随机初始化。")

    if os.path.exists(discriminator_pth_path):
        discriminator.load_state_dict(torch.load(discriminator_pth_path, map_location=device))
        discriminator.eval()
        print("已加载预训练的 Discriminator.")
    else:
        print("discriminator.pth 不存在，使用随机初始化。")

    # 这里原本用于训练GAN的代码，如果你确实不想重训，就注释掉即可
    print("开始训练 GAN 网络...")
    # train_gan(
    #     generator=generator,
    #     discriminator=discriminator,
    #     data_loader=train_loader,
    #     device=device,
    #     pixelcnn_pp=pixelcnn_pp,
    #     epochs=200,
    #     learning_rate=2e-4,
    #     nz=100,
    #     save_dir='generated_images',
    #     loss_dir='loss_curves_gan',
    #     epsilon=1e-4
    # )


    # train_cwgan_gp(
    #     generator=generator,
    #     discriminator=discriminator,
    #     data_loader=train_loader,
    #     device=device,
    #     nz=100,
    #     gp_lambda=10.0,
    #     n_critic=5,
    #     epochs=200,
    #     lr_g=1e-4,
    #     lr_d=4e-4
    # ) 
    # torch.save(generator.state_dict(), generator_pth_path)
    # torch.save(discriminator.state_dict(), discriminator_pth_path)
    
    print("GAN 网络训练完成，模型已保存。")

    # 4. 初始化并加载 K-way 网络（不重新训练，只加载）
    print("初始化 K-way 网络...")
    kway_pth_path = './models/kway_network.pth'
    kway_network = KwayNetwork(num_classes=10).to(device)

    # 同理，如果有 pre-trained 文件就加载
    if os.path.exists(kway_pth_path):
        kway_network.load_state_dict(torch.load(kway_pth_path, map_location=device))
        kway_network.eval()
        print("已加载预训练的 K-way 网络.")
    else:
        print("kway_network.pth 不存在，使用随机初始化。")

    # 如果想重训 K-way 网络，请取消注释
    print("开始训练 K-way 网络...")
    # activations_path = './models/activations.pkl'
    # train_kway_network(
    #     model=kway_network,
    #     data_loader=train_loader,
    #     generator=generator,
    #     device=device,
    #     epochs=50,
    #     learning_rate=1e-4,
    #     activation_save_path=activations_path,
    # )
    # torch.save(kway_network.state_dict(), kway_pth_path)
    print("K-way 网络训练完成，模型已保存。")

    # 5. 初始化 OpenMax
    activations_path = './models/activations.pkl'
    print("初始化 OpenMax...")
    openmax = OpenMax(
        activations_path=activations_path,
        tail_size=100,
        weibull_params_path='./models/weibulls.pkl',
        alpha=0.47,
    )
    print("OpenMax 初始化完成。")

    # 6. 生成用于 OpenMax 的未知类样本（如果要更新 Weibull 模型）
    print("生成用于 OpenMax 的未知类样本...")
    num_openmax_samples = 1000
    nz = generator.nz  # Generator 类中有 'nz' 属性
    fake_labels_openmax = torch.zeros(num_openmax_samples, dtype=torch.long, device=device)  # 'Unknown' 类别
    noise_openmax = torch.randn(num_openmax_samples, nz, 1, 1, device=device)

    kway_network.eval()
    generator.eval()

    with torch.no_grad():
        fake_images_openmax = generator(noise_openmax, fake_labels_openmax)
        # 假设要做简单的逆标准化
        fake_images_openmax = fake_images_openmax * 0.2023 + 0.4914
        fake_images_openmax = torch.clamp(fake_images_openmax, 0.0, 1.0)
        # 提取特征
        features_openmax, _ = kway_network(fake_images_openmax)

    features_openmax = features_openmax.cpu().numpy()

    # 添加到 OpenMax 激活数据 (class=0)
    for feat in features_openmax:
        openmax.activations[0].append(feat)

    # 重新拟合 Weibull 模型
    openmax._fit_weibulls()
    with open(openmax.weibull_params_path, 'wb') as f:
        pickle.dump({
            'weibull_models': openmax.weibull_models,
            'class_means': openmax.class_means
        }, f)
    print("OpenMax Weibull 模型已更新。")

    # 7. 评估模型
    print("评估模型在开放集和已知测试集上的表现...")
    try:
        threshold = 0.5
        print("开始评估模型...")
        metrics = evaluate_model(
            model=kway_network,
            known_test_loader=test_loader,
            open_set_loader=open_set_loader,
            device=device,
            threshold=threshold,
            openmax=openmax,
            discriminator=discriminator  # 用于双重验证
        )
        print("模型评估完成。")
    except AttributeError as e:
        print(f"评估模型时发生错误: {e}")
        return
    except Exception as e:
        print(f"评估模型时发生错误: {e}")
        return

    # 打印评估指标
    print("\n=== 评估指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率 (宏平均): {metrics['precision']:.4f}")
    print(f"召回率 (宏平均): {metrics['recall']:.4f}")
    print(f"F1 分数 (宏平均): {metrics['f1_score']:.4f}")
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("\n分类报告:")
    print(pd.DataFrame(metrics['classification_report']).transpose())

    # 保存结果
    class_names = ['Unknown', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_results(metrics, timestamp, class_names, save_dir='evaluation_results')

    print("训练和评估完成。")


if __name__ == "__main__":
    main()
